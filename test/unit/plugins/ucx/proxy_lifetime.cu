/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>
#include <nixl.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::fprintf(stderr, "Failed: %s at %d\n", #condition, __LINE__); \
            std::exit(1);                                                     \
        }                                                                     \
    } while (false)
#define NX(call) CHECK((call) == NIXL_SUCCESS)
#define CU(call) CHECK((call) == cudaSuccess)

static void
bytes(int fd, void *buffer, size_t size, bool send) {
    auto *ptr = static_cast<char *>(buffer);
    while (size) {
        const auto count = send ? write(fd, ptr, size) : read(fd, ptr, size);
        CHECK(count > 0);
        ptr += count;
        size -= count;
    }
}

static std::string
exchange(int fd, std::string own, bool parent) {
    std::string remote;
    for (bool send : {parent, !parent}) {
        uint64_t size = own.size();
        bytes(fd, &size, sizeof(size), send);
        CHECK(size < 16 * 1024 * 1024);
        if (!send) {
            remote.resize(size);
        }
        bytes(fd, send ? own.data() : remote.data(), size, send);
    }
    return remote;
}

__global__ void
submit(nixlMemViewH local, nixlMemViewH remote, nixl_status_t *status) {
    status[0] = nixlPut(nixlMemViewElem{local, 0, 0}, nixlMemViewElem{remote, 0, 0}, 64);
    status[1] = nixlAtomicAdd(7, nixlMemViewElem{remote, 0, 128});
}

int
main(int argc, char **argv) {
    const bool parent = argc == 1;
    int fd;
    pid_t child = -1;
    if (parent) {
        int sockets[2];
        CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        child = fork();
        CHECK(child >= 0);
        if (child == 0) {
            CHECK(prctl(PR_SET_PDEATHSIG, SIGKILL) == 0);
            close(sockets[0]);
            const auto arg = std::to_string(sockets[1]);
            execl("/proc/self/exe", argv[0], arg.c_str(), nullptr);
            _exit(1);
        }
        close(sockets[1]);
        fd = sockets[0];
    } else {
        fd = std::atoi(argv[1]);
    }
    CU(cudaSetDevice(0));
    nixlAgentConfig config;
    config.useProgThread = false;
    config.useListenThread = false;
    config.syncMode = std::getenv("NIXL_TEST_STRICT") ?
        nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT :
        nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    nixlAgent agent(parent ? "proxy-parent" : "proxy-peer", config);
    nixlBackendH *backend = nullptr;
    NX(agent.createBackend("UCX",
                           {{"device_proxy", "true"},
                            {"proxy_channel_count", "1"},
                            {"proxy_thread_count", "1"},
                            {"proxy_max_peers", "1"},
                            {"ucx_error_handling_mode", "none"}},
                           backend));
    void *buffer = nullptr;
    CU(cudaMalloc(&buffer, 4096));
    CU(cudaMemset(buffer, parent ? 0x5a : 0, 4096));
    CU(cudaDeviceSynchronize());
    nixl_reg_dlist_t reg(VRAM_SEG);
    reg.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer), 4096, 0, ""));
    NX(agent.registerMem(reg));
    std::string metadata, peer;
    NX(agent.getLocalMD(metadata));
    const auto remote_md = exchange(fd, metadata, parent);
    NX(agent.loadRemoteMD(remote_md, peer));
    uint64_t address = reinterpret_cast<uintptr_t>(buffer), remote_address;
    bytes(fd, &address, sizeof(address), true);
    bytes(fd, &remote_address, sizeof(remote_address), false);
    if (!parent) {
        for (unsigned round = 1; round <= 2; ++round) {
            char command;
            bytes(fd, &command, 1, false);
            unsigned char payload[136];
            CU(cudaMemcpy(payload, buffer, sizeof(payload), cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < 64; ++i) {
                CHECK(payload[i] == 0x5a);
            }
            uint64_t counter;
            std::memcpy(&counter, payload + 128, sizeof(counter));
            CHECK(counter == round * 7);
            bytes(fd, &command, 1, true);
        }
        char command;
        bytes(fd, &command, 1, false);
        bytes(fd, &command, 1, true);
        pause();
        return 1;
    }
    nixl_status_t *device_status = nullptr;
    CU(cudaMalloc(&device_status, 2 * sizeof(nixl_status_t)));
    for (unsigned round = 0; round < 3; ++round) {
        if (round) {
            NX(agent.loadRemoteMD(remote_md, peer));
        }
        nixl_local_dlist_t local(VRAM_SEG);
        local.addDesc(nixlBasicDesc(address, 4096, 0));
        nixl_remote_dlist_t remote(VRAM_SEG);
        remote.addDesc(nixlRemoteDesc(remote_address, 4096, 0, peer));
        nixlMemViewH src = nullptr, dst = nullptr;
        NX(agent.prepMemView(local, src));
        NX(agent.prepMemView(remote, dst));
        if (round == 2) {
            nixlMemViewH warmup = nullptr;
            NX(agent.prepMemView(local, warmup));
            submit<<<1, 1>>>(src, dst, device_status);
            CU(cudaDeviceSynchronize());
            agent.releaseMemView(warmup);
            char command = 'k';
            bytes(fd, &command, 1, true);
            bytes(fd, &command, 1, false);
            CHECK(kill(child, SIGKILL) == 0);
            int status;
            CHECK(waitpid(child, &status, 0) == child && WIFSIGNALED(status));
        }
        submit<<<1, 1>>>(src, dst, device_status);
        CU(cudaDeviceSynchronize());
        CU(cudaGetLastError());
        nixl_status_t status[2];
        CU(cudaMemcpy(status, device_status, sizeof(status), cudaMemcpyDeviceToHost));
        CHECK(status[0] == NIXL_IN_PROG && status[1] == NIXL_IN_PROG);
        NX(agent.invalidateRemoteMD(peer));
        const auto start = std::chrono::steady_clock::now();
        agent.releaseMemView(dst);
        agent.releaseMemView(src);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("round=%u killed=%u release_seconds=%.6f\n", round, round == 2, seconds);
        if (round < 2) {
            char command = 'v';
            bytes(fd, &command, 1, true);
            bytes(fd, &command, 1, false);
        }
    }
    NX(agent.deregisterMem(reg));
    CU(cudaFree(device_status));
    CU(cudaFree(buffer));
    close(fd);
    std::puts("PASS: payloads, atomic counts, invalidation, replacement, killed-peer release");
}
