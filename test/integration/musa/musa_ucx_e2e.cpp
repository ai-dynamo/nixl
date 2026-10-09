// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Hardware-only smoke test: two processes, metadata exchange, WRITE+notification, READ.
#include <nixl.h>
#include <musa_runtime_api.h>

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
[[noreturn]] void
fail(const std::string &message) {
    std::cerr << "FAIL: " << message << std::endl;
    // Do not unwind/free allocations that could still be referenced by a failed async operation.
    std::_Exit(1);
}

void
require(bool condition, const std::string &message) {
    if (!condition) {
        fail(message);
    }
}

void
check(nixl_status_t status, const char *operation) {
    require(status == NIXL_SUCCESS, std::string(operation) + ": status " + std::to_string(status));
}

void
checkSdk(musaError_t status, const char *operation) {
    require(status == musaSuccess, std::string(operation) + ": " + musaGetErrorString(status));
}

void
writeAll(int fd, const void *data, size_t length) {
    auto ptr = static_cast<const char *>(data);
    while (length) {
        const auto n = write(fd, ptr, length);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        require(n > 0, "control write failed or timed out");
        ptr += n;
        length -= static_cast<size_t>(n);
    }
}

void
readAll(int fd, void *data, size_t length) {
    auto ptr = static_cast<char *>(data);
    while (length) {
        const auto n = read(fd, ptr, length);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        require(n > 0, "peer exited, control read failed or timed out");
        ptr += n;
        length -= static_cast<size_t>(n);
    }
}

void
sendText(int fd, const std::string &text) {
    const uint64_t length = text.size();
    writeAll(fd, &length, sizeof(length));
    writeAll(fd, text.data(), text.size());
}

std::string
receiveText(int fd) {
    uint64_t length = 0;
    readAll(fd, &length, sizeof(length));
    require(length <= 4 * 1024 * 1024, "control frame too large");
    std::string text(length, '\0');
    readAll(fd, text.data(), text.size());
    return text;
}

std::string
exchange(int fd, int rank, const std::string &text) {
    if (rank == 0) {
        sendText(fd, text);
        return receiveText(fd);
    }
    auto peer = receiveText(fd);
    sendText(fd, text);
    return peer;
}

struct Buffer {
    std::vector<unsigned char> host;
    void *address = nullptr;
    bool device;

    Buffer(size_t length, bool on_device) : host(length), device(on_device) {
        address = host.data();
        if (device) {
            checkSdk(musaMalloc(&address, length), "musaMalloc");
        }
    }

    ~Buffer() {
        if (device) {
            musaFree(address);
        }
    }

    void
    fill(unsigned salt) {
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = static_cast<unsigned char>((i * 131 + (i >> 8) + salt) % 251);
        }
        if (device) {
            checkSdk(musaMemcpy(address, host.data(), host.size(), musaMemcpyHostToDevice),
                     "upload");
        }
        checkSdk(musaDeviceSynchronize(), "producer synchronize");
    }

    void
    verify(unsigned salt) {
        if (device) {
            checkSdk(musaMemcpy(host.data(), address, host.size(), musaMemcpyDeviceToHost),
                     "download");
        }
        for (size_t i = 0; i < host.size(); ++i) {
            require(host[i] == static_cast<unsigned char>((i * 131 + (i >> 8) + salt) % 251),
                    "payload mismatch at offset " + std::to_string(i));
        }
    }
};

void
waitNotification(nixlAgent &agent, const std::string &peer, const std::string &expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        nixl_notifs_t notifications;
        check(agent.getNotifs(notifications), "getNotifs");
        for (const auto &message : notifications[peer]) {
            if (message == expected) {
                return;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    fail("notification timeout");
}

void
transfer(nixlAgent &agent,
         nixl_xfer_op_t operation,
         const nixl_xfer_dlist_t &local,
         const nixl_xfer_dlist_t &remote,
         const std::string &peer,
         nixl_opt_args_t options) {
    nixlXferReqH *request = nullptr;
    check(agent.createXferReq(operation, local, remote, peer, request, &options), "createXferReq");
    auto status = agent.postXferReq(request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
        status = agent.getXferStatus(request);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    check(status, "transfer completion");
    check(agent.releaseXferReq(request), "releaseXferReq");
}

void
runRank(int rank, int fd, bool device_a, bool device_b, size_t bytes, int threads, int device_id) {
    alarm(90);
    timeval timeout{30, 0};
    require(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0,
            "recv timeout");
    require(setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0,
            "send timeout");
    checkSdk(musaSetDevice(device_id), "musaSetDevice");
    nixlAgentConfig config;
    config.useProgThread = true;
    config.useListenThread = false;
    nixlAgent agent("musa_hw_" + std::to_string(rank), config);
    nixlBackendH *backend = nullptr;
    check(agent.createBackend("MUSA_UCX", {{"num_threads", std::to_string(threads)}}, backend),
          "createBackend MUSA_UCX (missing provider is a failure, not a skip)");
    Buffer buffer(bytes, rank == 0 ? device_a : device_b);
    buffer.fill(rank == 0 ? 17 : 0);
    nixl_opt_args_t options;
    options.backends = {backend};
    const auto local_type = buffer.device ? VRAM_SEG : DRAM_SEG;
    const auto remote_type = (rank == 0 ? device_b : device_a) ? VRAM_SEG : DRAM_SEG;
    const auto address = reinterpret_cast<uintptr_t>(buffer.address);
    nixl_reg_dlist_t registration(local_type);
    registration.addDesc(nixlBlobDesc(address, bytes, device_id, ""));
    check(agent.registerMem(registration, &options), "registerMem");
    nixl_blob_t metadata;
    check(agent.getLocalMD(metadata), "getLocalMD");
    std::string peer;
    check(agent.loadRemoteMD(exchange(fd, rank, metadata), peer), "loadRemoteMD");
    const auto peer_address = std::stoull(exchange(fd, rank, std::to_string(address)));
    const auto peer_device = std::stoull(exchange(fd, rank, std::to_string(device_id)));
    nixl_xfer_dlist_t local(local_type), remote(remote_type);
    local.addDesc(nixlBasicDesc(address, bytes, device_id));
    remote.addDesc(nixlBasicDesc(peer_address, bytes, peer_device));
    check(agent.makeConnection(peer, &options), "makeConnection");

    if (rank == 0) {
        options.hasNotif = true;
        options.notifMsg = "write-complete";
        transfer(agent, NIXL_WRITE, local, remote, peer, options);
        require(receiveText(fd) == "read-ready", "WRITE not verified by peer");
        options.hasNotif = false;
        transfer(agent, NIXL_READ, local, remote, peer, options);
        buffer.verify(93);
        sendText(fd, "read-verified");
    } else {
        waitNotification(agent, peer, "write-complete");
        buffer.verify(17);
        buffer.fill(93);
        sendText(fd, "read-ready");
        require(receiveText(fd) == "read-verified", "READ not verified by peer");
    }
    (void)exchange(fd, rank, "finished");
    check(agent.invalidateRemoteMD(peer), "invalidateRemoteMD");
    check(agent.deregisterMem(registration, &options), "deregisterMem");
    close(fd);
}
} // namespace

int
main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "Usage: musa_ucx_e2e <host|device A> <host|device B> <bytes> "
                     "<threads> <device A> <device B>\n";
        return 1;
    }
    require((std::string(argv[1]) == "host" || std::string(argv[1]) == "device") &&
                (std::string(argv[2]) == "host" || std::string(argv[2]) == "device"),
            "memory kinds");
    try {
        const auto bytes = std::stoull(argv[3]);
        const int threads = std::stoi(argv[4]);
        const int device_a = std::stoi(argv[5]), device_b = std::stoi(argv[6]);
        require(bytes > 0 && bytes <= 64 * 1024 * 1024 && threads >= 0 && device_a >= 0 &&
                    device_b >= 0,
                "invalid size, threads or device");
        int sockets[2];
        require(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair");
        const auto child = fork(); // Initialize MUSA/UCX only after fork.
        require(child >= 0, "fork");
        const int rank = child == 0 ? 1 : 0;
        close(sockets[1 - rank]);
        runRank(rank,
                sockets[rank],
                std::string(argv[1]) == "device",
                std::string(argv[2]) == "device",
                bytes,
                threads,
                rank == 0 ? device_a : device_b);
        if (child == 0) {
            alarm(0);
            return 0;
        }
        int status = 0;
        require(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                    WEXITSTATUS(status) == 0,
                "peer failed");
        std::cout << "PASS: two processes, byte-checked WRITE+notification and READ\n";
        alarm(0);
        return 0;
    }
    catch (const std::exception &error) {
        fail(error.what());
    }
}
