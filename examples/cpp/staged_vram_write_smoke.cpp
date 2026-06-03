/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nixl.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char *kTargetAgent = "target";
constexpr const char *kInitiatorAgent = "initiator";
constexpr const char *kDone = "DONE_WRITE";

struct Args {
    std::string mode;
    std::string ip;
    int port = 5561;
    size_t bytes = 1024 * 1024;
    double timeout = 60.0;
    size_t chunkSize = 16 * 1024 * 1024;
    size_t slots = 4;
    std::string ucxDevices;
    std::string ucxTls = "rc,ud,self";
    bool staging = true;
};

[[noreturn]] void
usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --mode target|initiator [--ip target_ip] [--port port]"
              << " [--bytes n] [--timeout seconds] [--chunk-size n]"
              << " [--slots n] [--ucx-devices devs] [--ucx-tls tls]"
              << " [--staging 0|1]\n";
    std::exit(2);
}

Args
parseArgs(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key(argv[i]);
        auto needValue = [&]() -> std::string {
            if (++i >= argc) {
                usage(argv[0]);
            }
            return argv[i];
        };

        if (key == "--mode") {
            args.mode = needValue();
        } else if (key == "--ip") {
            args.ip = needValue();
        } else if (key == "--port") {
            args.port = std::stoi(needValue());
        } else if (key == "--bytes") {
            args.bytes = std::stoull(needValue());
        } else if (key == "--timeout") {
            args.timeout = std::stod(needValue());
        } else if (key == "--chunk-size") {
            args.chunkSize = std::stoull(needValue());
        } else if (key == "--slots") {
            args.slots = std::stoull(needValue());
        } else if (key == "--ucx-devices") {
            args.ucxDevices = needValue();
        } else if (key == "--ucx-tls") {
            args.ucxTls = needValue();
        } else if (key == "--staging") {
            args.staging = (needValue() != "0");
        } else {
            usage(argv[0]);
        }
    }

    if (args.mode != "target" && args.mode != "initiator") {
        usage(argv[0]);
    }
    if (args.mode == "initiator" && args.ip.empty()) {
        usage(argv[0]);
    }
    if (args.bytes == 0) {
        usage(argv[0]);
    }
    return args;
}

void
check(nixl_status_t status, const std::string &what) {
    if (status < NIXL_SUCCESS) {
        throw std::runtime_error(what + ": " + nixlEnumStrings::statusStr(status));
    }
}

void
checkCuda(cudaError_t status, const std::string &what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(what + ": " + cudaGetErrorString(status));
    }
}

template <class Predicate>
void
waitUntil(Predicate pred, const std::string &label, double timeoutSeconds) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(timeoutSeconds));
    while (true) {
        if (pred()) {
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("timeout waiting for " + label);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::vector<unsigned char>
makePattern(size_t bytes) {
    std::vector<unsigned char> pattern(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        pattern[i] = static_cast<unsigned char>((i * 131 + 17) & 0xff);
    }
    return pattern;
}

void *
allocDeviceBuffer(size_t bytes, const std::vector<unsigned char> *initial) {
    void *ptr = nullptr;
    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    if (initial) {
        checkCuda(cudaMemcpy(ptr, initial->data(), bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy host to device");
    } else {
        checkCuda(cudaMemset(ptr, 0xa5, bytes), "cudaMemset target");
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return ptr;
}

nixlBackendH *
createUcxBackend(nixlAgent &agent, const Args &args) {
    if (!args.ucxTls.empty()) {
        setenv("UCX_TLS", args.ucxTls.c_str(), 0);
    }

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    check(agent.getPluginParams("UCX", mems, params), "getPluginParams");
    params["vram_staging"] = args.staging ? "true" : "false";
    params["staging_chunk_size"] = std::to_string(args.chunkSize);
    params["staging_slots_per_gpu"] = std::to_string(args.slots);
    params["staging_force_progress_thread"] = "true";
    if (!args.ucxDevices.empty()) {
        params["ucx_devices"] = args.ucxDevices;
    }

    nixlBackendH *backend = nullptr;
    check(agent.createBackend("UCX", params, backend), "createBackend");
    return backend;
}

nixl_reg_dlist_t
makeRegDlist(void *ptr, size_t bytes) {
    nixlBlobDesc desc;
    desc.addr = reinterpret_cast<uintptr_t>(ptr);
    desc.len = bytes;
    desc.devId = 0;

    nixl_reg_dlist_t dlist(VRAM_SEG);
    dlist.addDesc(desc);
    return dlist;
}

nixl_xfer_dlist_t
makeXferDlist(const nixlBasicDesc &desc) {
    nixl_xfer_dlist_t dlist(VRAM_SEG);
    dlist.addDesc(desc);
    return dlist;
}

nixl_opt_args_t
backendOnly(nixlBackendH *backend) {
    nixl_opt_args_t extra;
    extra.backends.push_back(backend);
    return extra;
}

nixl_opt_args_t
socketArgs(const Args &args, nixlBackendH *backend) {
    nixl_opt_args_t extra = backendOnly(backend);
    extra.ipAddr = args.ip;
    extra.port = args.port;
    return extra;
}

void
runTarget(const Args &args) {
    nixlAgentConfig cfg(true, true, static_cast<uint16_t>(args.port));
    nixlAgent agent(kTargetAgent, cfg);
    nixlBackendH *backend = createUcxBackend(agent, args);
    nixl_opt_args_t extra = backendOnly(backend);

    void *dst = allocDeviceBuffer(args.bytes, nullptr);
    nixl_reg_dlist_t reg = makeRegDlist(dst, args.bytes);
    check(agent.registerMem(reg, &extra), "target registerMem");

    nixl_xfer_dlist_t empty(VRAM_SEG);
    waitUntil(
        [&]() { return agent.checkRemoteMD(kInitiatorAgent, empty) == NIXL_SUCCESS; },
        "initiator metadata",
        args.timeout);

    nixlBasicDesc dstDesc(reinterpret_cast<uintptr_t>(dst), args.bytes, 0);
    nixl_blob_t descMsg = std::string("DESC:") + dstDesc.serialize();
    check(agent.genNotif(kInitiatorAgent, descMsg, &extra), "target genNotif DESC");
    std::cout << "Target descriptors sent\n";

    nixl_notifs_t notifs;
    waitUntil(
        [&]() {
            check(agent.getNotifs(notifs, &extra), "target getNotifs");
            auto it = notifs.find(kInitiatorAgent);
            if (it == notifs.end()) {
                return false;
            }
            for (const auto &msg : it->second) {
                if (msg == kDone) {
                    return true;
                }
            }
            return false;
        },
        "write completion notification",
        args.timeout);

    std::vector<unsigned char> got(args.bytes);
    checkCuda(cudaMemcpy(got.data(), dst, args.bytes, cudaMemcpyDeviceToHost),
              "target cudaMemcpy device to host");
    checkCuda(cudaDeviceSynchronize(), "target cudaDeviceSynchronize");
    const auto expected = makePattern(args.bytes);
    for (size_t i = 0; i < args.bytes; ++i) {
        if (got[i] != expected[i]) {
            throw std::runtime_error("verification mismatch at byte " + std::to_string(i));
        }
    }

    std::cout << "Target verification passed: " << args.bytes << " bytes\n";
    check(agent.deregisterMem(reg, &extra), "target deregisterMem");
    checkCuda(cudaFree(dst), "target cudaFree");
}

void
runInitiator(const Args &args) {
    nixlAgentConfig cfg(true, true, 0);
    nixlAgent agent(kInitiatorAgent, cfg);
    nixlBackendH *backend = createUcxBackend(agent, args);
    nixl_opt_args_t extra = backendOnly(backend);

    const auto pattern = makePattern(args.bytes);
    void *src = allocDeviceBuffer(args.bytes, &pattern);
    nixl_reg_dlist_t reg = makeRegDlist(src, args.bytes);
    check(agent.registerMem(reg, &extra), "initiator registerMem");

    nixl_opt_args_t socket = socketArgs(args, backend);
    check(agent.fetchRemoteMD(kTargetAgent, &socket), "fetchRemoteMD target");
    nixl_xfer_dlist_t empty(VRAM_SEG);
    waitUntil(
        [&]() { return agent.checkRemoteMD(kTargetAgent, empty) == NIXL_SUCCESS; },
        "target metadata",
        args.timeout);
    check(agent.sendLocalMD(&socket), "sendLocalMD initiator");

    nixlBasicDesc targetDesc;
    bool haveDesc = false;
    nixl_notifs_t notifs;
    waitUntil(
        [&]() {
            check(agent.getNotifs(notifs, &extra), "initiator getNotifs");
            auto it = notifs.find(kTargetAgent);
            if (it == notifs.end()) {
                return false;
            }
            for (const auto &msg : it->second) {
                if (msg.rfind("DESC:", 0) == 0) {
                    targetDesc = nixlBasicDesc(msg.substr(5));
                    haveDesc = true;
                    return true;
                }
            }
            return false;
        },
        "target descriptors",
        args.timeout);
    if (!haveDesc) {
        throw std::runtime_error("target descriptor was not received");
    }

    nixlBasicDesc srcDesc(reinterpret_cast<uintptr_t>(src), args.bytes, 0);
    nixl_xfer_dlist_t local = makeXferDlist(srcDesc);
    nixl_xfer_dlist_t remote = makeXferDlist(targetDesc);

    nixl_opt_args_t xferExtra = backendOnly(backend);
    xferExtra.notif = kDone;

    nixlXferReqH *handle = nullptr;
    check(agent.createXferReq(NIXL_WRITE, local, remote, kTargetAgent, handle, &xferExtra),
          "createXferReq");
    nixlBackendH *chosen = nullptr;
    check(agent.queryXferBackend(handle, chosen), "queryXferBackend");
    std::cout << "Transfer backend: UCX\n";

    nixl_status_t status = agent.postXferReq(handle);
    if (status < NIXL_SUCCESS) {
        check(status, "postXferReq");
    }
    waitUntil(
        [&]() {
            if (status == NIXL_SUCCESS) {
                return true;
            }
            status = agent.getXferStatus(handle);
            if (status < NIXL_SUCCESS) {
                check(status, "getXferStatus");
            }
            return status == NIXL_SUCCESS;
        },
        "transfer completion",
        args.timeout);

    std::cout << "Initiator WRITE completed: " << args.bytes << " bytes\n";
    check(agent.releaseXferReq(handle), "releaseXferReq");
    check(agent.invalidateRemoteMD(kTargetAgent), "invalidateRemoteMD");
    check(agent.deregisterMem(reg, &extra), "initiator deregisterMem");
    checkCuda(cudaFree(src), "initiator cudaFree");
}

} // namespace

int
main(int argc, char **argv) {
    try {
        const Args args = parseArgs(argc, argv);
        if (args.mode == "target") {
            runTarget(args);
        } else {
            runInitiator(args);
        }
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "staged_vram_write_smoke failed: " << e.what() << "\n";
        return 1;
    }
}
