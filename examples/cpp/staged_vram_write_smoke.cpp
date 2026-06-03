/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nixl.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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
    size_t concurrency = 1;
    size_t iters = 1;
    size_t offsetStride = 0;
    size_t descriptors = 1;
    uint64_t initiatorId = 0;
    size_t initiatorCount = 1;
    size_t pollSleepUs = 10000;
    std::string ucxDevices;
    std::string ucxTls = "rc,ud,self";
    bool staging = true;
    bool prepped = false;
    bool skipDescMerge = false;
};

[[noreturn]] void
usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --mode target|initiator [--ip target_ip] [--port port]"
              << " [--bytes n] [--timeout seconds] [--chunk-size n]"
              << " [--slots n] [--concurrency n] [--iters n]"
              << " [--offset-stride n] [--descriptors n]"
              << " [--initiator-id n] [--initiator-count n]"
              << " [--poll-sleep-us n]"
              << " [--ucx-devices devs] [--ucx-tls tls] [--staging 0|1]"
              << " [--prepped 0|1] [--skip-desc-merge 0|1]\n";
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
        } else if (key == "--concurrency") {
            args.concurrency = std::stoull(needValue());
        } else if (key == "--iters") {
            args.iters = std::stoull(needValue());
        } else if (key == "--offset-stride") {
            args.offsetStride = std::stoull(needValue());
        } else if (key == "--descriptors") {
            args.descriptors = std::stoull(needValue());
        } else if (key == "--initiator-id") {
            args.initiatorId = std::stoull(needValue());
        } else if (key == "--initiator-count") {
            args.initiatorCount = std::stoull(needValue());
        } else if (key == "--poll-sleep-us") {
            args.pollSleepUs = std::stoull(needValue());
        } else if (key == "--ucx-devices") {
            args.ucxDevices = needValue();
        } else if (key == "--ucx-tls") {
            args.ucxTls = needValue();
        } else if (key == "--staging") {
            args.staging = (needValue() != "0");
        } else if (key == "--prepped") {
            args.prepped = (needValue() != "0");
        } else if (key == "--skip-desc-merge") {
            args.skipDescMerge = (needValue() != "0");
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
    if (args.bytes == 0 || args.concurrency == 0 || args.iters == 0 ||
        args.initiatorCount == 0 || args.descriptors == 0 ||
        args.descriptors > args.bytes ||
        args.descriptors > static_cast<size_t>(std::numeric_limits<int>::max())) {
        usage(argv[0]);
    }
    if (args.offsetStride != 0 && args.offsetStride < args.bytes) {
        usage(argv[0]);
    }
    if (args.mode == "initiator" && args.initiatorCount > 1 &&
        (args.initiatorId == 0 || args.initiatorId > args.initiatorCount)) {
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
waitUntil(Predicate pred, const std::string &label, double timeoutSeconds, size_t pollSleepUs) {
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
        if (pollSleepUs != 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(pollSleepUs));
        }
    }
}

size_t
checkedMul(size_t a, size_t b, const std::string &what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error("size overflow while computing " + what);
    }
    return a * b;
}

size_t
checkedAdd(size_t a, size_t b, const std::string &what) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error("size overflow while computing " + what);
    }
    return a + b;
}

size_t
effectiveOffsetStride(const Args &args) {
    return args.offsetStride == 0 ? args.bytes : args.offsetStride;
}

size_t
localTransferCount(const Args &args) {
    return checkedMul(args.concurrency, args.iters, "local transfer count");
}

size_t
totalTransferCount(const Args &args) {
    return checkedMul(localTransferCount(args), args.initiatorCount, "total transfer count");
}

size_t
regionBytesForCount(const Args &args, size_t count) {
    const size_t stride = effectiveOffsetStride(args);
    const size_t lastOffset = checkedMul(count - 1, stride, "last transfer offset");
    return checkedAdd(lastOffset, args.bytes, "registered GPU region size");
}

size_t
localRegionBytes(const Args &args) {
    return regionBytesForCount(args, localTransferCount(args));
}

size_t
targetRegionBytes(const Args &args) {
    return regionBytesForCount(args, totalTransferCount(args));
}

size_t
descriptorOffset(const Args &args, size_t descIndex) {
    const size_t base = args.bytes / args.descriptors;
    const size_t rem = args.bytes % args.descriptors;
    return descIndex * base + std::min(descIndex, rem);
}

size_t
descriptorLength(const Args &args, size_t descIndex) {
    const size_t base = args.bytes / args.descriptors;
    const size_t rem = args.bytes % args.descriptors;
    return base + (descIndex < rem ? 1 : 0);
}

std::string
initiatorAgentName(uint64_t initiator_id) {
    if (initiator_id == 0) {
        return kInitiatorAgent;
    }
    return std::string(kInitiatorAgent) + "-" + std::to_string(initiator_id);
}

std::string
initiatorAgentName(const Args &args) {
    return initiatorAgentName(args.initiatorId);
}

std::vector<uint64_t>
expectedInitiatorIds(const Args &args) {
    std::vector<uint64_t> ids;
    ids.reserve(args.initiatorCount);
    if (args.initiatorCount == 1) {
        ids.push_back(args.initiatorId);
        return ids;
    }

    for (size_t i = 1; i <= args.initiatorCount; ++i) {
        ids.push_back(static_cast<uint64_t>(i));
    }
    return ids;
}

size_t
initiatorSlotIndex(const Args &args, uint64_t initiator_id) {
    if (args.initiatorCount == 1) {
        return 0;
    }
    if (initiator_id == 0 || initiator_id > args.initiatorCount) {
        throw std::runtime_error("invalid initiator id for multi-initiator run");
    }
    return static_cast<size_t>(initiator_id - 1);
}

size_t
globalTransferIndex(const Args &args, uint64_t initiator_id, size_t local_transfer) {
    return checkedAdd(checkedMul(initiatorSlotIndex(args, initiator_id),
                                 localTransferCount(args),
                                 "initiator transfer base"),
                      local_transfer,
                      "global transfer index");
}

unsigned char
patternByte(size_t byteIndex, size_t transferIndex, uint64_t initiatorId) {
    return static_cast<unsigned char>(
        (byteIndex * 131 + transferIndex * 17 + initiatorId * 29 + 17) & 0xff);
}

std::vector<unsigned char>
makePattern(size_t bytes, size_t transferIndex = 0, uint64_t initiatorId = 0) {
    std::vector<unsigned char> pattern(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        pattern[i] = patternByte(i, transferIndex, initiatorId);
    }
    return pattern;
}

std::vector<unsigned char>
makeSourceRegion(const Args &args) {
    std::vector<unsigned char> pattern(localRegionBytes(args), 0);
    const size_t stride = effectiveOffsetStride(args);
    const size_t count = localTransferCount(args);
    for (size_t transfer = 0; transfer < count; ++transfer) {
        const size_t offset = transfer * stride;
        for (size_t i = 0; i < args.bytes; ++i) {
            pattern[offset + i] = patternByte(i, transfer, args.initiatorId);
        }
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
makeXferDlist(uintptr_t baseAddr, size_t transferOffset, uint64_t devId, const Args &args) {
    nixl_xfer_dlist_t dlist(VRAM_SEG);
    for (size_t desc = 0; desc < args.descriptors; ++desc) {
        const size_t desc_offset = descriptorOffset(args, desc);
        const size_t desc_len = descriptorLength(args, desc);
        nixlBasicDesc piece(baseAddr + transferOffset + desc_offset, desc_len, devId);
        dlist.addDesc(piece);
    }
    return dlist;
}

std::vector<int>
makeDescriptorIndices(const Args &args) {
    std::vector<int> indices;
    indices.reserve(args.descriptors);
    for (size_t desc = 0; desc < args.descriptors; ++desc) {
        indices.push_back(static_cast<int>(desc));
    }
    return indices;
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

    const std::vector<uint64_t> initiatorIds = expectedInitiatorIds(args);
    std::vector<std::string> initiatorNames;
    initiatorNames.reserve(initiatorIds.size());
    for (const uint64_t initiator_id : initiatorIds) {
        initiatorNames.push_back(initiatorAgentName(initiator_id));
    }

    const size_t localCount = localTransferCount(args);
    const size_t totalCount = totalTransferCount(args);
    const size_t stride = effectiveOffsetStride(args);
    const size_t totalBytes = targetRegionBytes(args);

    void *dst = allocDeviceBuffer(totalBytes, nullptr);
    nixl_reg_dlist_t reg = makeRegDlist(dst, totalBytes);
    check(agent.registerMem(reg, &extra), "target registerMem");

    nixl_xfer_dlist_t empty(VRAM_SEG);
    for (const std::string &initiatorName : initiatorNames) {
        waitUntil(
            [&]() { return agent.checkRemoteMD(initiatorName, empty) == NIXL_SUCCESS; },
            "initiator metadata",
            args.timeout,
            args.pollSleepUs);
    }

    nixlBasicDesc dstDesc(reinterpret_cast<uintptr_t>(dst), totalBytes, 0);
    nixl_blob_t descMsg = std::string("DESC:") + dstDesc.serialize();
    for (const std::string &initiatorName : initiatorNames) {
        check(agent.genNotif(initiatorName, descMsg, &extra), "target genNotif DESC");
    }
    std::cout << "Target descriptors sent to " << initiatorNames.size() << " initiator(s)\n";

    size_t doneCount = 0;
    waitUntil(
        [&]() {
            nixl_notifs_t notifs;
            check(agent.getNotifs(notifs, &extra), "target getNotifs");
            for (const std::string &initiatorName : initiatorNames) {
                auto it = notifs.find(initiatorName);
                if (it == notifs.end()) {
                    continue;
                }
                for (const auto &msg : it->second) {
                    if (msg == kDone) {
                        ++doneCount;
                    }
                }
            }
            return doneCount >= totalCount;
        },
        "write completion notification",
        args.timeout,
        args.pollSleepUs);

    std::vector<unsigned char> got(totalBytes);
    checkCuda(cudaMemcpy(got.data(), dst, totalBytes, cudaMemcpyDeviceToHost),
              "target cudaMemcpy device to host");
    checkCuda(cudaDeviceSynchronize(), "target cudaDeviceSynchronize");
    for (const uint64_t initiatorId : initiatorIds) {
        for (size_t transfer = 0; transfer < localCount; ++transfer) {
            const size_t globalTransfer = globalTransferIndex(args, initiatorId, transfer);
            const size_t offset = globalTransfer * stride;
            for (size_t i = 0; i < args.bytes; ++i) {
                const unsigned char expected = patternByte(i, transfer, initiatorId);
                if (got[offset + i] != expected) {
                    throw std::runtime_error("verification mismatch at initiator " +
                                             std::to_string(initiatorId) + " transfer " +
                                             std::to_string(transfer) + " byte " +
                                             std::to_string(i));
                }
            }
        }
    }

    std::cout << "Target verification passed: " << totalCount << " transfers, " << args.bytes
              << " bytes each, " << initiatorNames.size() << " initiator(s)\n";
    check(agent.deregisterMem(reg, &extra), "target deregisterMem");
    checkCuda(cudaFree(dst), "target cudaFree");
}

void
runInitiator(const Args &args) {
    nixlAgentConfig cfg(true, true, 0);
    const std::string initiatorName = initiatorAgentName(args);
    nixlAgent agent(initiatorName, cfg);
    nixlBackendH *backend = createUcxBackend(agent, args);
    nixl_opt_args_t extra = backendOnly(backend);

    const size_t count = localTransferCount(args);
    const size_t stride = effectiveOffsetStride(args);
    const size_t totalBytes = localRegionBytes(args);

    const auto pattern = makeSourceRegion(args);
    void *src = allocDeviceBuffer(totalBytes, &pattern);
    nixl_reg_dlist_t reg = makeRegDlist(src, totalBytes);
    check(agent.registerMem(reg, &extra), "initiator registerMem");

    nixl_opt_args_t socket = socketArgs(args, backend);
    check(agent.fetchRemoteMD(kTargetAgent, &socket), "fetchRemoteMD target");
    nixl_xfer_dlist_t empty(VRAM_SEG);
    waitUntil(
        [&]() { return agent.checkRemoteMD(kTargetAgent, empty) == NIXL_SUCCESS; },
        "target metadata",
        args.timeout,
        args.pollSleepUs);
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
        args.timeout,
        args.pollSleepUs);
    if (!haveDesc) {
        throw std::runtime_error("target descriptor was not received");
    }

    bool printedBackend = false;
    size_t completedTransfers = 0;
    for (size_t iter = 0; iter < args.iters; ++iter) {
        std::vector<nixlXferReqH *> handles(args.concurrency, nullptr);
        std::vector<nixl_status_t> statuses(args.concurrency, NIXL_IN_PROG);
        std::vector<nixlDlistH *> localPreps(args.concurrency, nullptr);
        std::vector<nixlDlistH *> remotePreps(args.concurrency, nullptr);

        for (size_t lane = 0; lane < args.concurrency; ++lane) {
            const size_t transfer = iter * args.concurrency + lane;
            const size_t localOffset = transfer * stride;
            const size_t remoteOffset =
                globalTransferIndex(args, args.initiatorId, transfer) * stride;

            nixl_xfer_dlist_t local =
                makeXferDlist(reinterpret_cast<uintptr_t>(src), localOffset, 0, args);
            nixl_xfer_dlist_t remote =
                makeXferDlist(targetDesc.addr, remoteOffset, targetDesc.devId, args);

            nixl_opt_args_t xferExtra = backendOnly(backend);
            xferExtra.notif = kDone;
            xferExtra.skipDescMerge = args.skipDescMerge;

            if (args.prepped) {
                check(agent.prepXferDlist(local, localPreps[lane], &extra),
                      "prepXferDlist local");
                check(agent.prepXferDlist(kTargetAgent, remote, remotePreps[lane], &extra),
                      "prepXferDlist remote");
                const std::vector<int> indices = makeDescriptorIndices(args);
                check(agent.makeXferReq(NIXL_WRITE, localPreps[lane], indices, remotePreps[lane],
                                        indices, handles[lane], &xferExtra),
                      "makeXferReq");
            } else {
                check(agent.createXferReq(NIXL_WRITE, local, remote, kTargetAgent, handles[lane],
                                          &xferExtra),
                      "createXferReq");
            }
            if (!printedBackend) {
                nixlBackendH *chosen = nullptr;
                check(agent.queryXferBackend(handles[lane], chosen), "queryXferBackend");
                std::cout << "Transfer backend: UCX\n";
                printedBackend = true;
            }

            statuses[lane] = agent.postXferReq(handles[lane]);
            if (statuses[lane] < NIXL_SUCCESS) {
                check(statuses[lane], "postXferReq");
            }
        }

        waitUntil(
            [&]() {
                bool allDone = true;
                for (size_t lane = 0; lane < args.concurrency; ++lane) {
                    if (statuses[lane] == NIXL_SUCCESS) {
                        continue;
                    }
                    statuses[lane] = agent.getXferStatus(handles[lane]);
                    if (statuses[lane] < NIXL_SUCCESS) {
                        check(statuses[lane], "getXferStatus");
                    }
                    allDone = allDone && statuses[lane] == NIXL_SUCCESS;
                }
                return allDone;
            },
            "transfer completion",
            args.timeout,
            args.pollSleepUs);

        for (auto *handle : handles) {
            check(agent.releaseXferReq(handle), "releaseXferReq");
        }
        for (auto *handle : localPreps) {
            if (handle) {
                check(agent.releasedDlistH(handle), "release local dlist handle");
            }
        }
        for (auto *handle : remotePreps) {
            if (handle) {
                check(agent.releasedDlistH(handle), "release remote dlist handle");
            }
        }
        completedTransfers += args.concurrency;
    }

    std::cout << "Initiator WRITE completed: " << completedTransfers << " transfers, "
              << args.bytes << " bytes each, " << args.descriptors
              << " descriptor(s) each, prepped=" << (args.prepped ? 1 : 0)
              << ", skip_desc_merge=" << (args.skipDescMerge ? 1 : 0) << "\n";
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
