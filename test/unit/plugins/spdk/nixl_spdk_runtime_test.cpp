/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runtime test for the SPDK plugin. It spins up a private SPDK runtime over an
// in-memory malloc bdev (no hugepages / no PCI, so it can run unprivileged) and
// drives real I/O through the public NIXL API to exercise the request-handle
// lifetime logic: a write/read round trip, a repost of a completed handle, a
// burst of release-while-in-flight cancellations, and a second backend sharing
// the process-wide SPDK runtime with the first.
//
// If the SPDK environment cannot be brought up (e.g. a CI box with no usable
// memory backing), the test prints SKIP and exits 0 rather than failing.

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>

#include "nixl.h"

namespace {

constexpr const char *kAgentName = "spdk-runtime-test";
constexpr const char *kBdevName = "Malloc0";
constexpr uint32_t kBlockSize = 512;
constexpr uint64_t kNumBlocks = 8192; // 4 MiB malloc bdev
constexpr uint64_t kBdevBytes = kBlockSize * kNumBlocks;
constexpr uint64_t kDevId = 1; // user-chosen handle tying DRAM <-> bdev
constexpr size_t kIoSize = 64 * 1024; // 128 blocks
constexpr uint64_t kBdevOffset = 1 << 20; // write at 1 MiB into the bdev

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return -__LINE__;                                                  \
        }                                                                      \
    } while (0)

#define CHECK_OK(status, msg) CHECK((status) == NIXL_SUCCESS, msg)

void
fillPattern(void *buf, size_t len, uint8_t seed) {
    auto *p = static_cast<uint8_t *>(buf);
    for (size_t i = 0; i < len; ++i) {
        p[i] = static_cast<uint8_t>((i * 31 + seed) & 0xff);
    }
}

bool
verifyPattern(const void *buf, size_t len, uint8_t seed) {
    const auto *p = static_cast<const uint8_t *>(buf);
    for (size_t i = 0; i < len; ++i) {
        if (p[i] != static_cast<uint8_t>((i * 31 + seed) & 0xff)) {
            return false;
        }
    }
    return true;
}

// Drive a single transfer to completion via the busy-poll status loop.
nixl_status_t
runToCompletion(nixlAgent &agent, nixlXferReqH *req) {
    nixl_status_t status = agent.postXferReq(req);
    if (status != NIXL_IN_PROG && status != NIXL_SUCCESS) {
        return status;
    }
    do {
        status = agent.getXferStatus(req);
    } while (status == NIXL_IN_PROG);
    return status;
}

nixl_b_params_t
makeParams(const nixl_b_params_t &base, const char *bdevName, const char *spdkName) {
    nixl_b_params_t params = base;
    params["bdev_type"] = "malloc";
    params["bdev_name"] = bdevName;
    params["bdev_num_blocks"] = std::to_string(kNumBlocks);
    params["bdev_block_size"] = std::to_string(kBlockSize);
    params["spdk_name"] = spdkName;
    // The default SPDK message mempool (256K entries) is far more than this test
    // needs; a few thousand entries keeps the footprint small.
    params["msg_mempool_size"] = "4095";
    return params;
}

int
runTest(nixlAgent &agent, nixlBackendH *backend, void *dram, size_t dramLen) {
    nixl_opt_args_t backendParams;
    backendParams.backends.push_back(backend);

    // Register the DRAM staging buffer and the bdev (by name, via metaInfo).
    nixl_reg_dlist_t dramReg(DRAM_SEG);
    dramReg.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(dram), dramLen, kDevId));
    CHECK_OK(agent.registerMem(dramReg, &backendParams), "register DRAM");

    nixl_reg_dlist_t bdevReg(BLK_SEG);
    bdevReg.addDesc(nixlBlobDesc(0, kBdevBytes, kDevId, kBdevName));
    CHECK_OK(agent.registerMem(bdevReg, &backendParams), "register bdev");

    // --- Phase 1: write / read / verify round trip ---
    fillPattern(dram, kIoSize, 0xa5);

    nixl_xfer_dlist_t wsrc(DRAM_SEG), wdst(BLK_SEG);
    wsrc.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(dram), kIoSize, kDevId));
    wdst.addDesc(nixlBasicDesc(kBdevOffset, kIoSize, kDevId));

    nixlXferReqH *wreq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_WRITE, wsrc, wdst, kAgentName, wreq, &backendParams),
             "create write req");
    CHECK_OK(runToCompletion(agent, wreq), "write transfer");
    CHECK_OK(agent.releaseXferReq(wreq), "release write req");

    std::memset(dram, 0, kIoSize);

    nixl_xfer_dlist_t rsrc(DRAM_SEG), rdst(BLK_SEG);
    rsrc.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(dram), kIoSize, kDevId));
    rdst.addDesc(nixlBasicDesc(kBdevOffset, kIoSize, kDevId));

    nixlXferReqH *rreq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_READ, rsrc, rdst, kAgentName, rreq, &backendParams),
             "create read req");
    CHECK_OK(runToCompletion(agent, rreq), "read transfer");
    CHECK_OK(agent.releaseXferReq(rreq), "release read req");
    CHECK(verifyPattern(dram, kIoSize, 0xa5), "round-trip data mismatch");
    std::printf("  phase 1 (write/read/verify): OK\n");

    // --- Phase 2: repost a completed handle ---
    // Post the same handle twice with different data; the second post must
    // actually re-run the I/O. If completion state is not reset on repost, the
    // handle reports the stale completion and the new data is never written.
    nixl_xfer_dlist_t psrc(DRAM_SEG), pdst(BLK_SEG);
    psrc.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(dram), kIoSize, kDevId));
    pdst.addDesc(nixlBasicDesc(kBdevOffset, kIoSize, kDevId));
    nixlXferReqH *preq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_WRITE, psrc, pdst, kAgentName, preq, &backendParams),
             "create repost req");

    fillPattern(dram, kIoSize, 0x11);
    CHECK_OK(runToCompletion(agent, preq), "repost write #1");
    fillPattern(dram, kIoSize, 0x22);
    CHECK_OK(runToCompletion(agent, preq), "repost write #2");
    CHECK_OK(agent.releaseXferReq(preq), "release repost req");

    std::memset(dram, 0, kIoSize);
    nixlXferReqH *vreq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_READ, rsrc, rdst, kAgentName, vreq, &backendParams),
             "create repost-verify req");
    CHECK_OK(runToCompletion(agent, vreq), "repost verify read");
    CHECK_OK(agent.releaseXferReq(vreq), "release repost-verify req");
    CHECK(verifyPattern(dram, kIoSize, 0x22), "repost did not re-run the write");
    std::printf("  phase 2 (repost): OK\n");

    // --- Phase 3: release while (potentially) in flight ---
    // Post and immediately release without waiting, many times. This drives the
    // cancel path and its handle-lifetime election; under ASAN a double-free or
    // use-after-free here would be caught.
    constexpr int kCancelIters = 256;
    for (int i = 0; i < kCancelIters; ++i) {
        nixl_xfer_dlist_t csrc(DRAM_SEG), cdst(BLK_SEG);
        csrc.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(dram), kIoSize, kDevId));
        cdst.addDesc(nixlBasicDesc(kBdevOffset, kIoSize, kDevId));
        nixlXferReqH *creq = nullptr;
        CHECK_OK(agent.createXferReq(NIXL_WRITE, csrc, cdst, kAgentName, creq, &backendParams),
                 "create cancel req");
        nixl_status_t status = agent.postXferReq(creq);
        CHECK(status == NIXL_IN_PROG || status == NIXL_SUCCESS, "post cancel req");
        CHECK_OK(agent.releaseXferReq(creq), "release in-flight req");
    }
    std::printf("  phase 3 (release-in-flight x%d): OK\n", kCancelIters);

    // Engine must still be healthy after the cancel burst.
    std::memset(dram, 0, kIoSize);
    nixlXferReqH *freq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_READ, rsrc, rdst, kAgentName, freq, &backendParams),
             "create final read req");
    CHECK_OK(runToCompletion(agent, freq), "final read transfer");
    CHECK_OK(agent.releaseXferReq(freq), "release final read req");
    std::printf("  phase 4 (post-cancel health check): OK\n");

    CHECK_OK(agent.deregisterMem(dramReg, &backendParams), "deregister DRAM");
    CHECK_OK(agent.deregisterMem(bdevReg, &backendParams), "deregister bdev");
    return 0;
}

// The SPDK runtime (env, thread lib, accel, bdev) is a process-wide singleton
// brought up by the first backend and torn down by the last. Run a second
// backend, on its own bdev, while the first is still alive: it must attach to
// the running runtime rather than re-initialize or fail. Then drop the first
// agent and keep going, which is the harder ordering: the shared control state
// must not belong to whichever backend happened to start first.
int
runSecondBackend(const nixl_b_params_t &base,
                 bool progThread,
                 std::unique_ptr<nixlAgent> &first,
                 void *dram,
                 size_t dramLen) {
    constexpr const char *kAgent2 = "spdk-runtime-test-2";
    constexpr const char *kBdev2 = "Malloc1";

    nixlAgentConfig cfg(progThread);
    nixlAgent agent(kAgent2, cfg);

    nixlBackendH *backend = nullptr;
    CHECK_OK(agent.createBackend("SPDK", makeParams(base, kBdev2, "nixl_spdk_rt2"), backend),
             "create second backend");
    CHECK(backend != nullptr, "second backend handle");

    nixl_opt_args_t backendParams;
    backendParams.backends.push_back(backend);

    nixl_reg_dlist_t dramReg(DRAM_SEG);
    dramReg.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(dram), dramLen, kDevId));
    CHECK_OK(agent.registerMem(dramReg, &backendParams), "register DRAM (2)");

    nixl_reg_dlist_t bdevReg(BLK_SEG);
    bdevReg.addDesc(nixlBlobDesc(0, kBdevBytes, kDevId, kBdev2));
    CHECK_OK(agent.registerMem(bdevReg, &backendParams), "register bdev (2)");

    fillPattern(dram, kIoSize, 0x5a);
    nixl_xfer_dlist_t src(DRAM_SEG), dst(BLK_SEG);
    src.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(dram), kIoSize, kDevId));
    dst.addDesc(nixlBasicDesc(kBdevOffset, kIoSize, kDevId));

    nixlXferReqH *wreq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_WRITE, src, dst, kAgent2, wreq, &backendParams),
             "create write req (2)");
    CHECK_OK(runToCompletion(agent, wreq), "write transfer (2)");
    CHECK_OK(agent.releaseXferReq(wreq), "release write req (2)");

    std::memset(dram, 0, kIoSize);
    nixlXferReqH *rreq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_READ, src, dst, kAgent2, rreq, &backendParams),
             "create read req (2)");
    CHECK_OK(runToCompletion(agent, rreq), "read transfer (2)");
    CHECK_OK(agent.releaseXferReq(rreq), "release read req (2)");
    CHECK(verifyPattern(dram, kIoSize, 0x5a), "second-backend data mismatch");

    // Tear down the backend that brought the runtime up, then keep using this
    // one.
    first.reset();

    std::memset(dram, 0, kIoSize);
    nixlXferReqH *sreq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_READ, src, dst, kAgent2, sreq, &backendParams),
             "create survivor read req");
    CHECK_OK(runToCompletion(agent, sreq), "survivor read transfer");
    CHECK_OK(agent.releaseXferReq(sreq), "release survivor read req");
    CHECK(verifyPattern(dram, kIoSize, 0x5a), "survivor data mismatch");

    CHECK_OK(agent.deregisterMem(dramReg, &backendParams), "deregister DRAM (2)");
    CHECK_OK(agent.deregisterMem(bdevReg, &backendParams), "deregister bdev (2)");
    return 0;
}

} // namespace

int
main() {
    // Progress-thread mode by default (exercises the concurrent cancel path);
    // override with NIXL_SPDK_TEST_PROG_THREAD=0 to run in the caller-driven
    // backend-locked mode.
    bool progThread = true;
    if (const char *v = std::getenv("NIXL_SPDK_TEST_PROG_THREAD")) {
        progThread = !(v[0] == '0');
    }
    nixlAgentConfig cfg(progThread);
    auto agent = std::make_unique<nixlAgent>(kAgentName, cfg);

    nixl_b_params_t params;
    nixl_mem_list_t mems;
    nixl_status_t status = agent->getPluginParams("SPDK", mems, params);
    if (status != NIXL_SUCCESS) {
        std::printf("SKIP: SPDK plugin not available (getPluginParams=%d)\n", status);
        return 0;
    }

    // Allow overriding any backend param from the environment as
    // NIXL_SPDK_TEST_<UPPER_KEY> so the harness can be tuned per host without a
    // rebuild (e.g. NIXL_SPDK_TEST_NO_HUGE=0 on a host with hugepages).
    for (const char *key : {"msg_mempool_size", "core_mask"}) {
        std::string envName = "NIXL_SPDK_TEST_";
        for (const char *c = key; *c; ++c) {
            envName += static_cast<char>(std::toupper(*c));
        }
        if (const char *v = std::getenv(envName.c_str())) {
            params[key] = v;
        }
    }

    nixlBackendH *backend = nullptr;
    // Configure the malloc bdev via the convenience params (no JSON, no file).
    status = agent->createBackend("SPDK", makeParams(params, kBdevName, "nixl_spdk_rt"), backend);
    if (status != NIXL_SUCCESS || backend == nullptr) {
        std::printf("SKIP: could not start SPDK runtime (createBackend=%d). "
                    "This usually means no usable DPDK memory in the environment.\n",
                    status);
        return 0;
    }

    // Upstream SPDK registers memory at 4 KiB granularity, so a plain
    // page-aligned buffer works with no hugepages required (64 KiB here is
    // 4 KiB-aligned but deliberately not 2 MiB-aligned).
    void *dram = nullptr;
    constexpr size_t kPageSize = 4096;
    const size_t dramLen = kIoSize;
    if (posix_memalign(&dram, kPageSize, dramLen) != 0 || dram == nullptr) {
        std::fprintf(stderr, "FAIL: posix_memalign failed\n");
        return 1;
    }

    int rc = runTest(*agent, backend, dram, dramLen);
    if (rc == 0) {
        // Still inside the first backend's lifetime, so both are live at once.
        rc = runSecondBackend(params, progThread, agent, dram, dramLen);
        if (rc == 0) {
            std::printf("  phase 6 (second concurrent backend, first released first): OK\n");
        }
    }

    free(dram);

    if (rc != 0) {
        std::printf("RESULT: FAILED (%d)\n", rc);
        return 1;
    }
    std::printf("RESULT: PASSED\n");
    return 0;
}
