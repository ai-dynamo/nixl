/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runtime test for the SPDK plugin. It spins up a private SPDK runtime over an
// in-memory malloc bdev (no hugepages / no PCI, so it can run unprivileged) and
// drives real I/O through the public NIXL API to exercise the request-handle
// lifetime logic: a write/read round trip, a repost of a completed handle, a
// burst of release-while-in-flight cancellations, an idle period spanning
// several background-maintenance wakeups, and a second backend sharing the
// process-wide SPDK runtime with the first.
//
// Set NIXL_SPDK_TEST_VFIO_SOCKET to exercise a real vfio-user block namespace
// with an imported shared payload. Explicit transport tests never skip.
// Otherwise unavailable environments return Meson's skip status (77).

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/mman.h>
#include <linux/memfd.h>

#include "nixl.h"

namespace {

constexpr const char *kAgentName = "spdk-runtime-test";
const char *kBdevName = "Malloc0";
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
    // needs; a few thousand entries keeps the footprint small. Only a default:
    // an explicit value in base (from NIXL_SPDK_TEST_MSG_MEMPOOL_SIZE) wins.
    if (!params.count("msg_mempool_size")) {
        params["msg_mempool_size"] = "4095";
    }
    return params;
}

int
runTest(nixlAgent &agent, nixlBackendH *backend, void *dram, size_t dramLen) {
    nixl_opt_args_t backendParams;
    backendParams.backends.push_back(backend);

    int privateFlags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (std::getenv("NIXL_SPDK_TEST_VFIO_SOCKET")) {
        privateFlags |= MAP_HUGETLB | MAP_HUGE_2MB;
    }
    void *privateBuf = mmap(nullptr, dramLen, PROT_READ | PROT_WRITE, privateFlags, -1, 0);
    CHECK(privateBuf != MAP_FAILED, "allocate private negative-test buffer");
    nixl_reg_dlist_t privateReg(DRAM_SEG);
    privateReg.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(privateBuf), dramLen, kDevId));
    CHECK(agent.registerMem(privateReg, &backendParams) != NIXL_SUCCESS,
          "private memory must be rejected even without a connected vfio-user controller");
    munmap(privateBuf, dramLen);
    std::printf("  private DRAM registration rejected: OK\n");

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

    // --- Phase 5: OBJ_SEG capability detection ---
    // OBJ_SEG registers one object (key via metaInfo) and only works on a device
    // that supports NVMe passthru and identifies as the KV command set. The
    // malloc bdev is neither, so registration must fail cleanly (no crash). A
    // real KV round trip needs a KV-capable device, which this in-memory
    // environment lacks.
    nixl_reg_dlist_t objReg(OBJ_SEG);
    objReg.addDesc(nixlBlobDesc(0, kIoSize, kDevId, "nixl-kv-key"));
    nixl_status_t objStatus = agent.registerMem(objReg, &backendParams);
    CHECK(objStatus != NIXL_SUCCESS, "OBJ_SEG registration on a non-KV device must be rejected");
    std::printf("  phase 5 (OBJ_SEG rejects non-KV device): OK (status=%d)\n", objStatus);

    // --- Phase 6: idle, then transfer again ---
    // Both threading models keep a background loop alive to service the control
    // plane while the application makes no calls: the progress thread wakes on a
    // bounded wait, the caller-driven modes run a maintenance thread. Sit idle
    // across several of those wakeups, then drive I/O again. Keep-alive itself is
    // not observable here, since that needs a fabrics target and an in-memory
    // malloc bdev is not one. What this covers is that the background loop leaves
    // a quiet backend's datapath intact, and that the teardown below races
    // correctly against a loop that is still running.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    fillPattern(dram, kIoSize, 0x5c);
    nixlXferReqH *ireq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_WRITE, wsrc, wdst, kAgentName, ireq, &backendParams),
             "create post-idle write req");
    CHECK_OK(runToCompletion(agent, ireq), "post-idle write transfer");
    CHECK_OK(agent.releaseXferReq(ireq), "release post-idle write req");

    std::memset(dram, 0, kIoSize);
    nixlXferReqH *ivreq = nullptr;
    CHECK_OK(agent.createXferReq(NIXL_READ, rsrc, rdst, kAgentName, ivreq, &backendParams),
             "create post-idle read req");
    CHECK_OK(runToCompletion(agent, ivreq), "post-idle read transfer");
    CHECK_OK(agent.releaseXferReq(ivreq), "release post-idle read req");
    CHECK(verifyPattern(dram, kIoSize, 0x5c), "post-idle round-trip data mismatch");
    std::printf("  phase 6 (idle across maintenance wakeups, then I/O): OK\n");

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
    const char *vfioSocket = std::getenv("NIXL_SPDK_TEST_VFIO_SOCKET");
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
        return vfioSocket ? 1 : 77;
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
    auto backendConfig = makeParams(params, kBdevName, "nixl_spdk_rt");
    if (vfioSocket) {
        kBdevName = "Vfion1";
        backendConfig["bdev_type"] = "nvme";
        backendConfig["bdev_name"] = "Vfio";
        backendConfig["bdev_trtype"] = "VFIOUSER";
        backendConfig["bdev_traddr"] = vfioSocket;
        backendConfig["bdev_subnqn"] = "nqn.2026-01.io.spdk:nixl-memory";
    }
    status = agent->createBackend("SPDK", backendConfig, backend);
    if (status != NIXL_SUCCESS || backend == nullptr) {
        std::printf("SKIP: could not start SPDK runtime (createBackend=%d). "
                    "Check the preceding environment/configuration errors.\n",
                    status);
        return vfioSocket ? 1 : 77;
    }

    // Keep malloc testing unprivileged. The VFIOUSER test instead exports a
    // whole hugepage so the server can register it in PA mode.
    void *dram = nullptr;
    const size_t dramLen = vfioSocket ? 2 * 1024 * 1024 : kIoSize;
    int fd = -1;
    if (vfioSocket && std::getenv("NIXL_SPDK_TEST_ANON_SHARED")) {
        dram = mmap(nullptr,
                    dramLen,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
                    -1,
                    0);
        CHECK(dram != MAP_FAILED, "map anonymous shared hugepage payload");
    } else {
        fd = memfd_create("nixl-runtime-payload",
                          MFD_CLOEXEC | (vfioSocket ? MFD_HUGETLB | MFD_HUGE_2MB : 0));
        CHECK(fd >= 0 && ftruncate(fd, dramLen) == 0, "create shared payload");
        dram = mmap(nullptr, dramLen, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        CHECK(dram != MAP_FAILED, "map shared payload");
    }

    int rc = runTest(*agent, backend, dram, dramLen);
    if (rc == 0 && !vfioSocket) {
        // Still inside the first backend's lifetime, so both are live at once.
        rc = runSecondBackend(params, progThread, agent, dram, dramLen);
        if (rc == 0) {
            std::printf("  phase 7 (second concurrent backend, first released first): OK\n");
        }
    }

    // Destroy the runtime before unmapping, including error exits where a
    // failed phase may have left registrations or requests outstanding.
    agent.reset();
    munmap(dram, dramLen);
    if (fd >= 0) {
        close(fd);
    }

    if (rc != 0) {
        std::printf("RESULT: FAILED (%d)\n", rc);
        return 1;
    }
    std::printf("RESULT: PASSED\n");
    return 0;
}
