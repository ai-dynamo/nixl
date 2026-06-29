/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Behavior contract for the GDS family of backends ("GDS" and "GDS_MT").
//
// These tests are intentionally black-box: they assert behavior only through the
// public nixlAgent / plugin-manager API and never reference any backend-internal
// class. This keeps them valid regardless of how the plugins are implemented
// (separate plugins, a merged plugin with a shared engine, etc.), so they serve
// as the source of truth while GDS and GDS_MT are unified into one plugin.
//
// Anything that needs a working cuFile driver (i.e. real GDS hardware) skips
// gracefully via GTEST_SKIP when the backend cannot be created, mirroring how
// the rest of the storage tests gate on CUDA availability.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "common.h"
#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"
#include "nixl_types.h"

namespace {

constexpr unsigned char kPattern = 0xAB;
constexpr auto kTransferTimeout = std::chrono::seconds(30);
constexpr auto kPollInterval = std::chrono::milliseconds(50);

bool
hasMem(const nixl_mem_list_t &mems, nixl_mem_t m) {
    return std::find(mems.begin(), mems.end(), m) != mems.end();
}

std::string
makeSizedFile(const std::string &name, size_t size) {
    const std::string path = (std::filesystem::temp_directory_path() / name).string();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (size > 0) {
        f.seekp(static_cast<std::streamoff>(size) - 1);
        f.put('\0');
    }
    return path;
}

// Drive one operation on the given lists, forcing the transfer onto `be`.
nixl_status_t
runTransfer(nixlAgent &agent,
            const std::string &self,
            nixlBackendH *be,
            nixl_xfer_op_t op,
            const nixl_xfer_dlist_t &mem,
            const nixl_xfer_dlist_t &file) {
    nixl_opt_args_t ep;
    ep.backends = {be};

    nixlXferReqH *req = nullptr;
    nixl_status_t status = agent.createXferReq(op, mem, file, self, req, &ep);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    status = agent.postXferReq(req);
    const auto deadline = std::chrono::steady_clock::now() + kTransferTimeout;
    while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(kPollInterval);
        status = agent.getXferStatus(req);
    }
    if (status == NIXL_IN_PROG) {
        status = NIXL_ERR_BACKEND;
    }

    const nixl_status_t release_status = agent.releaseXferReq(req);
    return (release_status == NIXL_SUCCESS) ? status : release_status;
}

nixl_status_t
runWriteThenRead(nixlAgent &agent,
                 const std::string &self,
                 nixlBackendH *be,
                 const nixl_xfer_dlist_t &mem,
                 const nixl_xfer_dlist_t &file) {
    const nixl_status_t write_status =
        runTransfer(agent, self, be, NIXL_WRITE, mem, file);
    return (write_status == NIXL_SUCCESS)
               ? runTransfer(agent, self, be, NIXL_READ, mem, file)
               : write_status;
}

// Round-trip a DRAM buffer through a path-mode file; returns true if the bytes
// survive write+read.
bool
dramRoundTrip(nixlAgent &agent, const std::string &self, nixlBackendH *be, size_t size) {
    const std::string path = makeSizedFile("nixl_gds_gtest_dram.bin", size);

    nixl_opt_args_t ep;
    ep.backends = {be};

    nixl_reg_dlist_t file_descs(FILE_SEG);
    nixlBlobDesc fdesc;
    fdesc.addr = 0;
    fdesc.len = size;
    fdesc.devId = 0;
    fdesc.metaInfo = "rw:" + path;
    file_descs.addDesc(fdesc);
    if (agent.registerMem(file_descs, &ep) != NIXL_SUCCESS) {
        std::filesystem::remove(path);
        return false;
    }

    void *buf = nullptr;
    if (posix_memalign(&buf, sysconf(_SC_PAGESIZE), size) != 0) {
        buf = nullptr;
    }

    bool ok = false;
    if (buf != nullptr) {
        nixl_reg_dlist_t mem_descs(DRAM_SEG);
        nixlBlobDesc mdesc;
        mdesc.addr = reinterpret_cast<uintptr_t>(buf);
        mdesc.len = size;
        mdesc.devId = 0;
        mem_descs.addDesc(mdesc);

        if (agent.registerMem(mem_descs, &ep) == NIXL_SUCCESS) {
            std::memset(buf, kPattern, size);
            nixl_xfer_dlist_t mx = mem_descs.trim();
            nixl_xfer_dlist_t fx = file_descs.trim();
            if (runTransfer(agent, self, be, NIXL_WRITE, mx, fx) == NIXL_SUCCESS) {
                std::memset(buf, 0, size);
                if (runTransfer(agent, self, be, NIXL_READ, mx, fx) == NIXL_SUCCESS) {
                    ok = true;
                    for (size_t i = 0; i < size; ++i) {
                        if (static_cast<unsigned char *>(buf)[i] != kPattern) {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            agent.deregisterMem(mem_descs, &ep);
        }
        free(buf);
    }

    agent.deregisterMem(file_descs, &ep);
    std::filesystem::remove(path);
    return ok;
}

// Try to create the backend; on failure, the caller should skip (no GDS hw).
nixl_status_t
tryCreate(nixlAgent &agent, const std::string &name, nixlBackendH *&be) {
    nixl_b_params_t params;
    return agent.createBackend(name, params, be);
}

class GdsBackend : public ::testing::TestWithParam<std::string> {};

// ---------------------------------------------------------------------------
// Group A: presence / identity. No cuFile driver (GPU) is required, but the
// plugin must actually have been built/discoverable; if it was not (no
// gds_path/CUDA/cuFile at build time), getPluginParams fails and we skip.
// ---------------------------------------------------------------------------

TEST_P(GdsBackend, AdvertisesDramVramFileMemTypes) {
    nixlAgentConfig cfg;
    nixlAgent agent("gds_presence_" + GetParam(), cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    if (agent.getPluginParams(GetParam(), mems, params) != NIXL_SUCCESS) {
        GTEST_SKIP() << GetParam() << " plugin not available in this build";
    }

    EXPECT_TRUE(hasMem(mems, DRAM_SEG));
    EXPECT_TRUE(hasMem(mems, VRAM_SEG));
    EXPECT_TRUE(hasMem(mems, FILE_SEG));
}

TEST(GdsBatchOptions, AdvertisesBatchConfiguration) {
    nixlAgentConfig cfg;
    nixlAgent agent("gds_batch_options", cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    if (agent.getPluginParams("GDS", mems, params) != NIXL_SUCCESS) {
        GTEST_SKIP() << "GDS plugin not available in this build";
    }

    EXPECT_EQ(params["batch_pool_size"], "16");
    EXPECT_EQ(params["batch_limit"], "128");
    EXPECT_EQ(params["max_request_size"], "16777216");
    EXPECT_EQ(params.find("submit_threads"), params.end());
    EXPECT_EQ(params.find("submit_cpus"), params.end());
}

TEST(GdsMtOptions, AdvertisesThreadCount) {
    nixlAgentConfig cfg;
    nixlAgent agent("gds_mt_options", cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    if (agent.getPluginParams("GDS_MT", mems, params) != NIXL_SUCCESS) {
        GTEST_SKIP() << "GDS_MT plugin not available in this build";
    }

    const auto thread_count = params.find("thread_count");
    ASSERT_NE(thread_count, params.end());
    EXPECT_EQ(thread_count->second,
              std::to_string(std::max(1u, std::thread::hardware_concurrency() / 2)));
}

// ---------------------------------------------------------------------------
// Group B: creation + round-trip (hardware-gated).
// ---------------------------------------------------------------------------

TEST_P(GdsBackend, CreateReportsRequestedType) {
    nixlAgentConfig cfg;
    nixlAgent agent("gds_create_" + GetParam(), cfg);

    nixlBackendH *be = nullptr;
    if (tryCreate(agent, GetParam(), be) != NIXL_SUCCESS || be == nullptr) {
        GTEST_SKIP() << GetParam() << " backend unavailable (no cuFile/GDS)";
    }

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    ASSERT_EQ(agent.getBackendParams(be, mems, params), NIXL_SUCCESS);
    EXPECT_TRUE(hasMem(mems, FILE_SEG));
}

TEST_P(GdsBackend, DramFileRoundTrip) {
    nixlAgentConfig cfg;
    const std::string self = "gds_dram_rt_" + GetParam();
    nixlAgent agent(self, cfg);

    nixlBackendH *be = nullptr;
    if (tryCreate(agent, GetParam(), be) != NIXL_SUCCESS || be == nullptr) {
        GTEST_SKIP() << GetParam() << " backend unavailable (no cuFile/GDS)";
    }

    EXPECT_TRUE(dramRoundTrip(agent, self, be, 1 * 1024 * 1024));
}

// ---------------------------------------------------------------------------
// Group C: validation (hardware-gated - needs a created backend).
// ---------------------------------------------------------------------------

TEST_P(GdsBackend, RejectsMemToMemTransfer) {
    nixlAgentConfig cfg;
    const std::string self = "gds_mem2mem_" + GetParam();
    nixlAgent agent(self, cfg);

    nixlBackendH *be = nullptr;
    if (tryCreate(agent, GetParam(), be) != NIXL_SUCCESS || be == nullptr) {
        GTEST_SKIP() << GetParam() << " backend unavailable (no cuFile/GDS)";
    }

    constexpr size_t size = 4096;
    nixl_opt_args_t ep;
    ep.backends = {be};

    void *a = nullptr;
    void *b = nullptr;
    ASSERT_EQ(posix_memalign(&a, sysconf(_SC_PAGESIZE), size), 0);
    ASSERT_EQ(posix_memalign(&b, sysconf(_SC_PAGESIZE), size), 0);

    nixl_reg_dlist_t da(DRAM_SEG);
    nixl_reg_dlist_t db(DRAM_SEG);
    nixlBlobDesc ad;
    ad.addr = reinterpret_cast<uintptr_t>(a);
    ad.len = size;
    ad.devId = 0;
    da.addDesc(ad);
    nixlBlobDesc bd;
    bd.addr = reinterpret_cast<uintptr_t>(b);
    bd.len = size;
    bd.devId = 0;
    db.addDesc(bd);

    if (agent.registerMem(da, &ep) == NIXL_SUCCESS &&
        agent.registerMem(db, &ep) == NIXL_SUCCESS) {
        nixl_xfer_dlist_t ax = da.trim();
        nixl_xfer_dlist_t bx = db.trim();
        nixlXferReqH *req = nullptr;
        const gtest::LogIgnoreGuard validation_error(
            "GDS: error: backend only supports I/O between memory");
        const gtest::LogIgnoreGuard prepare_error("createXferReq: backend '" + GetParam() +
                                                  "' failed to prepare the transfer request");
        // Neither side is FILE_SEG: GDS must not accept this.
        EXPECT_NE(agent.createXferReq(NIXL_WRITE, ax, bx, self, req, &ep), NIXL_SUCCESS);
        EXPECT_EQ(validation_error.getIgnoredCount(), 1);
        EXPECT_EQ(prepare_error.getIgnoredCount(), 1);
        agent.deregisterMem(db, &ep);
        agent.deregisterMem(da, &ep);
    }
    free(a);
    free(b);
}

// ---------------------------------------------------------------------------
// Group D: risk-targeting (hardware-gated). This is the key driver for the
// merge: registering the same fd twice and deregistering one registration must
// not break transfers through the other. On the batch (GDS) path today this is
// expected to fail due to the by-value file-handle cache; the unified backend
// fixes it via a refcounted handle cache.
// ---------------------------------------------------------------------------

TEST_P(GdsBackend, SharedFdPartialDeregisterStillTransfers) {
    nixlAgentConfig cfg;
    const std::string self = "gds_sharedfd_" + GetParam();
    nixlAgent agent(self, cfg);

    nixlBackendH *be = nullptr;
    if (tryCreate(agent, GetParam(), be) != NIXL_SUCCESS || be == nullptr) {
        GTEST_SKIP() << GetParam() << " backend unavailable (no cuFile/GDS)";
    }

    constexpr size_t blk = 64 * 1024;
    const std::string path = makeSizedFile("nixl_gds_gtest_sharedfd.bin", 2 * blk);
    const int fd = ::open(path.c_str(), O_RDWR);
    ASSERT_GE(fd, 0);

    nixl_opt_args_t ep;
    ep.backends = {be};

    // Two FILE_SEG registrations over the SAME fd (fd-mode: devId == fd), at
    // different file offsets, registered separately so the second hits the
    // backend's per-fd handle cache.
    auto fileDesc = [&](size_t offset) {
        nixl_reg_dlist_t d(FILE_SEG);
        nixlBlobDesc fdesc;
        fdesc.addr = offset; // file offset
        fdesc.len = blk;
        fdesc.devId = static_cast<uint64_t>(fd);
        d.addDesc(fdesc);
        return d;
    };

    nixl_reg_dlist_t file_a = fileDesc(0);
    nixl_reg_dlist_t file_b = fileDesc(blk);

    void *buf = nullptr;
    bool transferred = false;
    if (agent.registerMem(file_a, &ep) == NIXL_SUCCESS &&
        agent.registerMem(file_b, &ep) == NIXL_SUCCESS &&
        posix_memalign(&buf, sysconf(_SC_PAGESIZE), blk) == 0) {

        nixl_reg_dlist_t mem(DRAM_SEG);
        nixlBlobDesc mdesc;
        mdesc.addr = reinterpret_cast<uintptr_t>(buf);
        mdesc.len = blk;
        mdesc.devId = 0;
        mem.addDesc(mdesc);

        if (agent.registerMem(mem, &ep) == NIXL_SUCCESS) {
            // Drop the first registration; the shared fd handle must stay valid
            // for the still-registered second registration.
            ASSERT_EQ(agent.deregisterMem(file_a, &ep), NIXL_SUCCESS);

            std::memset(buf, kPattern, blk);
            nixl_xfer_dlist_t mx = mem.trim();
            nixl_xfer_dlist_t fx = file_b.trim();
            transferred = (runWriteThenRead(agent, self, be, mx, fx) == NIXL_SUCCESS);

            agent.deregisterMem(mem, &ep);
        }
        agent.deregisterMem(file_b, &ep);
    }
    if (buf != nullptr) {
        free(buf);
    }
    ::close(fd);
    std::filesystem::remove(path);

    EXPECT_TRUE(transferred)
        << GetParam()
        << ": transfer via a shared fd failed after a sibling registration was dropped";
}

INSTANTIATE_TEST_SUITE_P(GdsFamily,
                         GdsBackend,
                         ::testing::Values(std::string("GDS"), std::string("GDS_MT")),
                         [](const ::testing::TestParamInfo<std::string> &info) {
                             return info.param;
                         });

// ---------------------------------------------------------------------------
// Group E: GDS and GDS_MT are mutually exclusive within one agent (they each
// open the cuFile driver). Not parameterized - it exercises both names.
// ---------------------------------------------------------------------------

TEST(GdsBackendCombo, GdsThenGdsMtIsRejected) {
    nixlAgentConfig cfg;
    nixlAgent agent("gds_combo_gds_first", cfg);

    nixl_b_params_t params;
    nixlBackendH *first = nullptr;
    if (agent.createBackend("GDS", params, first) != NIXL_SUCCESS || first == nullptr) {
        GTEST_SKIP() << "GDS backend unavailable (no cuFile/GDS)";
    }

    nixlBackendH *second = nullptr;
    const gtest::LogIgnoreGuard illegal_combination(
        "createBackend: Plugin backend GDS_MT is in illegal combination with GDS");
    EXPECT_EQ(agent.createBackend("GDS_MT", params, second), NIXL_ERR_NOT_ALLOWED);
    EXPECT_EQ(illegal_combination.getIgnoredCount(), 1);
}

// Reverse order: a merge could easily regress one direction of the symmetric
// illegal-combination check, so assert both.
TEST(GdsBackendCombo, GdsMtThenGdsIsRejected) {
    nixlAgentConfig cfg;
    nixlAgent agent("gds_combo_gdsmt_first", cfg);

    nixl_b_params_t params;
    nixlBackendH *first = nullptr;
    if (agent.createBackend("GDS_MT", params, first) != NIXL_SUCCESS || first == nullptr) {
        GTEST_SKIP() << "GDS_MT backend unavailable (no cuFile/GDS)";
    }

    nixlBackendH *second = nullptr;
    const gtest::LogIgnoreGuard illegal_combination(
        "createBackend: Plugin backend GDS is in illegal combination with GDS_MT");
    EXPECT_EQ(agent.createBackend("GDS", params, second), NIXL_ERR_NOT_ALLOWED);
    EXPECT_EQ(illegal_combination.getIgnoredCount(), 1);
}

} // namespace
