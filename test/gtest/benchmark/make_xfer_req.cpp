/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file make_xfer_req.cpp
 * @brief Dedicated benchmark for nixlAgent::makeXferReq isolated from backend implementations
 *
 * This benchmark uses a single agent with mock backend and loopback transfers to measure
 * the pure overhead of makeXferReq without any real backend implementation costs.
 *
 * It measures:
 * - Parameter validation
 * - Descriptor list processing
 * - Descriptor merging optimization
 * - Memory allocation and copying
 *
 * By using loopback transfers (agent to itself), we eliminate multi-agent coordination
 * overhead and focus purely on the makeXferReq function performance.
 */

#include "common.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "mocks/gmock_engine.h"

#include "nixl.h"
#include "nixl_types.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>

using ::testing::_;
using ::testing::Return;
using ::testing::NiceMock;
using ::testing::Invoke;

namespace gtest {

/**
 * @class BenchmarkMakeXferReq
 * @brief Benchmark suite for nixlAgent::makeXferReq with mock backend isolation
 *
 * This test fixture creates a minimal environment with a single agent and mock backend
 * to measure the performance of makeXferReq in isolation from any real backend implementation.
 * Uses loopback transfers (agent to itself) to avoid multi-agent overhead.
 */
class BenchmarkMakeXferReq : public ::testing::Test {
protected:
    void
    SetUp() override {
        env_.addVar("NIXL_TELEMETRY_ENABLE", "n");

        // Setup mock backend engine
        setupMockEngine();

        // Create single agent with mock backend
        createAgent();
    }

    void
    TearDown() override {
        agent_.reset();
        gmock_engine_.reset();
    }

    /**
     * @brief Setup mock backend engine with minimal overhead
     */
    void
    setupMockEngine() {
        // Create a single mock engine for loopback transfers
        auto engine = std::make_unique<NiceMock<mocks::GMockBackendEngine>>();

        // Configure mock to support local operations (loopback)
        ON_CALL(*engine, supportsRemote()).WillByDefault(Return(false));
        ON_CALL(*engine, supportsLocal()).WillByDefault(Return(true));
        ON_CALL(*engine, supportsNotif()).WillByDefault(Return(false));
        ON_CALL(*engine, getSupportedMems()).WillByDefault(Return(nixl_mem_list_t{DRAM_SEG}));

        // registerMem: allocate a dummy metadata object
        ON_CALL(*engine, registerMem(_, _, _))
            .WillByDefault(
                Invoke([](const nixlBlobDesc &, const nixl_mem_t &, nixlBackendMD *&out) {
                    out = new nixlBackendMD(false);
                    return NIXL_SUCCESS;
                }));

        // deregisterMem: clean up metadata
        ON_CALL(*engine, deregisterMem(_)).WillByDefault(Invoke([](nixlBackendMD *meta) {
            delete meta;
            return NIXL_SUCCESS;
        }));

        // connect: minimal overhead (for local/self connection)
        ON_CALL(*engine, connect(_)).WillByDefault(Return(NIXL_SUCCESS));

        // getPublicData: return minimal public data
        ON_CALL(*engine, getPublicData(_, _))
            .WillByDefault(Invoke([](const nixlBackendMD *, std::string &str) {
                str = "mock_public_data";
                return NIXL_SUCCESS;
            }));

        // loadLocalMD: create local metadata copy
        ON_CALL(*engine, loadLocalMD(_, _))
            .WillByDefault(Invoke([](nixlBackendMD *, nixlBackendMD *&output) {
                output = new nixlBackendMD(false);
                return NIXL_SUCCESS;
            }));

        // unloadMD: clean up metadata
        ON_CALL(*engine, unloadMD(_)).WillByDefault(Invoke([](nixlBackendMD *input) {
            delete input;
            return NIXL_SUCCESS;
        }));

        // prepXfer: THIS IS THE KEY - minimal overhead to isolate makeXferReq
        ON_CALL(*engine, prepXfer(_, _, _, _, _, _))
            .WillByDefault(Invoke([](const nixl_xfer_op_t &,
                                     const nixl_meta_dlist_t &,
                                     const nixl_meta_dlist_t &,
                                     const std::string &,
                                     nixlBackendReqH *&handle,
                                     const nixl_opt_b_args_t *) {
                // Allocate a dummy handle - this is the only work the backend does
                handle = reinterpret_cast<nixlBackendReqH *>(new int(42));
                return NIXL_SUCCESS;
            }));

        // releaseReqH: clean up the dummy handle
        ON_CALL(*engine, releaseReqH(_)).WillByDefault(Invoke([](nixlBackendReqH *handle) {
            delete reinterpret_cast<int *>(handle);
            return NIXL_SUCCESS;
        }));

        gmock_engine_ = std::move(engine);
    }

    /**
     * @brief Create a single agent with mock backend for loopback transfers
     */
    void
    createAgent() {
        nixlAgentConfig config(false, // progress thread
                               false, // no listen thread needed for loopback
                               0, // no port needed
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_RW,
                               1,
                               0,
                               100000,
                               false);

        agent_ = std::make_unique<nixlAgent>(getAgentName(), config);

        // Create backend with mock engine
        nixl_b_params_t params;
        gmock_engine_->SetToParams(params);

        nixlBackendH *backend_handle = nullptr;
        nixl_status_t status = agent_->createBackend(GetMockBackendName(), params, backend_handle);

        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(backend_handle, nullptr);
        backend_handle_ptr_ = backend_handle;
    }

    /**
     * @brief Create and register memory buffers
     * @return vector of byte buffers that automatically clean up on destruction
     */
    std::vector<std::vector<uint8_t>>
    createRegisteredMem(size_t size, size_t count) {
        std::vector<std::vector<uint8_t>> buffers;
        buffers.reserve(count);
        nixlDescList<nixlBlobDesc> desc_list(DRAM_SEG, count);

        for (size_t i = 0; i < count; i++) {
            buffers.emplace_back(size);
            desc_list.addDesc(
                nixlBlobDesc(reinterpret_cast<uintptr_t>(buffers[i].data()), size, 0));
        }

        nixl_status_t status = agent_->registerMem(desc_list);
        if (status != NIXL_SUCCESS) {
            ADD_FAILURE() << "Failed to register memory: " << status;
            return {};
        }

        return buffers;
    }

    /**
     * @brief Deregister memory buffers from agent
     */
    void
    deregisterMem(const std::vector<std::vector<uint8_t>> &buffers, size_t size) {
        nixlDescList<nixlBlobDesc> desc_list(DRAM_SEG);
        for (const auto &buffer : buffers) {
            desc_list.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer.data()), size, 0));
        }

        agent_->deregisterMem(desc_list);
    }

    /**
     * @brief Statistics structure for benchmark results
     */
    struct BenchmarkStats {
        double mean_us;
        double min_us;
        double max_us;
        double stddev_us;
        double median_us;
        double p95_us;
        double p99_us;
        size_t iterations;
        size_t desc_count;
        size_t desc_size;
        bool skip_merge;
    };

    /**
     * @brief Calculate statistics from timing measurements
     */
    BenchmarkStats
    calculateStats(const std::vector<double> &times_us,
                   size_t desc_count,
                   size_t desc_size,
                   bool skip_merge) {
        BenchmarkStats stats;
        stats.iterations = times_us.size();
        stats.desc_count = desc_count;
        stats.desc_size = desc_size;
        stats.skip_merge = skip_merge;

        if (times_us.empty()) {
            return stats;
        }

        // Calculate mean
        double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
        stats.mean_us = sum / times_us.size();

        // Calculate min and max
        auto minmax = std::minmax_element(times_us.begin(), times_us.end());
        stats.min_us = *minmax.first;
        stats.max_us = *minmax.second;

        // Calculate standard deviation
        double sq_sum =
            std::accumulate(times_us.begin(), times_us.end(), 0.0, [stats](double acc, double val) {
                return acc + (val - stats.mean_us) * (val - stats.mean_us);
            });
        stats.stddev_us = std::sqrt(sq_sum / times_us.size());

        // Calculate median and percentiles
        std::vector<double> sorted_times = times_us;
        std::sort(sorted_times.begin(), sorted_times.end());

        size_t median_idx = sorted_times.size() / 2;
        if (sorted_times.size() % 2 == 0) {
            stats.median_us = (sorted_times[median_idx - 1] + sorted_times[median_idx]) / 2.0;
        } else {
            stats.median_us = sorted_times[median_idx];
        }

        size_t p95_idx = static_cast<size_t>(sorted_times.size() * 0.95);
        size_t p99_idx = static_cast<size_t>(sorted_times.size() * 0.99);
        if (p95_idx >= sorted_times.size()) p95_idx = sorted_times.size() - 1;
        if (p99_idx >= sorted_times.size()) p99_idx = sorted_times.size() - 1;

        stats.p95_us = sorted_times[p95_idx];
        stats.p99_us = sorted_times[p99_idx];

        return stats;
    }

    /**
     * @brief Print benchmark results in a formatted table
     */
    void
    printBenchmarkResults(const std::string &test_name, const BenchmarkStats &stats) {
        Logger() << "========================================";
        Logger() << "Benchmark: " << test_name;
        Logger() << "========================================";
        Logger() << "Configuration:";
        Logger() << "  Descriptor Count: " << stats.desc_count;
        Logger() << "  Descriptor Size:  " << stats.desc_size << " bytes";
        Logger() << "  Skip Merge:       " << (stats.skip_merge ? "Yes" : "No");
        Logger() << "  Iterations:       " << stats.iterations;
        Logger() << "";
        Logger() << "Results (microseconds):";
        Logger() << "  Mean:     " << std::fixed << std::setprecision(3) << stats.mean_us << " μs";
        Logger() << "  Median:   " << std::fixed << std::setprecision(3) << stats.median_us
                 << " μs";
        Logger() << "  Min:      " << std::fixed << std::setprecision(3) << stats.min_us << " μs";
        Logger() << "  Max:      " << std::fixed << std::setprecision(3) << stats.max_us << " μs";
        Logger() << "  Std Dev:  " << std::fixed << std::setprecision(3) << stats.stddev_us
                 << " μs";
        Logger() << "  P95:      " << std::fixed << std::setprecision(3) << stats.p95_us << " μs";
        Logger() << "  P99:      " << std::fixed << std::setprecision(3) << stats.p99_us << " μs";
        Logger() << "";

        // Calculate per-descriptor overhead
        double per_desc_ns = (stats.mean_us * 1000.0) / stats.desc_count;
        Logger() << "Per-Descriptor Overhead: " << std::fixed << std::setprecision(2) << per_desc_ns
                 << " ns/desc";
        Logger() << "========================================";
        Logger() << "";
    }

    /**
     * @brief Main benchmark function for makeXferReq
     */
    void
    benchmarkMakeXferReq(const std::string &test_name,
                         size_t desc_count,
                         size_t desc_size,
                         size_t iterations,
                         bool skip_desc_merge = false) {
        // Setup memory - both source and destination from same agent (loopback)
        auto src_buffers = createRegisteredMem(desc_size, desc_count);
        auto dst_buffers = createRegisteredMem(desc_size, desc_count);

        // Prepare descriptor lists
        nixlDescList<nixlBasicDesc> local_descs(DRAM_SEG);
        nixlDescList<nixlBasicDesc> remote_descs(DRAM_SEG);

        for (size_t i = 0; i < desc_count; i++) {
            local_descs.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(src_buffers[i].data()), desc_size, 0));
            remote_descs.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(dst_buffers[i].data()), desc_size, 0));
        }

        // Prepare DlistH handles
        // For loopback: both use NIXL_INIT_AGENT
        nixlDlistH *local_dlist = nullptr;
        nixlDlistH *remote_dlist = nullptr;

        nixl_status_t status = agent_->prepXferDlist(NIXL_INIT_AGENT, local_descs, local_dlist);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(local_dlist, nullptr);

        status = agent_->prepXferDlist(getAgentName(), remote_descs, remote_dlist);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(remote_dlist, nullptr);

        // Create index arrays
        std::vector<int> local_indices(desc_count);
        std::vector<int> remote_indices(desc_count);
        std::iota(local_indices.begin(), local_indices.end(), 0);
        std::iota(remote_indices.begin(), remote_indices.end(), 0);

        // Warmup iterations
        constexpr size_t WARMUP_ITERS = 10;
        for (size_t i = 0; i < WARMUP_ITERS; ++i) {
            nixlXferReqH *req_hndl = nullptr;
            nixl_opt_args_t extra_params;
            extra_params.skipDescMerge = skip_desc_merge;

            status = agent_->makeXferReq(NIXL_WRITE,
                                         local_dlist,
                                         local_indices,
                                         remote_dlist,
                                         remote_indices,
                                         req_hndl,
                                         &extra_params);
            ASSERT_EQ(status, NIXL_SUCCESS);
            ASSERT_NE(req_hndl, nullptr);

            status = agent_->releaseXferReq(req_hndl);
            ASSERT_EQ(status, NIXL_SUCCESS);
        }

        // Benchmark iterations
        std::vector<double> times_us;
        times_us.reserve(iterations);

        for (size_t i = 0; i < iterations; ++i) {
            nixlXferReqH *req_hndl = nullptr;
            nixl_opt_args_t extra_params;
            extra_params.skipDescMerge = skip_desc_merge;

            auto start = std::chrono::high_resolution_clock::now();

            status = agent_->makeXferReq(NIXL_WRITE,
                                         local_dlist,
                                         local_indices,
                                         remote_dlist,
                                         remote_indices,
                                         req_hndl,
                                         &extra_params);

            auto end = std::chrono::high_resolution_clock::now();

            ASSERT_EQ(status, NIXL_SUCCESS);
            ASSERT_NE(req_hndl, nullptr);

            double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
            times_us.push_back(elapsed_us);

            status = agent_->releaseXferReq(req_hndl);
            ASSERT_EQ(status, NIXL_SUCCESS);
        }

        // Calculate and print statistics
        BenchmarkStats stats = calculateStats(times_us, desc_count, desc_size, skip_desc_merge);
        printBenchmarkResults(test_name, stats);

        // Cleanup
        agent_->releasedDlistH(local_dlist);
        agent_->releasedDlistH(remote_dlist);
        deregisterMem(src_buffers, desc_size);
        deregisterMem(dst_buffers, desc_size);
    }

    std::string
    getAgentName() {
        return "benchmark_agent";
    }

    gtest::ScopedEnv env_;
    std::unique_ptr<nixlAgent> agent_;
    nixlBackendH *backend_handle_ptr_;
    std::unique_ptr<NiceMock<mocks::GMockBackendEngine>> gmock_engine_;
};

// ============================================================================
// Benchmark Test Cases
// ============================================================================

/**
 * @test MakeXferReq
 * @brief Run MakeXferReq with various parameters
 */
TEST_F(BenchmarkMakeXferReq, Merge) {
    const std::vector<std::tuple<size_t, std::string>> test_cases = {{16, "16"}, {256, "256"}};

    for (const auto &[count, name] : test_cases) {
        benchmarkMakeXferReq("DescCount_" + name, count, 4096, 500, false);
    }
}

TEST_F(BenchmarkMakeXferReq, NoMerge) {
    const std::vector<std::tuple<size_t, std::string>> test_cases = {{16, "16"}, {256, "256"}};

    for (const auto &[count, name] : test_cases) {
        benchmarkMakeXferReq("DescCount_" + name, count, 4096, 500, true);
    }
}

} // namespace gtest
