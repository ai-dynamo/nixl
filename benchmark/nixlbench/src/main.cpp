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

#include "config.h"
#include <iostream>
#include <nixl.h>
#include <sys/time.h>
#include "utils/utils.h"
#include "utils/transfer_layout.h"
#if HAVE_RAW_CLI
#include "utils/raw_cli.h"
#endif
#include "utils/scope_guard.h"
#include "worker/nixl/nixl_worker.h"
#if HAVE_NVSHMEM && HAVE_CUDA
#include "worker/nvshmem/nvshmem_worker.h"
#endif
#include <unistd.h>
#include <memory>
#include <csignal>

static int processBatchSizes(xferBenchWorker &worker,
                             std::vector<std::vector<xferBenchIOV>> &iov_lists,
                             size_t block_size, int num_threads) {
    for (size_t batch_size = xferBenchConfig::start_batch_size;
         !worker.signaled() &&
             batch_size <= xferBenchConfig::max_batch_size;
         batch_size *= 2) {
        const int descriptor_depth = xferBenchConfig::worker_type == XFERBENCH_WORKER_NIXL ?
            xferBenchConfig::iterationPlan(block_size).descriptor_depth :
            xferBenchConfig::pipeline_depth;
        size_t effective_batch = batch_size * descriptor_depth;
        auto local_trans_lists = nixlbench::createTransferDescLists(
            worker.isInitiator(), iov_lists, block_size, effective_batch, num_threads);

        if (worker.isTarget()) {
            if (xferBenchConfig::isStorageBackend()) {
                std::cerr << "storage backend should be always an initiator" << std::endl;
                return EXIT_FAILURE;
            }

            worker.exchangeIOV(local_trans_lists, block_size);
            worker.poll(block_size);
            if (worker.signaled()) {
                return EXIT_FAILURE;
            }

            if (!xferBenchUtils::validateTransfer(false, local_trans_lists, local_trans_lists)) {
                return EXIT_FAILURE;
            }
            if (!xferBenchUtils::printStats(true, block_size, batch_size, xferBenchStats())) {
                return EXIT_FAILURE;
            }
        } else if (worker.isInitiator()) {
            std::vector<std::vector<xferBenchIOV>> remote_trans_lists(
                worker.exchangeIOV(local_trans_lists, block_size));

            auto result = worker.transfer(block_size, local_trans_lists, remote_trans_lists);
            if (std::holds_alternative<int>(result)) {
                return 1;
            }

            if (!xferBenchUtils::validateTransfer(true, local_trans_lists, remote_trans_lists)) {
                return EXIT_FAILURE;
            }

            if (!xferBenchUtils::printStats(
                    false, block_size, batch_size, std::get<xferBenchStats>(result))) {
                return EXIT_FAILURE;
            }
        }
        if (xferBenchConfig::mode == XFERBENCH_MODE_SG &&
            xferBenchConfig::scheme == XFERBENCH_SCHEME_MANY_TO_ONE) {
            // Complete summary-message cleanup before the next IOV exchange reuses its keys.
            worker.synchronize();
        }
        if (batch_size > xferBenchConfig::max_batch_size / 2) {
            break;
        }
    }

    return 0;
}

namespace {
std::unique_ptr<xferBenchWorker>
createWorker() {
    if (xferBenchConfig::worker_type == XFERBENCH_WORKER_NIXL) {
        std::vector<std::string> devices = xferBenchConfig::parseDeviceList();
        if (devices.empty()) {
            std::cerr << "Failed to parse device list" << std::endl;
            return nullptr;
        }
        return std::make_unique<xferBenchNixlWorker>(devices);
    } else if (xferBenchConfig::worker_type == XFERBENCH_WORKER_NVSHMEM) {
#if HAVE_NVSHMEM && HAVE_CUDA
        return std::make_unique<xferBenchNvshmemWorker>();
#else
        std::cerr << "NVSHMEM worker requested but NVSHMEM or CUDA is not available" << std::endl;
        return nullptr;
#endif
    } else {
        std::cerr << "Unsupported worker type: " << xferBenchConfig::worker_type << std::endl;
        return nullptr;
    }
}
} // namespace

static int
runBenchmark() {
    int ret = 0;
    int num_workers = xferBenchConfig::workerNum();

    // Create the appropriate worker based on worker configuration
    std::unique_ptr<xferBenchWorker> worker_ptr = createWorker();
    if (!worker_ptr) {
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, worker_ptr->signalHandler);
    std::signal(SIGTERM, worker_ptr->signalHandler);

    // Ensure all processes are ready before exchanging metadata
    ret = worker_ptr->synchronizeStart();
    if (0 != ret) {
        std::cerr << "Failed to synchronize all processes" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> iov_lists = worker_ptr->allocateMemory(num_workers);
    auto mem_guard = make_scope_guard ([&] {
        worker_ptr->deallocateMemory(iov_lists);
    });

    ret = worker_ptr->exchangeMetadata();
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    if (worker_ptr->isInitiator() && worker_ptr->isMasterRank()) {
        if (!xferBenchConfig::plugin_parameters) {
            xferBenchConfig::printConfig();
        }
        xferBenchUtils::printStatsHeader();
    }

    for (size_t block_size = xferBenchConfig::start_block_size;
         !worker_ptr->signaled() &&
         block_size <= xferBenchConfig::max_block_size;
         block_size *= 2) {
        ret = processBatchSizes(*worker_ptr, iov_lists, block_size, num_workers);
        if (0 != ret) {
            return EXIT_FAILURE;
        }
        if (block_size > xferBenchConfig::max_block_size / 2) {
            break;
        }
    }

    ret = worker_ptr->synchronize(true); // Make sure environment is not used anymore
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    return worker_ptr->signaled() ? EXIT_FAILURE : EXIT_SUCCESS;
}

int
main(int argc, char *argv[]) {
#if HAVE_RAW_CLI
    if (nixlbench::isRawCommand(argc, argv)) {
        const auto result = nixlbench::prepareRawCommand(argc, argv, std::cout, std::cerr);
        if (result.status != EXIT_SUCCESS || !result.execute) {
            return result.status;
        }
        return runBenchmark();
    }
#endif

    // Preserve the flags-only interface by routing every non-raw invocation directly to gflags.
    if (xferBenchConfig::parseConfig(argc, argv) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return runBenchmark();
}
