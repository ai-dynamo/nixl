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

#include "config.h"
#include <iostream>
#include <nixl.h>
#include <sys/time.h>
#include "utils/utils.h"
#include "utils/scope_guard.h"
#include "worker/nixl/nixl_worker.h"
#if HAVE_NVSHMEM && HAVE_CUDA
#include "worker/nvshmem/nvshmem_worker.h"
#endif
#include <unistd.h>
#include <memory>
#include <csignal>

static std::pair<size_t, size_t> getStrideScheme(xferBenchWorker &worker, int num_threads) {
    int initiator_device, target_device;
    size_t buffer_size, count, stride;

    initiator_device = xferBenchConfig::num_initiator_dev;
    target_device = xferBenchConfig::num_target_dev;

    // Default value
    count = 1;
    buffer_size = xferBenchConfig::total_buffer_size / (initiator_device * num_threads);

    // TODO: add macro for schemes
    // Maybe, we can squeze ONE_TO_MANY and MANY_TO_ONE into TP scheme
    if (XFERBENCH_SCHEME_ONE_TO_MANY == xferBenchConfig::scheme) {
        if (worker.isInitiator()) {
            count = target_device;
        }
    } else if (XFERBENCH_SCHEME_MANY_TO_ONE == xferBenchConfig::scheme) {
        if (worker.isTarget()) {
            count = initiator_device;
        }
    } else if (XFERBENCH_SCHEME_TP == xferBenchConfig::scheme) {
        if (worker.isInitiator()) {
            if (initiator_device < target_device) {
                count = target_device / initiator_device;
            }
        } else if (worker.isTarget()) {
            if (target_device < initiator_device) {
                count = initiator_device / target_device;
            }
        }
    }
    stride = buffer_size / count;

    return std::make_pair(count, stride);
}

static std::vector<std::vector<xferBenchIOV>> createTransferDescLists(xferBenchWorker &worker,
                                                                      std::vector<std::vector<xferBenchIOV>> &iov_lists,
                                                                      size_t block_size,
                                                                      size_t batch_size,
                                                                      int num_threads) {
    auto [count, stride] = getStrideScheme(worker, num_threads);
    std::vector<std::vector<xferBenchIOV>> xfer_lists;

    for (const auto &iov_list: iov_lists) {
        std::vector<xferBenchIOV> xfer_list;

        for (const auto &iov : iov_list) {
            for (size_t i = 0; i < count; i++) {
                size_t dev_offset = ((i * stride) % iov.len);

                for (size_t j = 0; j < batch_size; j++) {
                    size_t block_offset = ((j * block_size) % iov.len);
                    if (block_offset + block_size > iov.len) {
                        // Prevent memory overflow when iov.len is not divisible by block_size
                        block_offset = 0;
                    }
                    xfer_list.push_back(xferBenchIOV((iov.addr + dev_offset) + block_offset,
                                                     block_size,
                                                     iov.devId,
                                                     iov.metaInfo));
                }
            }
        }

        xfer_lists.push_back(xfer_list);
    }

    return xfer_lists;
}

static int processBatchSizes(xferBenchWorker &worker,
                             std::vector<std::vector<xferBenchIOV>> &iov_lists,
                             size_t block_size, int num_threads) {
    for (size_t batch_size = xferBenchConfig::start_batch_size;
         !worker.signaled() &&
             batch_size <= xferBenchConfig::max_batch_size;
         batch_size *= 2) {
        auto local_trans_lists = createTransferDescLists(worker,
                                                         iov_lists,
                                                         block_size,
                                                         batch_size,
                                                         num_threads);

        if (worker.isTarget()) {
            if (xferBenchConfig::isStorageBackend()) {
                std::cerr << "storage backend should be always an initiator" << std::endl;
                return EXIT_FAILURE;
            }

            worker.exchangeIOV(local_trans_lists, block_size);
            worker.poll(block_size);

            if (xferBenchConfig::check_consistency && xferBenchConfig::op_type == XFERBENCH_OP_WRITE) {
                xferBenchUtils::checkConsistency(local_trans_lists);
            }
            if (IS_PAIRWISE_AND_SG()) {
                // TODO: This is here just to call throughput reduction
                // Separate reduction and print
                xferBenchUtils::printStats(true, block_size, batch_size, xferBenchStats());
            }
        } else if (worker.isInitiator()) {
            std::vector<std::vector<xferBenchIOV>> remote_trans_lists(
                worker.exchangeIOV(local_trans_lists, block_size));

            auto result = worker.transfer(block_size, local_trans_lists, remote_trans_lists);
            if (std::holds_alternative<int>(result)) {
                return 1;
            }

            if (xferBenchConfig::check_consistency) {
                if (xferBenchConfig::op_type == XFERBENCH_OP_READ) {
                    xferBenchUtils::checkConsistency(local_trans_lists);
                } else if (xferBenchConfig::op_type == XFERBENCH_OP_WRITE) {
                    // Only storage backends support consistency check for write on initiator
                    if (xferBenchConfig::isStorageBackend()) {
                        xferBenchUtils::checkConsistency(remote_trans_lists);
                    }
                }
            }

            xferBenchUtils::printStats(
                false, block_size, batch_size, std::get<xferBenchStats>(result));
        }
    }

    return 0;
}

static std::unique_ptr<xferBenchWorker> createWorker(int *argc, char ***argv) {
    if (xferBenchConfig::worker_type == "nixl") {
        std::vector<std::string> devices = xferBenchConfig::parseDeviceList();
        if (devices.empty()) {
            std::cerr << "Failed to parse device list" << std::endl;
            return nullptr;
        }
        return std::make_unique<xferBenchNixlWorker>(argc, argv, devices);
    } else if (xferBenchConfig::worker_type == "nvshmem") {
#if HAVE_NVSHMEM && HAVE_CUDA
        return std::make_unique<xferBenchNvshmemWorker>(argc, argv);
#else
        std::cerr << "NVSHMEM worker requested but NVSHMEM or CUDA is not available" << std::endl;
        return nullptr;
#endif
    } else {
        std::cerr << "Unsupported worker type: " << xferBenchConfig::worker_type << std::endl;
        return nullptr;
    }
}

int main(int argc, char *argv[]) {
    int ret = xferBenchConfig::parseConfig(argc, argv);
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    int num_threads = xferBenchConfig::num_threads;

    // Create the appropriate worker based on worker configuration
    std::unique_ptr<xferBenchWorker> worker_ptr = createWorker(&argc, &argv);
    if (!worker_ptr) {
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, worker_ptr->signalHandler);

    // Ensure all processes are ready before exchanging metadata
    ret = worker_ptr->synchronizeStart();
    if (0 != ret) {
        std::cerr << "Failed to synchronize all processes" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> iov_lists = worker_ptr->allocateMemory(num_threads);
    auto mem_guard = make_scope_guard ([&] {
        worker_ptr->deallocateMemory(iov_lists);
    });

    ret = worker_ptr->exchangeMetadata();
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    if (worker_ptr->isInitiator() && worker_ptr->isMasterRank()) {
        xferBenchConfig::printConfig();
        xferBenchUtils::printStatsHeader();
    }

    for (size_t block_size = xferBenchConfig::start_block_size;
         !worker_ptr->signaled() &&
         block_size <= xferBenchConfig::max_block_size;
         block_size *= 2) {
        ret = processBatchSizes(*worker_ptr, iov_lists, block_size, num_threads);
        if (0 != ret) {
            return EXIT_FAILURE;
        }
    }

    ret = worker_ptr->synchronize(); // Make sure environment is not used anymore
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    return worker_ptr->signaled() ? EXIT_FAILURE : EXIT_SUCCESS;
}
