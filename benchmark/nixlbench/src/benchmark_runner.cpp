/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_runner.h"

#include "config.h"
#include "utils/scope_guard.h"
#include "utils/utils.h"
#include "worker/nixl/nixl_worker.h"
#if HAVE_NVSHMEM && HAVE_CUDA
#include "worker/nvshmem/nvshmem_worker.h"
#endif

#include <charconv>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>
#include <variant>

std::pair<size_t, size_t>
getStrideScheme(xferBenchWorker &worker, int num_threads) {
    int initiator_device, target_device;
    size_t buffer_size, count, stride;

    initiator_device = xferBenchConfig::num_initiator_dev;
    target_device = xferBenchConfig::num_target_dev;

    count = 1;
    buffer_size = xferBenchConfig::total_buffer_size / (initiator_device * num_threads);

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

std::vector<std::vector<xferBenchIOV>>
createTransferDescLists(xferBenchWorker &worker,
                        std::vector<std::vector<xferBenchIOV>> &iov_lists,
                        size_t block_size,
                        size_t batch_size,
                        int num_threads,
                        bool randomized_read_location) {
    auto [count, stride] = getStrideScheme(worker, num_threads);
    std::vector<std::vector<xferBenchIOV>> xfer_lists;

    for (const auto &iov_list : iov_lists) {
        std::vector<xferBenchIOV> xfer_list;

        for (const auto &iov : iov_list) {
            for (size_t i = 0; i < count; i++) {
                size_t dev_offset = ((i * stride) % iov.len);

                for (size_t j = 0; j < batch_size; j++) {
                    size_t block_offset = ((j * block_size) % iov.len);
                    if (block_offset + block_size > iov.len) {
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

int
processBatchSizes(xferBenchWorker &worker,
                  std::vector<std::vector<xferBenchIOV>> &iov_lists,
                  size_t block_size,
                  int num_threads,
                  bool randomized_read_location) {
    for (size_t batch_size = xferBenchConfig::start_batch_size;
         !worker.signaled() && batch_size <= xferBenchConfig::max_batch_size;
         batch_size *= 2) {
        auto local_trans_lists =
            createTransferDescLists(worker, iov_lists, block_size, batch_size, num_threads, randomized_read_location);

        if (worker.isTarget()) {
            if (xferBenchConfig::isStorageBackend()) {
                std::cerr << "storage backend should be always an initiator" << std::endl;
                return EXIT_FAILURE;
            }

            worker.exchangeIOV(local_trans_lists, block_size);
            worker.poll(block_size);

            if (!xferBenchUtils::validateTransfer(false, local_trans_lists, local_trans_lists)) {
                return EXIT_FAILURE;
            }
            if (IS_PAIRWISE_AND_SG()) {
                xferBenchUtils::printStats(true, block_size, batch_size, xferBenchStats());
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

            xferBenchUtils::printStats(
                false, block_size, batch_size, std::get<xferBenchStats>(result));
        }
    }

    return 0;
}

std::unique_ptr<xferBenchWorker>
createWorker() {
    if (xferBenchConfig::worker_type == "nixl") {
        std::vector<std::string> devices = xferBenchConfig::parseDeviceList();
        if (devices.empty()) {
            std::cerr << "Failed to parse device list" << std::endl;
            return nullptr;
        }
        return std::make_unique<xferBenchNixlWorker>(devices);
    } else if (xferBenchConfig::worker_type == "nvshmem") {
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

size_t
parse_file_size(const std::string &input) {
    if (input.empty()) return 0;

    size_t suffix_pos = input.find_first_not_of("0123456789");
    const char *number_end = suffix_pos == std::string_view::npos ? input.data() + input.size() :
                                                                    input.data() + suffix_pos;

    size_t value = 0;
    auto [ptr, ec] = std::from_chars(input.data(), number_end, value);

    if (ec != std::errc{} || ptr != number_end) return 0;

    if (suffix_pos == std::string_view::npos) return value;

    std::string_view suffix(input.data() + suffix_pos, input.size() - suffix_pos);
    auto to_upper = [](char c) {
        return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    };

    size_t multiplier = 1;
    switch (to_upper(suffix[0])) {
    case 'K':
        multiplier = 1000LL;
        break;
    case 'M':
        multiplier = 1000000LL;
        break;
    case 'G':
        multiplier = 1000000000LL;
        break;
    case 'T':
        multiplier = 1000000000000LL;
        break;
    default:
        return value;
    }

    if (suffix.size() >= 2 && to_upper(suffix[1]) == 'I') {
        switch (to_upper(suffix[0])) {
        case 'K':
            multiplier = 1LL << 10;
            break;
        case 'M':
            multiplier = 1LL << 20;
            break;
        case 'G':
            multiplier = 1LL << 30;
            break;
        case 'T':
            multiplier = 1LL << 40;
            break;
        default:
            break;
        }
    }

    return value * multiplier;
}

int
runBenchmarkWithCurrentConfig() {
    int num_threads = xferBenchConfig::num_threads;

    std::unique_ptr<xferBenchWorker> worker_ptr = createWorker();
    if (!worker_ptr) {
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, worker_ptr->signalHandler);

    int ret = worker_ptr->synchronizeStart();
    if (0 != ret) {
        std::cerr << "Failed to synchronize all processes" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> iov_lists = worker_ptr->allocateMemory(num_threads);
    auto mem_guard = make_scope_guard([&] { worker_ptr->deallocateMemory(iov_lists); });

    ret = worker_ptr->exchangeMetadata();
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    if (worker_ptr->isInitiator() && worker_ptr->isMasterRank()) {
        xferBenchConfig::printConfig();
        xferBenchUtils::printStatsHeader();
    }

    for (size_t block_size = xferBenchConfig::start_block_size;
         !worker_ptr->signaled() && block_size <= xferBenchConfig::max_block_size;
         block_size *= 2) {
        ret = processBatchSizes(*worker_ptr, iov_lists, block_size, num_threads);
        if (0 != ret) {
            return EXIT_FAILURE;
        }
    }

    ret = worker_ptr->synchronize();
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    return worker_ptr->signaled() ? EXIT_FAILURE : EXIT_SUCCESS;
}
