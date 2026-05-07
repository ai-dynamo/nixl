/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/transfer_descriptor_strategy.h"

#include "benchmark/nixl_storage_allocator.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>

namespace nixlbench {
namespace {

std::pair<std::size_t, std::size_t>
getStrideScheme(const transferDescriptorConfig &config) {
    std::size_t count = 1;
    const std::size_t buffer_size =
        config.total_buffer_size /
        (static_cast<std::size_t>(config.num_initiator_dev) *
         static_cast<std::size_t>(config.num_threads));

    if (config.scheme == XFERBENCH_SCHEME_ONE_TO_MANY) {
        if (config.is_initiator) {
            count = static_cast<std::size_t>(config.num_target_dev);
        }
    } else if (config.scheme == XFERBENCH_SCHEME_MANY_TO_ONE) {
        if (config.is_target) {
            count = static_cast<std::size_t>(config.num_initiator_dev);
        }
    } else if (config.scheme == XFERBENCH_SCHEME_TP) {
        if (config.is_initiator) {
            if (config.num_initiator_dev < config.num_target_dev) {
                count = static_cast<std::size_t>(config.num_target_dev / config.num_initiator_dev);
            }
        } else if (config.is_target) {
            if (config.num_target_dev < config.num_initiator_dev) {
                count = static_cast<std::size_t>(config.num_initiator_dev / config.num_target_dev);
            }
        }
    }

    return std::make_pair(count, buffer_size / count);
}

bool
isConfigValid(const transferDescriptorConfig &config) {
    return config.block_size > 0 && config.batch_size > 0 && config.num_threads > 0 &&
        config.num_initiator_dev > 0 && config.num_target_dev > 0 && config.total_buffer_size > 0;
}

} // namespace

fixedIterationPolicy::fixedIterationPolicy(int iterations, benchmarkAllocationLifecycle lifecycle)
    : remaining_(iterations),
      lifecycle_(lifecycle) {}

bool
fixedIterationPolicy::allocateOnce() const {
    return lifecycle_ == benchmarkAllocationLifecycle::AllocateOnce;
}

bool
fixedIterationPolicy::hasNext() const {
    return remaining_ > 0;
}

void
fixedIterationPolicy::advance() {
    --remaining_;
}

transferDescriptorConfig
makeTransferDescriptorConfig(const benchmarkConfig &config,
                             std::size_t block_size,
                             std::size_t batch_size,
                             bool is_initiator,
                             bool is_target) {
    transferDescriptorConfig descriptor_config;
    descriptor_config.block_size = block_size;
    descriptor_config.batch_size = batch_size;
    descriptor_config.num_threads = config.transfer.num_threads;
    descriptor_config.num_initiator_dev = config.worker.num_initiator_dev;
    descriptor_config.num_target_dev = config.worker.num_target_dev;
    descriptor_config.total_buffer_size = config.transfer.total_buffer_size;
    descriptor_config.scheme = config.transfer.scheme;
    descriptor_config.mode = config.transfer.mode;
    descriptor_config.is_initiator = is_initiator;
    descriptor_config.is_target = is_target;
    return descriptor_config;
}

std::unique_ptr<transferDescriptorStrategy>
makeTransferDescriptorStrategy(const benchmarkConfig &config,
                               bool randomized_rw_location,
                               remoteIovStrategy *remote_strategy,
                               bool is_initiator,
                               bool is_target) {
    auto descriptor_config = makeTransferDescriptorConfig(config,
                                                          config.transfer.start_block_size,
                                                          config.transfer.start_batch_size,
                                                          is_initiator,
                                                          is_target);
    if (remote_strategy != nullptr) {
        return std::make_unique<remoteOffsetTransferDescriptorStrategy>(descriptor_config,
                                                                        *remote_strategy,
                                                                        randomized_rw_location);
    }

    return std::make_unique<offsetTransferDescriptorStrategy>(descriptor_config,
                                                              randomized_rw_location);
}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
createTransferDescLists(const transferDescriptorConfig &config,
                        const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                        bool randomized_rw_location,
                        std::mt19937 *rng) {
    if (!isConfigValid(config)) {
        std::cerr << "Invalid transfer descriptor configuration" << std::endl;
        return EXIT_FAILURE;
    }

    auto [count, stride] = getStrideScheme(config);
    std::vector<std::vector<xferBenchIOV>> xfer_lists;
    xfer_lists.reserve(iov_lists.size());
    std::random_device rd;
    std::mt19937 default_rng(rd());
    std::mt19937 &shuffle_rng = rng != nullptr ? *rng : default_rng;

    for (const auto &iov_list : iov_lists) {
        std::vector<xferBenchIOV> xfer_list;

        for (const auto &iov : iov_list) {
            if (iov.len == 0) {
                return EXIT_FAILURE;
            }

            std::vector<std::size_t> indices(count);
            std::iota(indices.begin(), indices.end(), 0);
            if (randomized_rw_location) {
                std::shuffle(indices.begin(), indices.end(), shuffle_rng);
            }

            for (std::size_t i = 0; i < count; i++) {
                std::size_t dev_offset = ((indices[i] * stride) % iov.len);

                for (std::size_t j = 0; j < config.batch_size; j++) {
                    std::size_t block_offset = ((j * config.block_size) % iov.len);
                    if (block_offset + config.block_size > iov.len) {
                        block_offset = 0;
                    }
                    xfer_list.push_back(xferBenchIOV((iov.addr + dev_offset) + block_offset,
                                                     config.block_size,
                                                     iov.devId,
                                                     iov.metaInfo));
                }
            }
        }

        xfer_lists.push_back(xfer_list);
    }

    return xfer_lists;
}

offsetTransferDescriptorStrategy::offsetTransferDescriptorStrategy(
    transferDescriptorConfig config,
    bool randomized_rw_location)
    : config_(std::move(config)),
      randomized_rw_location_(randomized_rw_location) {}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
offsetTransferDescriptorStrategy::create(const benchmarkAllocation &allocation) {
    return createTransferDescLists(config_, allocation.local_iovs, randomized_rw_location_);
}

remoteOffsetTransferDescriptorStrategy::remoteOffsetTransferDescriptorStrategy(
    transferDescriptorConfig config,
    remoteIovStrategy &remote_strategy,
    bool randomized_rw_location)
    : config_(std::move(config)),
      remote_strategy_(remote_strategy),
      randomized_rw_location_(randomized_rw_location) {}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
remoteOffsetTransferDescriptorStrategy::create(const benchmarkAllocation &allocation) {
    auto local_result = createTransferDescLists(config_,
                                               allocation.local_iovs,
                                               randomized_rw_location_);
    if (std::holds_alternative<int>(local_result)) {
        return std::get<int>(local_result);
    }

    return remote_strategy_.createTransferIovs(
        std::get<std::vector<std::vector<xferBenchIOV>>>(local_result), config_.block_size);
}

} // namespace nixlbench
