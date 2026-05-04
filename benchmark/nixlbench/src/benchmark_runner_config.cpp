/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_runner_config.h"

#include "utils/utils.h"

namespace nixlbench {

std::pair<std::size_t, std::size_t>
getStrideScheme(const benchmarkConfig &config,
                bool is_initiator,
                bool is_target,
                int num_threads) {
    const int initiator_device = config.worker.num_initiator_dev;
    const int target_device = config.worker.num_target_dev;

    std::size_t count = 1;
    const std::size_t buffer_size =
        config.transfer.total_buffer_size /
        (static_cast<std::size_t>(initiator_device) * static_cast<std::size_t>(num_threads));

    if (config.transfer.scheme == XFERBENCH_SCHEME_ONE_TO_MANY) {
        if (is_initiator) {
            count = target_device;
        }
    } else if (config.transfer.scheme == XFERBENCH_SCHEME_MANY_TO_ONE) {
        if (is_target) {
            count = initiator_device;
        }
    } else if (config.transfer.scheme == XFERBENCH_SCHEME_TP) {
        if (is_initiator) {
            if (initiator_device < target_device) {
                count = target_device / initiator_device;
            }
        } else if (is_target) {
            if (target_device < initiator_device) {
                count = initiator_device / target_device;
            }
        }
    }

    return std::make_pair(count, buffer_size / count);
}

} // namespace nixlbench
