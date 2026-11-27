/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "utils.h"

namespace nixlbench {

// Size of each role-local device allocation before backend alignment.
inline size_t
deviceBufferSize(bool initiator, int num_threads) {
    const int devices =
        initiator ? xferBenchConfig::num_initiator_dev : xferBenchConfig::num_target_dev;
    return xferBenchConfig::total_buffer_size / (devices * num_threads);
}

inline std::pair<size_t, size_t>
getStrideScheme(bool initiator, int num_threads) {
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
        if (initiator) {
            count = target_device;
        }
    } else if (XFERBENCH_SCHEME_MANY_TO_ONE == xferBenchConfig::scheme) {
        if (!initiator) {
            count = initiator_device;
            if (XFERBENCH_MODE_SG == xferBenchConfig::mode) {
                buffer_size = xferBenchConfig::total_buffer_size / num_threads;
            }
        }
    } else if (XFERBENCH_SCHEME_TP == xferBenchConfig::scheme) {
        if (initiator) {
            if (initiator_device < target_device) {
                count = target_device / initiator_device;
            }
        } else if (!initiator) {
            if (target_device < initiator_device) {
                count = initiator_device / target_device;
            }
        }
    }
    stride = buffer_size / count;

    // For hugepages: ensure stride is a multiple of 2MB so addresses stay 2MB-aligned
    // This ensures dev_offset = (i * stride) % iov.len is always a multiple of 2MB
    if (xferBenchConfig::use_hugepages) {
        stride = ROUND_UP(stride, HUGEPAGE_SIZE);
    }

    return std::make_pair(count, stride);
}

inline std::vector<std::vector<xferBenchIOV>>
createTransferDescLists(bool initiator,
                        const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                        size_t block_size,
                        size_t batch_size,
                        int num_threads) {
    auto [count, stride] = getStrideScheme(initiator, num_threads);
    std::vector<std::vector<xferBenchIOV>> xfer_lists;

    for (const auto &iov_list : iov_lists) {
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

} // namespace nixlbench
