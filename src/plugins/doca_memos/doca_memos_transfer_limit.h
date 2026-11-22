/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_DOCA_MEMOS_TRANSFER_LIMIT_H
#define NIXL_DOCA_MEMOS_TRANSFER_LIMIT_H

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

// MXVL describes stored values, not the kernel's single-command transfer cap.
// Like DOCA's NVMe-kernel transport, map ngXnY to the nvmeXnY queue in sysfs.
inline uint64_t
nixlMemosKernelTransferLimit(const std::string &device,
                             const std::filesystem::path &sysfs = "/sys") {
    std::error_code error;
    auto path = std::filesystem::canonical(device, error);
    const auto name = (error ? std::filesystem::path(device) : path).filename().string();
    unsigned controller, ns;
    char extra;
    if (std::sscanf(name.c_str(), "ng%un%u%c", &controller, &ns, &extra) != 2 &&
        std::sscanf(name.c_str(), "nvme%un%u%c", &controller, &ns, &extra) != 2) {
        return 0;
    }
    const auto block = "nvme" + std::to_string(controller) + "n" + std::to_string(ns);
    std::ifstream input(sysfs / "block" / block / "queue" / "max_hw_sectors_kb");
    uint64_t kib = 0;
    if (!(input >> kib) || kib == 0 || kib > UINT32_MAX / 1024) {
        return 0;
    }
    return kib * 1024;
}

inline uint32_t
nixlMemosChunkSize(uint32_t namespace_max, uint64_t kernel_max, uint32_t override_bytes) {
    if (namespace_max == 0 || (kernel_max == 0 && override_bytes == 0)) {
        return 0;
    }
    uint64_t size = namespace_max;
    if (kernel_max) {
        size = std::min(size, kernel_max);
    }
    if (override_bytes) {
        size = std::min(size, uint64_t{override_bytes});
    }
    return static_cast<uint32_t>(size);
}

#endif // NIXL_DOCA_MEMOS_TRANSFER_LIMIT_H
