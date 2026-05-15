/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef NIXL_SRC_UTILS_COMMON_HW_INFO_H
#define NIXL_SRC_UTILS_COMMON_HW_INFO_H

namespace nixl {

/**
 * @brief Hardware information gathered by scanning PCI devices.
 *
 * Scans the sysfs PCI device directory to detect available hardware.
 */
class hwInfo {
public:
    /** Number of NVIDIA GPUs detected via PCI vendor 0x10de. */
    unsigned numNvidiaGpus = 0;
    /** Number of AMD GPUs detected via PCI vendor 0x1002. */
    unsigned numAmdGpus = 0;
    /** Number of Mellanox InfiniBand HCAs detected via PCI vendor 0x15b3 + class 0x0207. */
    unsigned numIbDevices = 0;
    /** Number of AWS Elastic Fabric Adapters detected via PCI vendor 0x1d0f + EFA device IDs. */
    unsigned numEfaDevices = 0;

    /** Return a cached singleton instance of hwInfo. */
    [[nodiscard]] static const hwInfo &
    instance();

private:
    /**
     * Construct hwInfo by scanning /sys/bus/pci/devices once. Counts populated
     * fields on the public surface; failed sysfs reads are logged at TRACE
     * level and skipped. Use ::instance() to obtain the cached singleton.
     */
    hwInfo();
    hwInfo(const hwInfo &) = delete;
    hwInfo &
    operator=(const hwInfo &) = delete;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_HW_INFO_H
