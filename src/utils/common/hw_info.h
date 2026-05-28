/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief Lazily-detected, process-wide snapshot of the host's GPU and RDMA NIC inventory.
 *
 * Populated once on first call to @ref instance() by scanning PCI devices under
 * @c /sys/bus/pci/devices and matching vendor/class IDs. No CUDA, ROCm, or libibverbs
 * runtime is required. All fields are zero if the sysfs scan fails or no matching
 * devices are present.
 */
class hwInfo {
public:
    /** Number of NVIDIA GPUs found on the PCI bus (vendor 0x10de, GPU class). */
    unsigned numNvidiaGpus = 0;
    /** Number of AMD GPUs found on the PCI bus (vendor 0x1002, GPU class). */
    unsigned numAmdGpus = 0;
    /** Number of Mellanox InfiniBand / RoCE devices found on the PCI bus (vendor 0x15b3, class 0x0207). */
    unsigned numIbDevices = 0;
    /** Number of AWS EFA devices found on the PCI bus (vendor 0x1d0f, EFA device IDs). */
    unsigned numEfaDevices = 0;

    /**
     * @brief Returns the process-wide singleton, populating it via the PCI sysfs scan
     *        on first call. Subsequent calls return the cached snapshot.
     * @return Const reference to the cached @ref hwInfo instance.
     */
    [[nodiscard]] static const hwInfo &
    instance();

private:
    hwInfo();
    hwInfo(const hwInfo &) = delete;
    hwInfo &
    operator=(const hwInfo &) = delete;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_HW_INFO_H
