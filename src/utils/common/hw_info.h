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
 * @brief Hardware information gathered by scanning PCI devices.
 *
 * Scans the sysfs PCI device directory to detect available
 * NVIDIA GPUs and InfiniBand devices on the system.
 */
struct hwInfo {
    unsigned numNvidiaGpus = 0;
    unsigned numIbDevices = 0;

    hwInfo();
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_HW_INFO_H
