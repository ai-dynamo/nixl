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

/**
 * Unit tests for the Scality AI Connector DRAM-across-NICs device mapping.
 *
 * targetCudaDevice() decides which CUDA device a buffer's MR (and thus its
 * cuObject primary GID / NIC) is bound to. It needs no CUDA or cuObject, so
 * this test is always built.
 */
#include <gtest/gtest.h>

#include "rest_accel/scality_ai_connector/device_select.h"

namespace gtest::obj {

/** VRAM binds to the buffer's own GPU, unchanged regardless of GPU count. */
TEST(DeviceSelectTest, VramBindsToOwnGpu) {
    EXPECT_EQ(targetCudaDevice(VRAM_SEG, 0, 8), 0);
    EXPECT_EQ(targetCudaDevice(VRAM_SEG, 3, 8), 3);
    EXPECT_EQ(targetCudaDevice(VRAM_SEG, 7, 8), 7);
    // VRAM mapping ignores gpuCount (the buffer is a real device pointer).
    EXPECT_EQ(targetCudaDevice(VRAM_SEG, 5, 0), 5);
}

/** DRAM spreads round-robin across the available GPUs. */
TEST(DeviceSelectTest, DramSpreadsAcrossGpus) {
    EXPECT_EQ(targetCudaDevice(DRAM_SEG, 0, 4), 0);
    EXPECT_EQ(targetCudaDevice(DRAM_SEG, 1, 4), 1);
    EXPECT_EQ(targetCudaDevice(DRAM_SEG, 2, 4), 2);
    EXPECT_EQ(targetCudaDevice(DRAM_SEG, 3, 4), 3);
    EXPECT_EQ(targetCudaDevice(DRAM_SEG, 4, 4), 0);
    EXPECT_EQ(targetCudaDevice(DRAM_SEG, 5, 4), 1);
}

/** DRAM with no GPUs present leaves the current device unchanged. */
TEST(DeviceSelectTest, DramNoGpusKeepsCurrentDevice) {
    EXPECT_EQ(targetCudaDevice(DRAM_SEG, 0, 0), -1);
    EXPECT_EQ(targetCudaDevice(DRAM_SEG, 3, 0), -1);
}

/** Other segment types never switch the device. */
TEST(DeviceSelectTest, OtherSegmentsKeepCurrentDevice) {
    EXPECT_EQ(targetCudaDevice(OBJ_SEG, 1, 8), -1);
    EXPECT_EQ(targetCudaDevice(BLK_SEG, 1, 8), -1);
}

} // namespace gtest::obj
