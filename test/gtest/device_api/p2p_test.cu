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

#include "p2p_test.h"

namespace nixl::device_api {
p2pTest::p2pTest(const std::vector<size_t> &sizes)
    : sender_("sender", getNumChannels()),
      receiver_("receiver", getNumChannels()),
      sizes_(sizes) {}

void
p2pTest::addDataBuffers() {
    for (const auto &size : sizes_) {
        srcBuffers_.emplace_back(size);
        dstBuffers_.emplace_back(size, getDstMemType());
    }
}

void
p2pTest::addSignalBuffers() {
    const size_t signal_size = receiver_.getGpuSignalSize();
    srcBuffers_.emplace_back(signal_size);
    dstBuffers_.emplace_back(signal_size, getDstMemType());
}

nixlGpuXferReqH
p2pTest::createGpuXferReq() {
    sender_.registerMem(srcBuffers_);
    receiver_.registerMem(dstBuffers_);
    receiver_.prepGpuSignal(dstBuffers_.back());
    sender_.loadRemoteMD(receiver_.getLocalMD());
    receiver_.loadRemoteMD(sender_.getLocalMD());
    return sender_.createGpuXferReq(srcBuffers_, dstBuffers_);
}
} // namespace nixl::device_api
