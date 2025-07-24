/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef __PLUGIN_TEST_H
#define __PLUGIN_TEST_H

#include <gtest/gtest.h>
#include <absl/log/check.h>
#include "backend_engine.h"

namespace gtest::plugins {
/*
 * Base class for all plugin tests.
 */
class setupBackendTestFixture : public testing::TestWithParam<nixlBackendInitParams> {
protected:
    std::shared_ptr<nixlBackendEngine> localBackendEngine_;
    std::shared_ptr<nixlBackendEngine> remoteBackendEngine_;

    void
    SetUp() {
        ASSERT_FALSE(localBackendEngine_->getInitErr())
            << "Failed to initialize local backend engine";
        if (remoteBackendEngine_) {
            ASSERT_FALSE(remoteBackendEngine_->getInitErr())
                << "Failed to initialize remote backend engine";
        }
    }
};

} // namespace gtest::plugins
#endif // __PLUGIN_TEST_H
