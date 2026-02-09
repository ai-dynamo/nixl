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

#ifndef OBJ_PLUGIN_AWS_SDK_INIT_H
#define OBJ_PLUGIN_AWS_SDK_INIT_H

#include <aws/core/Aws.h>
#include <mutex>
#include <cstdlib>

namespace nixl_s3_utils {

/**
 * Initialize the AWS SDK in a thread-safe manner.
 * This function uses std::call_once to ensure that Aws::InitAPI is called
 * exactly once, even in multi-threaded environments or when multiple S3 clients
 * are created.
 *
 * The AWS SDK is automatically shut down at program exit via std::atexit.
 */
inline void
initAWSSDK() {
    static std::once_flag aws_init_flag;
    static Aws::SDKOptions *aws_options = nullptr;

    std::call_once(aws_init_flag, []() {
        aws_options = new Aws::SDKOptions();
        Aws::InitAPI(*aws_options);

        // Register cleanup at program exit
        std::atexit([]() {
            Aws::ShutdownAPI(*aws_options);
            delete aws_options;
        });
    });
}

} // namespace nixl_s3_utils

#endif // OBJ_PLUGIN_AWS_SDK_INIT_H
