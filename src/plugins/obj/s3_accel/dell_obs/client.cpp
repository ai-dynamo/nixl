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

#include "client.h"
#include <iostream>

awsDellOBSClient::awsDellOBSClient(nixl_b_params_t *custom_params,
                                   std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3AccelClient(custom_params, executor) {
    // Dell OBS-specific initialization can be added here
    std::cout << "Initialized Dell OBS Object Client" << std::endl;
}
