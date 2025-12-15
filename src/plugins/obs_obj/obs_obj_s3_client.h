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

#ifndef OBS_OBJ_S3_CLIENT_H
#define OBS_OBJ_S3_CLIENT_H

#include "s3_client/s3_client.h"

class obsObjS3Client : public awsS3Client {
public:
    obsObjS3Client(nixl_b_params_t *custom_params,
                   std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr)
        : awsS3Client(custom_params, executor) {}

    void
    putObjectRdmaAsync(std::string_view key,
                      uintptr_t data_ptr,
                      size_t data_len,
                      size_t offset,
                      const std::string &rdma_desc,
                      put_object_callback_t callback);

    void
    getObjectRdmaAsync(std::string_view key,
                      uintptr_t data_ptr,
                      size_t data_len,
                      size_t offset,
                      const std::string &rdma_desc,
                      get_object_callback_t callback);

};

#endif // OBS_OBJ_S3_CLIENT_H
