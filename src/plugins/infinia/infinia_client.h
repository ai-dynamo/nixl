/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DataDirect Networks, Inc.
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

#ifndef __INFINIA_CLIENT_H
#define __INFINIA_CLIENT_H

#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <list>
#include <future>
#include <memory>
#include <set>

#include "nixl_types.h"
#include "common/nixl_log.h"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <red/red_async.hpp>

class InfiniaClient {
public:
    InfiniaClient(const std::string& cluster_name,
                  const std::string& tenant_name,
                  const std::string& subtenant_name,
                  const std::string& bucket_name,
                  const uint32_t sthreads,
                  const uint32_t num_buffers,
                  const uint32_t num_ring_entries,
                  const std::string& coremasks);

    ~InfiniaClient();

    red_status_t initialize();
    void cleanup();

    bool isInitialized() const {
        return initialized_;
    }

    // Get red_config_t for accessing resources
    red_async::red_config_t* get_config() {
        return config.get();
    }

    private:
    // Client configuration parameters
    std::string cluster_name_;
    std::string tenant_name_;
    std::string subtenant_name_;
    std::string bucket_name_;
    uint32_t sthreads_;
    uint32_t num_buffers_;
    uint32_t num_ring_entries_;
    std::string coremasks_;

    // RED async API objects
    std::unique_ptr<red_async::red_config_t> config;

    bool initialized_;
};

#endif
