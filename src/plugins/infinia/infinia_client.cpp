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

#include <cassert>
#include <iostream>
#include <cstring>
#include <cstdlib>

#include "infinia_client.h"

InfiniaClient::InfiniaClient(const std::string &cluster_name,
                             const std::string &tenant_name,
                             const std::string &subtenant_name,
                             const std::string &bucket_name,
                             const uint32_t sthreads,
                             const uint32_t num_buffers,
                             const uint32_t num_ring_entries,
                             const std::string &coremasks)
    : cluster_name_(cluster_name),
      tenant_name_(tenant_name),
      subtenant_name_(subtenant_name),
      bucket_name_(bucket_name),
      sthreads_(sthreads),
      num_buffers_(num_buffers),
      num_ring_entries_(num_ring_entries),
      coremasks_(coremasks),
      initialized_(false) {}

InfiniaClient::~InfiniaClient() {
    if (initialized_) {
        cleanup();
    }
}

red_status_t
InfiniaClient::initialize() {
    if (initialized_) {
        return RED_SUCCESS;
    }

    NIXL_DEBUG << "Initializing InfiniaClient connection using red_config_t";

    // Create and configure red_config_t helper
    config_ = std::make_unique<red_async::red_config_t>();

    config_->set_cluster_name(cluster_name_);
    config_->set_tenant_name(tenant_name_);
    config_->set_subtenant_name(subtenant_name_);
    config_->set_dataset_base(bucket_name_);
    config_->set_client_sthreads(sthreads_);
    config_->set_num_buffers(num_buffers_);
    config_->set_num_ring_entries(num_ring_entries_);
    if (!coremasks_.empty()) {
        config_->set_coremask(coremasks_);
    }
    config_->set_poller_thread(true); // Required for async operations
    config_->set_num_contexts(1); // Single context for NIXL plugin
    config_->set_create_tenants(false); // Don't auto-create tenants but report error if not exists
    config_->set_create_dataset(false); // Don't auto-create dataset but report error if not exists
    config_->set_delete_dataset(false);
    config_->set_create_sys_user(false);

    // Initialize RED library, sessions, datasets, and roots
    red_status_t rs = red_async::red_config_t::initialize(config_.get());
    if (rs != RED_SUCCESS) {
        NIXL_ERROR << "red_config_t::initialize failed: " << red_strerror(rs);
        config_.reset();
        return rs;
    }

    initialized_ = true;

    NIXL_DEBUG << absl::StrFormat("InfiniaClient initialized with cluster=%s"
                                  ", tenant=%s"
                                  ", subtenant=%s"
                                  ", bucket=%s"
                                  ", sthreads=%u"
                                  ", num_buffers=%u"
                                  ", num_ring_entries=%u"
                                  ", coremasks=%s",
                                  cluster_name_.c_str(),
                                  tenant_name_.c_str(),
                                  subtenant_name_.c_str(),
                                  bucket_name_.c_str(),
                                  sthreads_,
                                  num_buffers_,
                                  num_ring_entries_,
                                  coremasks_.c_str());

    return RED_SUCCESS;
}

void
InfiniaClient::cleanup() {
    if (!initialized_) {
        return;
    }

    NIXL_DEBUG << "Cleaning up InfiniaClient connection";

    // Shutdown using red_config_t helper
    if (config_) {
        red_status_t rs = red_async::red_config_t::shutdown(config_.get());
        if (rs != RED_SUCCESS) {
            NIXL_ERROR << "red_config_t::shutdown failed: " << red_strerror(rs);
        }
        config_.reset();
    }

    initialized_ = false;
    NIXL_DEBUG << "InfiniaClient cleanup complete";
}
