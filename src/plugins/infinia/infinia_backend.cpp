/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) DataDirect Networks, Inc.
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

#include "infinia_backend.h"
#include <iostream>
#include <string.h>
#include <unordered_set>
#include <unordered_map>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include "common/nixl_log.h"
#include "common/configuration.h"
#include <thread>
#include <fstream>
#include <sstream>
#include <map>
#include <filesystem>
#include <string_view>
#include <ranges>
#include <algorithm>
#include <cctype>
#include <unistd.h> // for close()

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace {
// Helper function to convert memory type to string
const char *
memTypeToStr(nixl_mem_t mem) {
    switch (mem) {
    case DRAM_SEG:
        return "DRAM_SEG";
    case VRAM_SEG:
        return "VRAM_SEG";
    case BLK_SEG:
        return "BLK_SEG";
    case OBJ_SEG:
        return "OBJ_SEG";
    case FILE_SEG:
        return "FILE_SEG";
    default:
        return "UNKNOWN_SEG";
    }
}

// Helper function to cast generic handle to Infinia-specific handle
nixlInfiniaBackendReqH &
castInfiniaHandle(nixlBackendReqH *handle) {
    if (!handle) {
        throw std::invalid_argument("Received null handle");
    }
    return dynamic_cast<nixlInfiniaBackendReqH &>(*handle);
}

// Parse "TENANT/SUBTENANT" into two C-string pointers using a backing storage
static bool
splitTenantSubtenant(const char *s,
                     const char *&tenant_ptr,
                     const char *&subtenant_ptr,
                     std::string &backing_storage) {
    if (!s) {
        return false;
    }
    const char *slash = std::strchr(s, '/');
    if (!slash || slash == s || *(slash + 1) == '\0') {
        return false; // require non-empty tenant and subtenant
    }
    backing_storage.assign(s);
    size_t idx = backing_storage.find('/');
    backing_storage[idx] = '\0';
    tenant_ptr = backing_storage.c_str();
    subtenant_ptr = backing_storage.c_str() + idx + 1;
    return true;
}

// Helper for simple string-valued environment or config-file overrides
void
applyStringOverrideFromConfig(const char *key, std::string &target) {
    if (auto cfg = nixl::config::getValueOptional<std::string>(key)) {
        if (!cfg->empty()) {
            target = *cfg;
        }
    }
}

// Helper for TOML overrides of numeric values, applied only when at default
template<typename ConfigType, typename ValueType>
void
applyConfigOverride(const char *key, const ValueType default_value, ValueType &target) {
    if (target == default_value) {
        if (auto v = nixl::config::getValueOptional<ConfigType>(key)) {
            target = static_cast<ValueType>(*v);
        }
    }
}

// Helper for string TOML overrides that also track an explicit "set" flag
void
applyStringTomlOverrideIfNotSet(const char *key, std::string &target, bool &was_set_flag) {
    if (!was_set_flag) {
        if (auto v = nixl::config::getValueOptional<std::string>(key)) {
            target = *v;
            was_set_flag = true;
        }
    }
}

// Helper for boolean TOML overrides that also track an explicit "set" flag
void
applyBoolTomlOverrideIfNotSet(const char *key, bool &target, bool &was_set_flag) {
    if (!was_set_flag) {
        if (auto v = nixl::config::getValueOptional<bool>(key)) {
            target = *v;
            was_set_flag = true;
        }
    }
}
} // namespace

// -----------------------------------------------------------------------------
// Infinia Engine Implementation
// -----------------------------------------------------------------------------

infinia_engine::infinia_engine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      infinia_cluster_(INFINIA_DEFAULT_CLUSTER),
      infinia_tenant_(INFINIA_DEFAULT_TENANT),
      infinia_subtenant_(INFINIA_DEFAULT_SUBTENANT),
      infinia_dataset_(INFINIA_DEFAULT_DATASET),
      infinia_sthreads_(INFINIA_DEFAULT_STHREADS),
      infinia_num_buffers_(INFINIA_DEFAULT_BUFFERS),
      infinia_num_ring_entries_(INFINIA_DEFAULT_RING_ENTRIES),
      infinia_coremasks_(INFINIA_DEFAULT_COREMASK),
      infinia_coremasks_set_(false),
      use_dmabuf_(true), // Enable by default
      use_dmabuf_set_(false),
      initialized_(false) {

    red_status_t rs;

    NIXL_INFO << "INFINIA: v" << INFINIA_PLUGIN_VERSION;

    if (!init_params) {
        initErr = true;
        NIXL_ERROR << "Infinia backend: Invalid initialization parameters";
        return;
    }

    // Initialize batch task configuration with defaults
    batch_config_.max_retries = red_async::RED_ASYNC_DEFAULT_MAX_RETRIES;
    batch_config_.batch_size = red_async::RED_ASYNC_DEFAULT_BATCH_SIZE;

    // Extract Infinia-specific configuration from custom parameters
    if (init_params->customParams) {
        auto params = init_params->customParams;

        // Look for Infinia cluster configuration
        auto cluster_it = params->find("cluster");
        if (cluster_it != params->end()) {
            infinia_cluster_ = cluster_it->second;
        }

        // Look for Infinia tenant configuration
        auto tenant_it = params->find("tenant");
        if (tenant_it != params->end()) {
            infinia_tenant_ = tenant_it->second;
        }

        // Look for Infinia subtenant configuration
        auto subtenant_it = params->find("subtenant");
        if (subtenant_it != params->end()) {
            infinia_subtenant_ = subtenant_it->second;
        }

        // Look for Infinia dataset configuration
        auto dataset_it = params->find("dataset");
        if (dataset_it != params->end()) {
            infinia_dataset_ = dataset_it->second;
        }

        // Look for Infinia sthreads configuration
        auto sthreads_it = params->find("sthreads");
        if (sthreads_it != params->end() && !sthreads_it->second.empty()) {
            try {
                infinia_sthreads_ = std::stoul(sthreads_it->second);
            }
            catch (const std::exception &e) {
                NIXL_WARN << absl::StrFormat("Invalid sthreads value '%s', using default %u",
                                             sthreads_it->second.c_str(),
                                             INFINIA_DEFAULT_STHREADS);
            }
        }

        // Look for Infinia num_buffers configuration
        auto num_buffers_it = params->find("num_buffers");
        if (num_buffers_it != params->end() && !num_buffers_it->second.empty()) {
            try {
                infinia_num_buffers_ = std::stoul(num_buffers_it->second);
            }
            catch (const std::exception &e) {
                NIXL_WARN << absl::StrFormat("Invalid num_buffers value '%s', using default %u",
                                             num_buffers_it->second.c_str(),
                                             INFINIA_DEFAULT_BUFFERS);
            }
        }

        // Look for Infinia num_ring_entries configuration
        auto num_ring_entries_it = params->find("num_ring_entries");
        if (num_ring_entries_it != params->end() && !num_ring_entries_it->second.empty()) {
            try {
                infinia_num_ring_entries_ = std::stoul(num_ring_entries_it->second);
            }
            catch (const std::exception &e) {
                NIXL_WARN << absl::StrFormat(
                    "Invalid num_ring_entries value '%s', using default %u",
                    num_ring_entries_it->second.c_str(),
                    INFINIA_DEFAULT_RING_ENTRIES);
            }
        }

        // Look for Infinia coremasks configuration
        auto coremasks_it = params->find("coremasks");
        if (coremasks_it != params->end()) {
            infinia_coremasks_ = coremasks_it->second;
            infinia_coremasks_set_ = true;
        }

        // Look for DMA-BUF enable/disable configuration
        auto use_dmabuf_it = params->find("use_dmabuf");
        if (use_dmabuf_it != params->end() && !use_dmabuf_it->second.empty()) {
            std::string val = use_dmabuf_it->second;
            // Convert to lowercase for comparison
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            use_dmabuf_ = (val == "true");
            use_dmabuf_set_ = true;
            NIXL_DEBUG << absl::StrFormat("DMA-BUF support %s",
                                          use_dmabuf_ ? "enabled" : "disabled");
        }

        // Look for Infinia max_retries configuration
        auto max_retries_it = params->find("max_retries");
        if (max_retries_it != params->end() && !max_retries_it->second.empty()) {
            try {
                batch_config_.max_retries = std::stoull(max_retries_it->second);
            }
            catch (const std::exception &) {
                NIXL_WARN << absl::StrFormat("Invalid max_retries value '%s', using default %zu",
                                             max_retries_it->second.c_str(),
                                             red_async::RED_ASYNC_DEFAULT_MAX_RETRIES);
            }
        }

        // Look for Infinia batch_size configuration
        auto batch_size_it = params->find("batch_size");
        if (batch_size_it != params->end() && !batch_size_it->second.empty()) {
            try {
                batch_config_.batch_size = std::stoull(batch_size_it->second);
            }
            catch (const std::exception &) {
                NIXL_WARN << absl::StrFormat("Invalid batch_size value '%s', using default %zu",
                                             batch_size_it->second.c_str(),
                                             red_async::RED_ASYNC_DEFAULT_BATCH_SIZE);
            }
        }
    }

    // Environment or config-file override for cluster
    applyStringOverrideFromConfig(RED_CLUSTER_ENV, infinia_cluster_);

    // Environment or config-file override for tenant/subtenant
    if (auto tenant_cfg = nixl::config::getValueOptional<std::string>(RED_TENANT_ENV)) {
        if (!tenant_cfg->empty()) {
            const char *tenant_ptr = nullptr;
            const char *subtenant_ptr = nullptr;
            std::string tenant_buf; // holds split strings' storage
            if (splitTenantSubtenant(tenant_cfg->c_str(), tenant_ptr, subtenant_ptr, tenant_buf)) {
                infinia_tenant_ = tenant_ptr;
                infinia_subtenant_ = subtenant_ptr;
            } else {
                // Only tenant provided; keep subtenant as configured earlier
                infinia_tenant_ = *tenant_cfg;
            }
        }
    }

    // Environment or config-file override for dataset
    applyStringOverrideFromConfig(RED_DATASET_ENV, infinia_dataset_);

    // TOML [infinia] overrides for tuning parameters if they remain at defaults
    applyConfigOverride<uint32_t, uint32_t>(
        "infinia.sthreads", INFINIA_DEFAULT_STHREADS, infinia_sthreads_);

    applyConfigOverride<uint32_t, uint32_t>(
        "infinia.num_buffers", INFINIA_DEFAULT_BUFFERS, infinia_num_buffers_);

    applyConfigOverride<uint32_t, uint32_t>(
        "infinia.num_ring_entries", INFINIA_DEFAULT_RING_ENTRIES, infinia_num_ring_entries_);

    applyStringTomlOverrideIfNotSet(
        "infinia.coremasks", infinia_coremasks_, infinia_coremasks_set_);

    applyConfigOverride<uint32_t, decltype(batch_config_.max_retries)>(
        "infinia.max_retries", red_async::RED_ASYNC_DEFAULT_MAX_RETRIES, batch_config_.max_retries);

    applyConfigOverride<uint32_t, decltype(batch_config_.batch_size)>(
        "infinia.batch_size", red_async::RED_ASYNC_DEFAULT_BATCH_SIZE, batch_config_.batch_size);

    applyBoolTomlOverrideIfNotSet("infinia.use_dmabuf", use_dmabuf_, use_dmabuf_set_);

    // Set default params if not provided
    if (infinia_cluster_.empty()) {
        infinia_cluster_ = INFINIA_DEFAULT_CLUSTER;
    }

    if (infinia_tenant_.empty()) {
        infinia_tenant_ = INFINIA_DEFAULT_TENANT;
    }

    if (infinia_subtenant_.empty()) {
        infinia_subtenant_ = INFINIA_DEFAULT_SUBTENANT;
    }

    if (infinia_dataset_.empty()) {
        infinia_dataset_ = INFINIA_DEFAULT_DATASET;
    }

    // Only apply default coremask if it was never explicitly set
    // This allows users to explicitly set an empty string to disable coremask
    if (!infinia_coremasks_set_ && infinia_coremasks_.empty()) {
        infinia_coremasks_ = INFINIA_DEFAULT_COREMASK;
    }

    // Debug log: dump final resolved configuration after params/env/TOML/defaults
    NIXL_DEBUG << absl::StrFormat(
        "INFINIA effective config: cluster=%s tenant=%s subtenant=%s dataset=%s "
        "sthreads=%u num_buffers=%u num_ring_entries=%u coremasks=%s use_dmabuf=%s "
        "max_retries=%zu batch_size=%zu",
        infinia_cluster_.c_str(),
        infinia_tenant_.c_str(),
        infinia_subtenant_.c_str(),
        infinia_dataset_.c_str(),
        infinia_sthreads_,
        infinia_num_buffers_,
        infinia_num_ring_entries_,
        infinia_coremasks_.c_str(),
        use_dmabuf_ ? "true" : "false",
        batch_config_.max_retries,
        batch_config_.batch_size);

    client_ = std::make_shared<InfiniaClient>(infinia_cluster_,
                                              infinia_tenant_,
                                              infinia_subtenant_,
                                              infinia_dataset_,
                                              infinia_sthreads_,
                                              infinia_num_buffers_,
                                              infinia_num_ring_entries_,
                                              infinia_coremasks_);

    rs = client_->initialize();
    if (rs != RED_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Failed to initialize Infinia client: %d", rs);
        initErr = true;
        return;
    }

    NIXL_INFO << absl::StrFormat("Infinia engine initialized: max_retries=%zu, "
                                 "sthreads=%u, buffers=%u, entries=%u, coremask=%s",
                                 batch_config_.max_retries,
                                 infinia_sthreads_,
                                 infinia_num_buffers_,
                                 infinia_num_ring_entries_,
                                 infinia_coremasks_);

    initialized_ = true;
}

infinia_engine::~infinia_engine() {
    if (client_) {
        client_->cleanup();
    }

    NIXL_DEBUG << "Infinia backend destroyed";
}

#ifdef HAVE_CUDA
nixl_status_t
infinia_engine::registerGpuMemoryDmabuf(const nixlBlobDesc &mem, nixlInfiniaMetadata *metadata) {
    if (!use_dmabuf_) {
        // DMA-BUF disabled, fallback to traditional registration
        return NIXL_ERR_NOT_SUPPORTED;
    }

    CUdeviceptr dev_ptr = (CUdeviceptr)mem.addr;
    int dmabuf_fd = -1;
    void *buffer = reinterpret_cast<void *>(mem.addr);

    // Set CUDA device context (required for DMA-BUF operations)
    cudaError_t cuda_err = cudaSetDevice(mem.devId);
    if (cuda_err != cudaSuccess) {
        NIXL_WARN << absl::StrFormat("Failed to set CUDA device %d: %s, "
                                     "falling back to traditional registration",
                                     (int)mem.devId,
                                     cudaGetErrorString(cuda_err));
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // Check DMA-BUF support on this device
    int supported = 0;
    CUdevice cuda_dev;
    CUresult cu_res = cuDeviceGet(&cuda_dev, mem.devId);
    if (cu_res == CUDA_SUCCESS) {
        cu_res = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cuda_dev);
    }
    if (cu_res != CUDA_SUCCESS || !supported) {
        NIXL_DEBUG << "DMA-BUF not supported on this GPU, using traditional registration";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // Check page alignment requirements (DMA-BUF typically requires page alignment)
    static size_t host_page_size = sysconf(_SC_PAGESIZE);
    bool is_aligned = (mem.len % host_page_size) == 0 && ((mem.addr % host_page_size) == 0);

    if (!is_aligned) {
        NIXL_WARN << absl::StrFormat(
            "GPU memory not page-aligned (addr=0x%lx, size=%zu, page_size=%zu), "
            "falling back to traditional registration",
            mem.addr,
            mem.len,
            host_page_size);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // Export CUDA memory to DMA-BUF
    cu_res = cuMemGetHandleForAddressRange(&dmabuf_fd,
                                           dev_ptr,
                                           mem.len,
                                           CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                           0 // flags
    );

    if (cu_res != CUDA_SUCCESS || dmabuf_fd < 0) {
        NIXL_WARN << absl::StrFormat("Failed to export GPU memory to DMA-BUF (CUDA error=%d), "
                                     "falling back to traditional registration",
                                     (int)cu_res);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    NIXL_DEBUG << absl::StrFormat("Exported GPU memory to DMA-BUF: fd=%d, addr=0x%lx, size=%zu",
                                  dmabuf_fd,
                                  mem.addr,
                                  mem.len);

    // Register using DMA-BUF API
    red_status_t rs =
        red_async::red_config_t::register_user_dmabuf(dmabuf_fd, // DMA-BUF file descriptor
                                                      0, // offset
                                                      buffer, // IOVA (GPU virtual address)
                                                      mem.len, // length
                                                      &metadata->iomem_handle // output handle
        );

    if (rs == RED_SUCCESS) {
        metadata->dmabuf_fd = dmabuf_fd;
        NIXL_DEBUG << absl::StrFormat("Successfully registered GPU memory via DMA-BUF "
                                      "(fd=%d devId=0x%lx addr=%p len=%zu key=\'%s\'",
                                      dmabuf_fd,
                                      metadata->devId,
                                      metadata->buffer,
                                      metadata->length,
                                      metadata->objKey.c_str());
        return NIXL_SUCCESS;
    } else {
        NIXL_ERROR << absl::StrFormat(
            "Failed to register DMA-BUF memory (fd=%d, addr=%p, size=%zu): status=%d",
            dmabuf_fd,
            buffer,
            mem.len,
            static_cast<int>(rs));
        close(dmabuf_fd);
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
infinia_engine::unregisterDmabuf(nixlInfiniaMetadata *metadata) {
    if (metadata->dmabuf_fd < 0) {
        // Not registered via DMA-BUF
        return NIXL_ERR_INVALID_PARAM;
    }

    red_status_t rs = red_async::red_config_t::unregister_user_iomem(metadata->iomem_handle);
    if (rs != RED_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Failed to unregister DMA-BUF memory (fd=%d): status=%d",
                                      metadata->dmabuf_fd,
                                      static_cast<int>(rs));
        // Continue with cleanup even if unregistration fails
    } else {
        NIXL_DEBUG << absl::StrFormat("Successfully unregistered DMA-BUF memory (fd=%d)",
                                      metadata->dmabuf_fd);
    }

    // Close the DMA-BUF file descriptor
    if (close(metadata->dmabuf_fd) != 0) {
        NIXL_WARN << absl::StrFormat(
            "Failed to close DMA-BUF fd=%d: %s", metadata->dmabuf_fd, strerror(errno));
    }
    metadata->dmabuf_fd = -1;

    return NIXL_SUCCESS;
}
#endif

bool
infinia_engine::validateTransferParams(const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent) const {
    // Validate operation type
    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        NIXL_ERROR << absl::StrFormat("Unsupported operation type: %d", operation);
        return false;
    }

    // Validate descriptor counts match
    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat("Descriptor count mismatch - local: %zu, remote: %zu",
                                      local.descCount(),
                                      remote.descCount());
        return false;
    }

    if (local.getType() != DRAM_SEG && local.getType() != VRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Unsupported local memory type: %d", local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Unsupported remote memory type: %d", remote.getType());
        return false;
    }

    return true;
}

nixl_status_t
infinia_engine::registerMem(const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {

    NIXL_DEBUG << absl::StrFormat(
        "INFINIA: REGISTER mem=%s addr=0x%lx len=%zu (%.2f MiB) devId=0x%lx meta='%s'",
        memTypeToStr(nixl_mem),
        mem.addr,
        mem.len,
        mem.len / (1024.0 * 1024.0),
        mem.devId,
        mem.metaInfo.c_str());

    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    // Check if memory type is supported
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        // --- OBJ_SEG: Extract key and store mapping ---
        std::string obj_key;
        if (!mem.metaInfo.empty()) {
            obj_key = mem.metaInfo;
        } else if (mem.devId != 0) {
            obj_key = std::to_string(mem.devId);
        } else {
            NIXL_ERROR << "OBJ_SEG registration with empty metaInfo AND devId==0";
            return NIXL_ERR_INVALID_PARAM;
        }

        // Store devId -> key mapping for later lookup in prepXfer
        devid_to_key_map_[mem.devId] = obj_key;

        auto metadata = std::make_unique<nixlInfiniaMetadata>(nixl_mem, mem.devId, obj_key);
        out = metadata.release();

    } else {
        // --- DRAM_SEG / VRAM_SEG: Register memory with RED ---
        auto metadata = std::make_unique<nixlInfiniaMetadata>(nixl_mem, mem.devId, "");
        metadata->buffer = reinterpret_cast<void *>(mem.addr);
        metadata->length = mem.len;

        // Only register if not a probe call (addr/len are valid)
        if (mem.addr != 0 && mem.len != 0) {
            void *buffer = reinterpret_cast<void *>(mem.addr);
            red_status_t rs;

#ifdef HAVE_CUDA
            // Try DMA-BUF registration for GPU memory if enabled
            if (nixl_mem == VRAM_SEG) {
                nixl_status_t dmabuf_status = registerGpuMemoryDmabuf(mem, metadata.get());

                if (dmabuf_status == NIXL_SUCCESS) {
                    // Successfully registered via DMA-BUF
                    out = metadata.release();
                    return NIXL_SUCCESS;
                }
                // Fall through to traditional registration if DMA-BUF failed or not supported
            }
#endif
            // Traditional registration for DRAM or non-DMA-BUF GPU memory
            red_memory_types_e mem_type =
                (nixl_mem == VRAM_SEG) ? RED_MEMORY_TYPE_GPU : RED_MEMORY_TYPE_CPU;

            rs = red_async::red_config_t::register_user_memory(
                buffer, mem.len, &metadata->iomem_handle, mem_type);

            if (rs != RED_SUCCESS) {
                NIXL_ERROR << absl::StrFormat("Failed to register memory (%p, size=%zu): status=%d",
                                              buffer,
                                              mem.len,
                                              static_cast<int>(rs));
                return NIXL_ERR_BACKEND;
            }
        }

        out = metadata.release();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::deregisterMem(nixlBackendMD *meta) {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    if (meta == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Cast to Infinia metadata and remove from map. Use unique_ptr to ensure cleanup even if an
    // exception occurs
    std::unique_ptr<nixlInfiniaMetadata> infinia_meta(static_cast<nixlInfiniaMetadata *>(meta));

    NIXL_DEBUG << absl::StrFormat(
        "INFINIA: DEREGISTER mem=%s addr=%p len=%zu (%.2f MiB) devId=0x%lx key='%s'",
        memTypeToStr(infinia_meta->nixlMem),
        infinia_meta->buffer,
        infinia_meta->length,
        infinia_meta->length / (1024.0 * 1024.0),
        infinia_meta->devId,
        infinia_meta->objKey.c_str());

    // Remove from mapping (thread-safe)
    devid_to_key_map_.erase(infinia_meta->devId);

    if ((infinia_meta->nixlMem == VRAM_SEG || infinia_meta->nixlMem == DRAM_SEG) &&
        infinia_meta->buffer != nullptr) {

#ifdef HAVE_CUDA
        // If registered via DMA-BUF, use helper function
        if (infinia_meta->dmabuf_fd >= 0) {
            (void)unregisterDmabuf(infinia_meta.get());
        } else
#endif
        {
            // Traditional unregistration for DRAM or non-DMA-BUF GPU memory
            red_status_t rs = red_async::red_config_t::unregister_user_memory(infinia_meta->buffer);
            if (rs != RED_SUCCESS) {
                NIXL_ERROR << absl::StrFormat("Failed to unregister memory (%p): status=%d",
                                              infinia_meta->buffer,
                                              static_cast<int>(rs));
                // Continue with cleanup even if unregistration fails
            } else {
                NIXL_DEBUG << absl::StrFormat("Successfully unregistered memory (%p)",
                                              infinia_meta->buffer);
            }
        }
    }

    // Smart pointer automatically deletes the object.
    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::queryMem(const nixl_reg_dlist_t &descs,
                         std::vector<nixl_query_resp_t> &resp) const {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << absl::StrFormat(
        "INFINIA: QUERYMEM mem=%s count=%d", memTypeToStr(descs.getType()), descs.descCount());

    // Pre-allocate response vector with nullopt for all descriptors
    // This handles skipped descriptors and maintains proper ordering
    resp.assign(descs.descCount(), std::nullopt);

    // Create a BatchTask for this queryMem call and reserve capacity
    red_async::BatchTask batch_task(client_->getConfig(), &batch_config_);
    batch_task.reserve(descs.descCount());

    // Storage for keys and stat buffers (must persist until batch completes)
    std::vector<std::string> keys;
    keys.reserve(descs.descCount());
    std::vector<struct stat> stat_buffers(descs.descCount());

    // Track mapping from batch operation index to descriptor index
    // Needed because skipped descriptors create gaps in operation indices
    std::vector<size_t> op_to_desc_idx;
    op_to_desc_idx.reserve(descs.descCount());

    // Add all HEAD operations to the batch
    for (int i = 0; i < descs.descCount(); ++i) {
        const auto &desc = descs[i];

        // Mirror registerMem's key selection logic
        if (!desc.metaInfo.empty()) {
            keys.push_back(desc.metaInfo);
        } else if (desc.devId != 0) {
            keys.push_back(std::to_string(desc.devId));
        } else {
            NIXL_WARN << "Skipping OBJ query with empty metaInfo and devId==0";
            // resp[i] already initialized to nullopt, just skip adding operation
            continue;
        }

        // Record mapping from batch operation index to descriptor index
        op_to_desc_idx.push_back(i);

        // Build HEAD operation with stat buffer
        red_async::red_batch_operation_t op;
        op.operation_type = red_async::RED_ASYNC_OP_CACHE_HEAD;
        op.key = keys.back().c_str();
        op.key_len = static_cast<uint32_t>(keys.back().length());
        op.offset = 0;
        op.sg_list = nullptr; // HEAD doesn't need data transfer
        op.kv_flag = 0;
        op.di = nullptr;
        op.version_out = nullptr;

        // Provide stat buffer for HEAD operation
        op.op_specific.statbuf = &stat_buffers[i];

        batch_task.add_operation(std::move(op));
    }

    // Execute the batch ops in parallel
    red_status_t rs = batch_task.start();
    if (rs != RED_SUCCESS) {
        // Batch start failed (possibly due to Infinia node failure/SCR eviction)
        // Treat all keys as "not found" to maintain inference availability
        // This allows READ path to fall back to recomputation and WRITE path to attempt caching
        NIXL_WARN << absl::StrFormat(
            "Failed to start HEAD batch (status=%d) - treating all keys as not found. "
            "Possible Infinia node failure. Inference continues with cache misses.",
            static_cast<int>(rs));
        // resp already initialized to all std::nullopt (line 699)
        return NIXL_SUCCESS; // Return success to avoid LMCache crash
    }

    // Wait for completion (blocks until all operations complete)
    try {
        batch_task.wait();
    }
    catch (const std::exception &e) {
        // Exception during wait (possibly due to node failure)
        // Treat all keys as not found to maintain inference availability
        NIXL_WARN << absl::StrFormat(
            "Exception during HEAD batch wait: %s - treating all keys as not found. "
            "Inference continues with cache misses.",
            e.what());
        // resp already initialized to all std::nullopt
        return NIXL_SUCCESS; // Return success to avoid crash
    }

    // Get results (wait() guarantees batch is ready)
    red_async::rae_batch_task_result_t result;
    try {
        result = batch_task.get_result();
    }
    catch (const std::exception &e) {
        // Exception getting results (possibly due to node failure)
        NIXL_WARN << absl::StrFormat(
            "Exception getting HEAD batch results: %s - treating all keys as not found. "
            "Inference continues with cache misses.",
            e.what());
        // resp already initialized to all std::nullopt
        return NIXL_SUCCESS; // Return success to avoid crash
    }

    // Track failed operations for logging
    size_t failed_operations = 0;

    // Process results using index mapping to maintain descriptor order
    for (size_t op_idx = 0; op_idx < result.operation_results.size(); ++op_idx) {
        const auto &op_result = result.operation_results[op_idx];
        const size_t desc_idx = op_to_desc_idx[op_idx];

        if (op_result.status == RED_SUCCESS) {
            // Key exists
            resp[desc_idx] = nixl_query_resp_t{nixl_b_params_t{}};
            NIXL_DEBUG << absl::StrFormat("INFINIA: QUERYMEM key='%s' found=true",
                                          keys[op_idx].c_str());
        } else if (op_result.status == RED_ENOENT) {
            // Key does not exist (normal cache miss)
            resp[desc_idx] = std::nullopt;
            NIXL_DEBUG << absl::StrFormat("INFINIA: QUERYMEM key='%s' found=false",
                                          keys[op_idx].c_str());
        } else {
            // Other error - treat as key not found to maintain inference availability
            // This handles node failures, SCR evictions, and network issues gracefully
            NIXL_WARN << absl::StrFormat(
                "HEAD operation for key '%s' failed (status=%d) - treating as not found. "
                "Possible Infinia node failure.",
                keys[op_idx].c_str(),
                static_cast<int>(op_result.status));
            resp[desc_idx] = std::nullopt;
            failed_operations++;
        }
    }

    // Log summary if there were failures
    if (failed_operations > 0) {
        NIXL_WARN << absl::StrFormat(
            "INFINIA QUERYMEM: %zu/%zu HEAD operations failed - all treated as cache miss. "
            "Inference continues but may experience performance degradation.",
            failed_operations,
            result.operation_results.size());
    }

    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::connect(const std::string &remote_agent) {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    // TODO: Add actual Infinia connection logic here
    NIXL_DEBUG << "Connecting to remote agent: " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::disconnect(const std::string &remote_agent) {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    // TODO: Add actual Infinia disconnection logic here
    NIXL_DEBUG << "Disconnecting from remote agent: " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::unloadMD(nixlBackendMD *input) {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    // TODO: Add actual metadata cleanup here
    NIXL_DEBUG << "Unloading metadata";
    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::prepXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {

    NIXL_DEBUG << absl::StrFormat(
        "INFINIA: PREPXFER op=%s, local: mem=%s count=%d remote: mem=%s count=%d agent='%s'",
        operation == NIXL_READ ? "READ" : "WRITE",
        memTypeToStr(local.getType()),
        local.descCount(),
        memTypeToStr(remote.getType()),
        remote.descCount(),
        remote_agent.c_str());

    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    if (!validateTransferParams(operation, local, remote, remote_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    try {
        auto backend_handle = std::make_unique<nixlInfiniaBackendReqH>(
            operation, local, remote, remote_agent, opt_args, client_, batch_config_);

        // Reserve capacity to prevent vector reallocation
        backend_handle->reserveOperations(local.descCount());

        // Collect all requests (some may need splits)
        for (auto [local_it, remote_it] = std::make_pair(local.begin(), remote.begin());
             local_it != local.end() && remote_it != remote.end();
             ++local_it, ++remote_it) {
            // Look up the key from the devId mapping (thread-safe)
            std::string key;
            {
                auto key_it = devid_to_key_map_.find(remote_it->devId);
                if (key_it == devid_to_key_map_.end()) {
                    NIXL_ERROR << absl::StrFormat("INFINIA: KEY FAILURE for devId=0x%lx",
                                                  remote_it->devId);
                    NIXL_ERROR << "Memory may not be registered, or devId mismatch between "
                                  "registration and transfer!";
                    NIXL_ERROR << "========================================";
                    NIXL_DEBUG << "  devid_to_key_map_ contains " << devid_to_key_map_.size()
                               << " entries";
                    if (devid_to_key_map_.size() <= 10) {
                        NIXL_DEBUG << "  --- Current devid_to_key_map_ contents ---";
                        for (const auto &[dev_id, key_str] : devid_to_key_map_) {
                            NIXL_DEBUG << "    devId 0x" << std::hex << dev_id << std::dec
                                       << " -> \"" << key_str << "\"";
                        }
                    }

                    return NIXL_ERR_INVALID_PARAM;
                }
                key = key_it->second; // Copy the key while holding the lock
            }

            void *val = reinterpret_cast<void *>(local_it->addr);

            // Get pre-registered memory handle from local metadata
            // The local descriptor should have metadata pointer set by the agent
            nixlInfiniaMetadata *local_metadata =
                static_cast<nixlInfiniaMetadata *>(local_it->metadataP);
            if (local_metadata == nullptr) {
                NIXL_ERROR << absl::StrFormat("No metadata found for local memory (addr=0x%lx). "
                                              "Memory may not be registered.",
                                              local_it->addr);
                return NIXL_ERR_INVALID_PARAM;
            }

            // Verify that the entire transfer buffer is within the registered memory region
            uintptr_t transfer_start = reinterpret_cast<uintptr_t>(val);
            uintptr_t transfer_end = transfer_start + local_it->len;
            uintptr_t registered_start = reinterpret_cast<uintptr_t>(local_metadata->buffer);
            uintptr_t registered_end = registered_start + local_metadata->length;

            if (transfer_start < registered_start || transfer_end > registered_end) {
                NIXL_ERROR << absl::StrFormat("Memory range validation failed:\n"
                                              "Transfer:   [%p, %p) size=%zu\n"
                                              "Registered: [%p, %p) size=%zu\n"
                                              "Offset from registered base: %zd bytes",
                                              val,
                                              reinterpret_cast<void *>(transfer_end),
                                              local_it->len,
                                              local_metadata->buffer,
                                              reinterpret_cast<void *>(registered_end),
                                              local_metadata->length,
                                              (ssize_t)(transfer_start - registered_start));
                return NIXL_ERR_INVALID_PARAM;
            }

            // Debug log for successful validation (only in debug builds)
            NIXL_DEBUG << absl::StrFormat(
                "INFINIA: MEMORY RANGE OK for transfer [%p+%zu] within registered [%p+%zu]",
                val,
                local_it->len,
                local_metadata->buffer,
                local_metadata->length);

            // Use pre-registered memory handle
            red_iomem_hndl_t iomem_handle = local_metadata->iomem_handle;

            // Determine operation type
            red_async::red_async_op_type_t op_type = (operation == NIXL_READ) ?
                red_async::RED_ASYNC_OP_CACHE_GET :
                red_async::RED_ASYNC_OP_CACHE_PUT;

            // Add operation directly to the batch task
            backend_handle->addOperation(
                op_type, key, val, local_it->len, iomem_handle, local_metadata->nixlMem);

            NIXL_DEBUG << absl::StrFormat(
                "INFINIA: ADDED OPERATION op=%s key='%s' addr=%p size=%zu mem_type=%s",
                op_type == red_async::RED_ASYNC_OP_CACHE_GET ? "GET" : "PUT",
                key.c_str(),
                val,
                local_it->len,
                memTypeToStr(local_metadata->nixlMem));
        }

        nixl_status_t status = backend_handle->prepareTransfer();
        if (status != NIXL_SUCCESS) {
            // Note: No cleanup needed here - memory will be unregistered later in deregisterMem()
            return status;
        }

        handle = backend_handle.release();

        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error preparing transfer: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
infinia_engine::postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    if (handle == nullptr) {
        NIXL_ERROR << "postXfer: Received null handle";
        return NIXL_ERR_INVALID_PARAM;
    }

    try {
        auto &backend_handle = castInfiniaHandle(handle);

        NIXL_DEBUG << absl::StrFormat("INFINIA: POSTXFER op=%s local: mem=%s count=%d remote: "
                                      "mem=%s count=%d agent='%s' operations=%zu",
                                      operation == NIXL_READ ? "READ" : "WRITE",
                                      memTypeToStr(local.getType()),
                                      local.descCount(),
                                      memTypeToStr(remote.getType()),
                                      remote.descCount(),
                                      remote_agent.c_str(),
                                      backend_handle.getOperationCount());

        // Launch the coroutine task in a background thread
        // The task will execute all batch operations asynchronously
        return backend_handle.postTransfer();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error posting transfer: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
infinia_engine::checkXfer(nixlBackendReqH *handle) const {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    try {
        auto &backend_handle = castInfiniaHandle(handle);

        // Check the transfer status
        return backend_handle.checkTransfer();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error checking transfer: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
infinia_engine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Futures are automatically released when handle is deleted
    delete handle;
    return NIXL_SUCCESS;
}

// Remote operation methods
nixl_status_t
infinia_engine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    // TODO: Serialize metadata for remote access
    str = "infinia_metadata_placeholder";
    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::getConnInfo(std::string &str) const {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::loadRemoteConnInfo(const std::string &remote_agent,
                                   const std::string &remote_conn_info) {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    // TODO: Parse and store remote connection information
    NIXL_DEBUG << "Ignoring remote connection info for " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
infinia_engine::loadRemoteMD(const nixlBlobDesc &input,
                             const nixl_mem_t &nixl_mem,
                             const std::string &remote_agent,
                             nixlBackendMD *&output) {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    // TODO: Create remote metadata object
    NIXL_DEBUG << "Ignoring remote metadata for " << remote_agent;
    output = nullptr;
    return NIXL_SUCCESS;
}

// Local operation methods
nixl_status_t
infinia_engine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    // For local operations, input and output metadata can be the same
    output = input;
    return NIXL_SUCCESS;
}

// -----------------------------------------------------------------------------
// Infinia Request Handle Implementation
// -----------------------------------------------------------------------------

nixlInfiniaBackendReqH::nixlInfiniaBackendReqH(const nixl_xfer_op_t &operation,
                                               const nixl_meta_dlist_t &local,
                                               const nixl_meta_dlist_t &remote,
                                               const std::string &remote_agent,
                                               const nixl_opt_b_args_t *opt_args,
                                               const std::shared_ptr<InfiniaClient> &client,
                                               const red_async::rae_batch_config_t &batch_config)
    : operation_(operation),
      local_(local),
      remote_(remote),
      remote_agent_(remote_agent),
      transfer_prepared_(false),
      transfer_posted_(false),
      transfer_completed_(false),
      transfer_status_(RED_SUCCESS),
      client_(client),
      batch_config_(batch_config),
      batch_task_(client->getConfig(), &batch_config) {

    NIXL_DEBUG << absl::StrFormat("Created Infinia request handle for %s operation",
                                  (operation_ == NIXL_READ) ? "READ" : "WRITE");
}

nixlInfiniaBackendReqH::~nixlInfiniaBackendReqH() {
    NIXL_DEBUG << "Destroying Infinia request handle";

    // Wait for batch execution to complete if still running
    if (batch_task_.is_started() && !batch_task_.is_ready()) {
        try {
            batch_task_.wait();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Exception while waiting for batch completion: " << e.what();
        }
    }
}

void
nixlInfiniaBackendReqH::reserveOperations(size_t count) {
    // Reserve capacity in all vectors to prevent reallocation during addOperation()
    // This is critical because we pass pointers to vector elements to batch_task_,
    // and those pointers must remain valid until the batch completes
    keys_.reserve(count);
    sg_elements_.reserve(count);
    sg_lists_.reserve(count);
    batch_task_.reserve(count);
}

void
nixlInfiniaBackendReqH::addOperation(red_async::red_async_op_type_t op_type,
                                     const std::string &key,
                                     void *value_addr,
                                     size_t value_size,
                                     red_iomem_hndl_t iomem_handle,
                                     nixl_mem_t mem_type) {
    // Store the key string (must persist until batch completes)
    keys_.push_back(key);

    // Store scatter-gather element and list (must persist until batch completes)
    red_sg_elem_t elem;
    elem.iomem = iomem_handle;
    elem.addr = value_addr;
    elem.offset = 0;
    elem.size = value_size;
    sg_elements_.push_back(elem);

    red_sg_list_t sg;
    sg.val_size = value_size;
    sg.num_elems = 1;
    sg.sg_elem = &sg_elements_.back(); // Point to the stored element
    sg.flags = RED_SGL_HOST;

    // Set flags based on memory type - enable GPU Direct for GPU memory
    if (mem_type == VRAM_SEG) {
        sg.flags = RED_SGL_GPU_DIRECT;
    }

    sg_lists_.push_back(sg);

    // Build the operation
    red_async::red_batch_operation_t op;
    op.operation_type = op_type;
    op.key = keys_.back().c_str(); // Point to the stored key
    op.key_len = static_cast<uint32_t>(keys_.back().length());
    op.offset = 0;
    op.sg_list = &sg_lists_.back(); // Point to the stored sg_list
    op.kv_flag = 0;
    op.di = nullptr;
    op.version_out = nullptr;

    batch_task_.add_operation(std::move(op));
}

nixl_status_t
nixlInfiniaBackendReqH::prepareTransfer() {
    if (transfer_prepared_) {
        NIXL_WARN << "Transfer already prepared";
        return NIXL_SUCCESS;
    }

    // All operations have already been added via addOperation()
    // Just mark as prepared
    transfer_prepared_ = true;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlInfiniaBackendReqH::postTransfer() {
    if (!transfer_prepared_) {
        NIXL_ERROR << "Transfer not prepared";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (transfer_posted_ && !transfer_completed_) {
        NIXL_WARN << "Transfer already posted and still in progress";
        return NIXL_IN_PROG;
    }

    // Check if this is a repost after completion (handle reuse)
    // Per BackendGuide: "transfer request will be prepped only once, but can be posted multiple
    // times"
    if (transfer_completed_) {
        NIXL_DEBUG << "Reposting completed transfer - batch_task_ was reset in checkTransfer()";
        transfer_completed_ = false;
        transfer_status_ = RED_SUCCESS;
    }

    NIXL_DEBUG << absl::StrFormat("Posting Infinia transfer with %zu operations", keys_.size());

    if (keys_.empty()) {
        transfer_posted_ = true;
        transfer_completed_ = true;
        transfer_status_ = RED_SUCCESS;
        return NIXL_SUCCESS;
    }

    // Start all operations using BatchTask API
    try {
        NIXL_DEBUG << absl::StrFormat("Starting batch with config: max_retries=%zu",
                                      batch_config_.max_retries);

        // Start all operations in parallel
        red_status_t rs = batch_task_.start();

        if (rs == RED_SUCCESS) {
            transfer_posted_ = true;
            return NIXL_IN_PROG;
        } else if (rs == RED_EAGAIN) {
            NIXL_ERROR << "Too many operations hit RED_EAGAIN during submission";
            // Handle submission failures based on operation type
            if (operation_ == NIXL_READ) {
                NIXL_WARN << "INFINIA READ submission failed (RED_EAGAIN) - treating as cache miss";
                return NIXL_ERR_NOT_FOUND;
            } else if (operation_ == NIXL_WRITE) {
                NIXL_WARN
                    << "INFINIA WRITE submission failed (RED_EAGAIN) - skipping cache population";
                return NIXL_SUCCESS; // Report success to avoid crash
            }
            return NIXL_ERR_BACKEND;
        } else if (rs == RED_EINVAL) {
            NIXL_ERROR << "BatchTask already started or invalid state";
            return NIXL_ERR_INVALID_PARAM;
        } else {
            NIXL_ERROR << "Failed to start batch: status=" << static_cast<int>(rs);
            // Handle startup failures based on operation type
            if (operation_ == NIXL_READ) {
                NIXL_WARN << absl::StrFormat(
                    "INFINIA READ batch start failed (status=%d) - treating as cache miss "
                    "to maintain inference availability",
                    static_cast<int>(rs));
                return NIXL_ERR_NOT_FOUND;
            } else if (operation_ == NIXL_WRITE) {
                NIXL_WARN << absl::StrFormat(
                    "INFINIA WRITE batch start failed (status=%d) - skipping cache population "
                    "to maintain inference availability",
                    static_cast<int>(rs));
                return NIXL_SUCCESS; // Report success to avoid crash
            }
            return NIXL_ERR_BACKEND;
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to start batch: " << e.what();
        // Handle exceptions based on operation type to maintain inference availability
        if (operation_ == NIXL_READ) {
            NIXL_WARN << "INFINIA READ batch start exception - treating as cache miss to maintain "
                         "inference availability";
            return NIXL_ERR_NOT_FOUND;
        } else if (operation_ == NIXL_WRITE) {
            NIXL_WARN << "INFINIA WRITE batch start exception - skipping cache population to "
                         "maintain inference availability";
            return NIXL_SUCCESS; // Report success to avoid crash
        }
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlInfiniaBackendReqH::checkTransfer() {
    if (!transfer_posted_) {
        NIXL_ERROR << "Transfer not posted";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check if batch execution is complete (non-blocking)
    if (!batch_task_.is_ready()) {
        return NIXL_IN_PROG;
    }

    // Get the result (will block if not ready, but we already checked is_ready())
    try {
        red_async::rae_batch_task_result_t result = batch_task_.get_result();

        size_t total_operations = result.operation_results.size();
        size_t failed_operations = result.failed_indices.size();
        size_t successful_operations = total_operations - failed_operations;

        NIXL_DEBUG << absl::StrFormat(
            "INFINIA: CHECKXFER rs=%d total=%zu success=%zu failed=%zu errors=%s",
            result.overall_status,
            total_operations,
            successful_operations,
            failed_operations,
            failed_operations > 0 ? "true" : "false");

        if (failed_operations > 0) {
            NIXL_WARN << absl::StrFormat("INFINIA: ERROR rs=%d total=%zu success=%zu failed=%zu",
                                         result.overall_status,
                                         total_operations,
                                         successful_operations,
                                         failed_operations);

            // Log failed operations using failed_indices
            if (!result.failed_indices.empty()) {
                const size_t max_detailed_failures = 100;
                size_t logged_failures = 0;

                for (size_t idx : result.failed_indices) {
                    if (logged_failures < max_detailed_failures &&
                        idx < result.operation_results.size()) {
                        const auto &op_result = result.operation_results[idx];
                        NIXL_WARN << absl::StrFormat(
                            "  Failed op[%zu]: key=\"%s\", operation=%s, status=%d",
                            idx,
                            op_result.key.c_str(),
                            op_result.operation_type == red_async::RED_ASYNC_OP_CACHE_GET ? "GET" :
                                                                                            "PUT",
                            op_result.status);
                        logged_failures++;
                    }
                }

                if (logged_failures < result.failed_indices.size()) {
                    NIXL_WARN << absl::StrFormat("  ... and %zu more failures (showing first %zu)",
                                                 result.failed_indices.size() - logged_failures,
                                                 max_detailed_failures);
                }
            }
        }

        // Store the final status
        transfer_status_ = result.overall_status;

        // Reset BatchTask state for potential reuse (allows reposting)
        red_status_t reset_rs = batch_task_.reset_state();
        if (reset_rs != RED_SUCCESS) {
            NIXL_WARN << "Failed to reset BatchTask state: status=" << static_cast<int>(reset_rs);
        }

        // Mark transfer as completed (allows reposting)
        transfer_posted_ = false;
        transfer_completed_ = true;

        // Handle failures based on operation type to maintain inference availability
        if (transfer_status_ != RED_SUCCESS) {
            if (operation_ == NIXL_READ) {
                // For READ operations that failed, treat as cache miss
                // When an Infinia node goes down (SCR eviction), READ failures should not crash
                // the inference engine. Instead, treat them as cache misses so LMCache falls back
                // to local computation. This allows inference to continue with degraded performance
                // (lower prefix cache hit rate) rather than complete failure.
                NIXL_WARN << absl::StrFormat(
                    "INFINIA READ failed (status=%d, failed_ops=%zu/%zu) - treating as cache miss "
                    "to maintain inference availability. Possible Infinia node failure/SCR "
                    "eviction.",
                    static_cast<int>(transfer_status_),
                    failed_operations,
                    total_operations);
                return NIXL_ERR_NOT_FOUND;
            } else if (operation_ == NIXL_WRITE) {
                // For WRITE operations that failed, skip cache population but don't crash
                // Cache writes are optimization (not critical path). If Infinia node is down,
                // inference can continue without caching - next request will just have cache miss.
                // This prevents inference worker crashes during node failures/SCR evictions.
                NIXL_WARN << absl::StrFormat("INFINIA WRITE failed (status=%d, failed_ops=%zu/%zu) "
                                             "- skipping cache population. "
                                             "Inference continues but cache hit rate may be "
                                             "affected. Possible Infinia node failure.",
                                             static_cast<int>(transfer_status_),
                                             failed_operations,
                                             total_operations);
                return NIXL_SUCCESS; // Report success to LMCache to avoid crash
            }
        }

        return transfer_status_ == RED_SUCCESS ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Exception while getting batch result: " << e.what();
        transfer_posted_ = false;
        transfer_completed_ = true;

        // Handle exceptions based on operation type to maintain inference availability
        if (operation_ == NIXL_READ) {
            NIXL_WARN << "INFINIA READ exception - treating as cache miss to maintain inference "
                         "availability";
            return NIXL_ERR_NOT_FOUND;
        } else if (operation_ == NIXL_WRITE) {
            NIXL_WARN << "INFINIA WRITE exception - skipping cache population to maintain "
                         "inference availability";
            return NIXL_SUCCESS; // Report success to avoid crash
        }

        return NIXL_ERR_BACKEND;
    }
}
