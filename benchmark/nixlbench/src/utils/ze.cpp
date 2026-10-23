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

#include "ze.h"

#if HAVE_ZE

#include <level_zero/ze_api.h>

#include <dlfcn.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

/* The function pointer types are spelled out rather than recovered from &zeInit, because taking the
   address of a ze_api.h declaration would create exactly the link-time dependency on libze_loader
   that dlopen exists to avoid. */
using ze_init_fn_t = ze_result_t (*)(ze_init_flags_t);
using ze_driver_get_fn_t = ze_result_t (*)(uint32_t *, ze_driver_handle_t *);
using ze_device_get_fn_t = ze_result_t (*)(ze_driver_handle_t, uint32_t *, ze_device_handle_t *);
using ze_context_create_fn_t = ze_result_t (*)(ze_driver_handle_t,
                                               const ze_context_desc_t *,
                                               ze_context_handle_t *);
using ze_mem_alloc_device_fn_t = ze_result_t (*)(ze_context_handle_t,
                                                 const ze_device_mem_alloc_desc_t *,
                                                 size_t,
                                                 size_t,
                                                 ze_device_handle_t,
                                                 void **);
using ze_mem_free_fn_t = ze_result_t (*)(ze_context_handle_t, void *);
using ze_mem_get_alloc_properties_fn_t = ze_result_t (*)(ze_context_handle_t,
                                                         const void *,
                                                         ze_memory_allocation_properties_t *,
                                                         ze_device_handle_t *);
using ze_device_get_command_queue_group_properties_fn_t =
    ze_result_t (*)(ze_device_handle_t, uint32_t *, ze_command_queue_group_properties_t *);
using ze_command_list_create_immediate_fn_t = ze_result_t (*)(ze_context_handle_t,
                                                              ze_device_handle_t,
                                                              const ze_command_queue_desc_t *,
                                                              ze_command_list_handle_t *);
using ze_command_list_append_memory_copy_fn_t = ze_result_t (*)(ze_command_list_handle_t,
                                                                void *,
                                                                const void *,
                                                                size_t,
                                                                ze_event_handle_t,
                                                                uint32_t,
                                                                ze_event_handle_t *);
using ze_command_list_append_memory_fill_fn_t = ze_result_t (*)(ze_command_list_handle_t,
                                                                void *,
                                                                const void *,
                                                                size_t,
                                                                size_t,
                                                                ze_event_handle_t,
                                                                uint32_t,
                                                                ze_event_handle_t *);

/* Page alignment: the buffers are handed to nixl for registration, and a NIC importing a dmabuf
   wants them page-aligned. */
constexpr size_t kAlignment = 4096;

class zeRuntime {
public:
    static zeRuntime &
    get() {
        static zeRuntime instance;
        return instance;
    }

    /* Devices visible to Level Zero, or an empty vector if it is unusable on this host. */
    const std::vector<ze_device_handle_t> &
    devices() {
        std::call_once(once_, [this]() { init(); });
        return devices_;
    }

    ze_result_t
    alloc(void **addr, size_t size, int devid) {
        if (!validDevice(devid)) return ZE_RESULT_ERROR_INVALID_ARGUMENT;

        ze_device_mem_alloc_desc_t desc = {};
        desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        desc.ordinal = 0;

        ze_result_t r =
            fn_mem_alloc_device_(context_, &desc, size, kAlignment, devices_[devid], addr);
        if (r != ZE_RESULT_SUCCESS) return r;

        /* Remember the device so free() and memset() do not need the caller to repeat it -- the
           Neuron and CUDA wrappers this sits alongside take the pointer alone too. */
        std::lock_guard<std::mutex> lock(mutex_);
        ptr_device_[*addr] = devid;
        return ZE_RESULT_SUCCESS;
    }

    ze_result_t
    free(void *addr) {
        if (!ready()) return ZE_RESULT_ERROR_UNINITIALIZED;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ptr_device_.erase(addr);
        }
        return fn_mem_free_(context_, addr);
    }

    /**
     * Copy device memory back to host. Needed for --check_consistency: without it the VRAM path
     * cannot be verified at all, and a transfer that never lands in device memory looks like a
     * success. copyVramToHost() in utils.cpp has Neuron, CUDA and ROCm branches; this is the Level
     * Zero one.
     */
    ze_result_t
    memcpyDtoH(void *host_addr, const void *device_addr, size_t len) {
        if (!ready()) return ZE_RESULT_ERROR_UNINITIALIZED;

        /* The copy has to be queued on the owning device's command list. */
        int devid = 0;
        ze_result_t r = ownerDevice(device_addr, devid);
        if (r != ZE_RESULT_SUCCESS) return r;

        ze_command_list_handle_t cmdlist = nullptr;
        r = fillList(devid, cmdlist);
        if (r != ZE_RESULT_SUCCESS) return r;

        /* Synchronous command list, so this returns only once the copy has completed. */
        return fn_append_memory_copy_(cmdlist, host_addr, device_addr, len, nullptr, 0, nullptr);
    }

    ze_result_t
    memset(void *addr, int val, size_t count) {
        if (!ready()) return ZE_RESULT_ERROR_UNINITIALIZED;

        int devid = 0;
        ze_result_t r = ownerDevice(addr, devid);
        if (r != ZE_RESULT_SUCCESS) return r;

        ze_command_list_handle_t cmdlist = nullptr;
        r = fillList(devid, cmdlist);
        if (r != ZE_RESULT_SUCCESS) return r;

        const uint8_t pattern = static_cast<uint8_t>(val);
        /* The command list is ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, so this returns only once the
           fill has completed and no explicit synchronise call is needed. */
        return fn_append_memory_fill_(
            cmdlist, addr, &pattern, sizeof(pattern), count, nullptr, 0, nullptr);
    }

private:
    zeRuntime() = default;
    zeRuntime(const zeRuntime &) = delete;
    zeRuntime &
    operator=(const zeRuntime &) = delete;

    bool
    ready() {
        return !devices().empty() && context_ != nullptr;
    }

    bool
    validDevice(int devid) {
        return ready() && devid >= 0 && static_cast<size_t>(devid) < devices_.size();
    }

    /* Index in devices_ of the device that owns @a addr, which may point anywhere inside an
       allocation: the consistency check copies from offsets into a buffer, not only from its base,
       and on a multi-GPU host queuing that on the wrong device's list fails or depends on P2P. */
    ze_result_t
    ownerDevice(const void *addr, int &devid) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = ptr_device_.find(const_cast<void *>(addr));
            if (it != ptr_device_.end()) {
                devid = it->second;
                return ZE_RESULT_SUCCESS;
            }
        }

        ze_memory_allocation_properties_t props = {};
        props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
        ze_device_handle_t device = nullptr;
        ze_result_t r = fn_mem_get_alloc_properties_(context_, addr, &props, &device);
        if (r != ZE_RESULT_SUCCESS) return r;
        if (props.type == ZE_MEMORY_TYPE_UNKNOWN) return ZE_RESULT_ERROR_INVALID_ARGUMENT;

        if (device == nullptr) {
            /* A host allocation belongs to no device, and any device's list can reach it. */
            devid = 0;
            return ZE_RESULT_SUCCESS;
        }
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (devices_[i] == device) {
                devid = static_cast<int>(i);
                return ZE_RESULT_SUCCESS;
            }
        }
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }

    static void *
    dlopenLibze() {
        static void *const handle = dlopen("libze_loader.so.1", RTLD_NOW);
        return handle;
    }

    template<class Fn>
    static bool
    loadSymbol(void *handle, const char *name, Fn &out) {
        out = reinterpret_cast<Fn>(dlsym(handle, name));
        return out != nullptr;
    }

    void
    init() {
        void *handle = dlopenLibze();
        if (handle == nullptr) return;

        if (!loadSymbol(handle, "zeInit", fn_init_) ||
            !loadSymbol(handle, "zeDriverGet", fn_driver_get_) ||
            !loadSymbol(handle, "zeDeviceGet", fn_device_get_) ||
            !loadSymbol(handle, "zeContextCreate", fn_context_create_) ||
            !loadSymbol(handle, "zeMemAllocDevice", fn_mem_alloc_device_) ||
            !loadSymbol(handle, "zeMemFree", fn_mem_free_) ||
            !loadSymbol(handle, "zeMemGetAllocProperties", fn_mem_get_alloc_properties_) ||
            !loadSymbol(handle,
                        "zeDeviceGetCommandQueueGroupProperties",
                        fn_queue_group_properties_) ||
            !loadSymbol(handle, "zeCommandListCreateImmediate", fn_command_list_create_immediate_) ||
            !loadSymbol(handle, "zeCommandListAppendMemoryFill", fn_append_memory_fill_) ||
            !loadSymbol(handle, "zeCommandListAppendMemoryCopy", fn_append_memory_copy_)) {
            return;
        }

        /* ZE_INIT_FLAG_GPU_ONLY matches what libfabric's hmem_ze.c asks for, so both agree on
           which devices exist and on the device ordinals used to name them. */
        if (fn_init_(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) return;

        uint32_t driver_count = 0;
        if (fn_driver_get_(&driver_count, nullptr) != ZE_RESULT_SUCCESS || driver_count == 0) {
            return;
        }
        std::vector<ze_driver_handle_t> drivers(driver_count);
        if (fn_driver_get_(&driver_count, drivers.data()) != ZE_RESULT_SUCCESS) return;

        /* Driver 0 only: libfabric's ZE support is single-driver and asserts the driver index is 0,
           so a device on any other driver could not be registered anyway. */
        driver_ = drivers[0];

        uint32_t device_count = 0;
        if (fn_device_get_(driver_, &device_count, nullptr) != ZE_RESULT_SUCCESS ||
            device_count == 0) {
            return;
        }
        std::vector<ze_device_handle_t> devices(device_count);
        if (fn_device_get_(driver_, &device_count, devices.data()) != ZE_RESULT_SUCCESS) return;

        ze_context_desc_t context_desc = {};
        context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        if (fn_context_create_(driver_, &context_desc, &context_) != ZE_RESULT_SUCCESS) return;

        devices_ = std::move(devices);
    }

    /* An immediate command list for @a devid, created on first use and cached for the process. */
    ze_result_t
    fillList(int devid, ze_command_list_handle_t &out) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cmdlists_.find(devid);
        if (it != cmdlists_.end()) {
            out = it->second;
            return ZE_RESULT_SUCCESS;
        }

        /* Find a queue group that can carry a fill rather than assuming group 0 does: the compute
           group is group 0 on the current Intel GPUs but that is not guaranteed by the spec. */
        uint32_t group_count = 0;
        ze_result_t r = fn_queue_group_properties_(devices_[devid], &group_count, nullptr);
        if (r != ZE_RESULT_SUCCESS || group_count == 0) {
            return r == ZE_RESULT_SUCCESS ? ZE_RESULT_ERROR_UNINITIALIZED : r;
        }
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto &g : groups) {
            g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        r = fn_queue_group_properties_(devices_[devid], &group_count, groups.data());
        if (r != ZE_RESULT_SUCCESS) return r;

        uint32_t ordinal = group_count;
        for (uint32_t i = 0; i < group_count; ++i) {
            if (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                ordinal = i;
                break;
            }
        }
        if (ordinal == group_count) {
            for (uint32_t i = 0; i < group_count; ++i) {
                if (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) {
                    ordinal = i;
                    break;
                }
            }
        }
        if (ordinal == group_count) return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;

        ze_command_queue_desc_t queue_desc = {};
        queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        queue_desc.ordinal = ordinal;
        queue_desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
        queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

        ze_command_list_handle_t cmdlist = nullptr;
        r = fn_command_list_create_immediate_(context_, devices_[devid], &queue_desc, &cmdlist);
        if (r != ZE_RESULT_SUCCESS) return r;

        cmdlists_[devid] = cmdlist;
        out = cmdlist;
        return ZE_RESULT_SUCCESS;
    }

    std::once_flag once_;
    std::mutex mutex_;

    ze_driver_handle_t driver_ = nullptr;
    ze_context_handle_t context_ = nullptr;
    std::vector<ze_device_handle_t> devices_;
    std::unordered_map<int, ze_command_list_handle_t> cmdlists_;
    std::unordered_map<void *, int> ptr_device_;

    ze_init_fn_t fn_init_ = nullptr;
    ze_driver_get_fn_t fn_driver_get_ = nullptr;
    ze_device_get_fn_t fn_device_get_ = nullptr;
    ze_context_create_fn_t fn_context_create_ = nullptr;
    ze_mem_alloc_device_fn_t fn_mem_alloc_device_ = nullptr;
    ze_mem_free_fn_t fn_mem_free_ = nullptr;
    ze_mem_get_alloc_properties_fn_t fn_mem_get_alloc_properties_ = nullptr;
    ze_device_get_command_queue_group_properties_fn_t fn_queue_group_properties_ = nullptr;
    ze_command_list_create_immediate_fn_t fn_command_list_create_immediate_ = nullptr;
    ze_command_list_append_memory_fill_fn_t fn_append_memory_fill_ = nullptr;
    ze_command_list_append_memory_copy_fn_t fn_append_memory_copy_ = nullptr;
};

} // namespace

int
zeAccelDeviceCount() {
    const auto &devices = zeRuntime::get().devices();
    return devices.empty() ? -1 : static_cast<int>(devices.size());
}

int
zeAccelMalloc(void **addr, size_t buffer_size, int devid) {
    return static_cast<int>(zeRuntime::get().alloc(addr, buffer_size, devid));
}

int
zeAccelFree(void *addr) {
    return static_cast<int>(zeRuntime::get().free(addr));
}

int
zeAccelMemset(void *addr, int val, size_t count) {
    return static_cast<int>(zeRuntime::get().memset(addr, val, count));
}

int
zeAccelMemcpyDtoH(void *host_addr, const void *device_addr, size_t len) {
    return static_cast<int>(zeRuntime::get().memcpyDtoH(host_addr, device_addr, len));
}

#else /* !HAVE_ZE */

int
zeAccelDeviceCount() {
    return -1;
}

int
zeAccelMalloc(void **, size_t, int) {
    return -1;
}

int
zeAccelFree(void *) {
    return -1;
}

int
zeAccelMemset(void *, int, size_t) {
    return -1;
}

int
zeAccelMemcpyDtoH(void *, const void *, size_t) {
    return -1;
}

#endif /* HAVE_ZE */
