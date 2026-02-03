/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Amazon.com, Inc. and affiliates.
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

#include "neuron.h"

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

void *
dlopen_libnrt() {
#define TRY_DLOPEN(path)                       \
    do {                                       \
        void *handle = dlopen(path, RTLD_NOW); \
        if (handle) return handle;             \
    } while (0)

    static void *const libnrt_handle = []() -> void * {
        TRY_DLOPEN("/opt/aws/neuron/lib/libnrt.so.1");
        TRY_DLOPEN("libnrt.so.1");
        return nullptr;
    }();

#undef TRY_DLOPEN

    return libnrt_handle;
}

template<class Fn>
Fn *
_load_nrt_symbol(const char *fn_name, Fn *) {
    void *libnrt_handle = dlopen_libnrt();
    if (libnrt_handle) {
        return reinterpret_cast<Fn *>(dlsym(libnrt_handle, fn_name));
    }
    return nullptr;
}

#define LOAD_NRT_SYMBOL(sym) _load_nrt_symbol(#sym, &sym)

int
nrt_init(int framework, const char *fw_version, const char *fal_version) {
    static const auto fn = LOAD_NRT_SYMBOL(nrt_init);
    if (!fn) return -1;
    return fn(framework, fw_version, fal_version);
}

struct nrt_tensor;

int
nrt_tensor_allocate(int tensor_placement,
                    int vnc,
                    size_t size,
                    const char *name,
                    nrt_tensor **tensor) {
    static const auto fn = LOAD_NRT_SYMBOL(nrt_tensor_allocate);
    if (!fn) return -1;
    return fn(tensor_placement, vnc, size, name, tensor);
}

void
nrt_tensor_free(nrt_tensor **tensor) {
    static const auto fn = LOAD_NRT_SYMBOL(nrt_tensor_free);
    return fn(tensor);
}

int
nrt_tensor_read(const nrt_tensor *tensor, void *buf, size_t offset, size_t size) {
    static const auto fn = LOAD_NRT_SYMBOL(nrt_tensor_read);
    if (!fn) return -1;
    return fn(tensor, buf, offset, size);
}

int
nrt_tensor_write(nrt_tensor *tensor, const void *buf, size_t offset, size_t size) {
    static const auto fn = LOAD_NRT_SYMBOL(nrt_tensor_write);
    if (!fn) return -1;
    return fn(tensor, buf, offset, size);
}

void *
nrt_tensor_get_va(const nrt_tensor *tensor) {
    static const auto fn = LOAD_NRT_SYMBOL(nrt_tensor_get_va);
    if (!fn) return nullptr;
    return fn(tensor);
}

int
nrt_get_visible_vnc_count(uint32_t *vnc_count) {
    static const auto fn = LOAD_NRT_SYMBOL(nrt_get_visible_vnc_count);
    if (!fn) return -1;
    return fn(vnc_count);
}

struct NrtTensorDeleter {
    void
    operator()(nrt_tensor *tensor) const {
        nrt_tensor_free(&tensor);
    }
};

using NrtTensorPtr = std::unique_ptr<nrt_tensor, NrtTensorDeleter>;

std::unordered_map<const void *, NrtTensorPtr> allocation_tracker;
std::mutex allocation_tracker_mutex;

nrt_tensor *
getTensorFromVA(const void *va) {
    std::lock_guard lock{allocation_tracker_mutex};

    auto it = allocation_tracker.find(va);
    if (it == allocation_tracker.end()) {
        return nullptr;
    }
    return it->second.get();
}

} // namespace

int
neuronCoreCount() {
    static const int core_count = []() {
        uint32_t vnc_count;
        if (nrt_init(1 /* framework_type=NO_FW */, "nixl_bench", "nixl_bench") == 0 &&
            nrt_get_visible_vnc_count(&vnc_count) == 0) {
            return static_cast<int>(vnc_count);
        }
        return -1;
    }();

    return core_count;
}

int
neuronMalloc(void **addr, size_t buffer_size, int devid) {
    nrt_tensor *tensor;
    int status;

    status = nrt_tensor_allocate(0 /* placement=device */, devid, buffer_size, nullptr, &tensor);
    if (status != 0) return status;

    NrtTensorPtr ptr{tensor};
    *addr = nrt_tensor_get_va(tensor);
    if (*addr == nullptr) {
        return -1;
    }

    std::lock_guard lock{allocation_tracker_mutex};
    allocation_tracker.emplace(*addr, std::move(ptr));

    return 0;
}

int
neuronFree(void *addr) {
    if (!addr) return 0;

    std::lock_guard lock{allocation_tracker_mutex};
    return allocation_tracker.erase(addr) - 1;
}

int
neuronMemcpy(void *dest, const void *src, size_t count, neuronMemcpyKind kind) {
    nrt_tensor *tensor = getTensorFromVA(kind == neuronMemcpyHostToDevice ? dest : src);
    if (tensor == nullptr) {
        return -1;
    }

    if (kind == neuronMemcpyHostToDevice) {
        return nrt_tensor_write(tensor, src, 0, count);
    } else {
        return nrt_tensor_read(tensor, dest, 0, count);
    }
}

int
neuronMemset(void *addr, int val, size_t count) {
    nrt_tensor *tensor = getTensorFromVA(addr);
    if (tensor == nullptr) {
        return -1;
    }

    constexpr size_t kMaxChunkSize = 1UL << 21; // 2MB
    std::vector<unsigned char> buf(kMaxChunkSize, static_cast<unsigned char>(val));
    int status = 0;
    size_t offset = 0;
    while (offset < count && status == 0) {
        const size_t write_len = std::min(kMaxChunkSize, count - offset);
        status = nrt_tensor_write(tensor, buf.data(), offset, write_len);
        offset += write_len;
    }
    return status;
}
