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
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
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

// NrtAllocation owns (and manages) the tensor passed in.
struct NrtAllocation {
    std::unique_ptr<nrt_tensor, NrtTensorDeleter> tensor;
    size_t size;

    NrtAllocation(nrt_tensor *t, size_t s) : tensor(t), size(s) {}

    NrtAllocation(NrtAllocation &&) = default;
};

// Sorted by base address for O(log n) range lookup via upper_bound.
std::map<uintptr_t, NrtAllocation> allocation_tracker;
std::mutex allocation_tracker_mutex;

// Find the allocation containing the given VA. Caller must hold allocation_tracker_mutex.
// Returns {tensor, offset, alloc_size} or {nullptr, 0, 0}.
std::tuple<nrt_tensor *, size_t, size_t>
findTensorForVA(const void *va) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(va);

    // Find the first entry with base > addr, then step back one.
    auto it = allocation_tracker.upper_bound(addr);
    if (it == allocation_tracker.begin()) {
        // This is the case either addr is smaller than all in the map, or
        // in an empty map, where begin() == end(), it is set to end().
        return {nullptr, 0, 0};
    }
    --it;
    size_t offset = addr - it->first;
    if (offset < it->second.size) {
        return {it->second.tensor.get(), offset, it->second.size};
    }
    return {nullptr, 0, 0};
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
    if (status != 0) {
        return status;
    }

    *addr = nrt_tensor_get_va(tensor);
    if (*addr == nullptr) {
        nrt_tensor_free(&tensor);
        return -1;
    }

    std::lock_guard<std::mutex> lock{allocation_tracker_mutex};
    uintptr_t base = reinterpret_cast<uintptr_t>(*addr);
    auto [it, inserted] = allocation_tracker.emplace(base, NrtAllocation{tensor, buffer_size});
    if (!inserted) {
        *addr = nullptr;
        return -1;
    }

    return 0;
}

int
neuronFree(void *addr) {
    if (!addr) {
        return 0;
    }

    std::lock_guard<std::mutex> lock{allocation_tracker_mutex};
    const auto erased = allocation_tracker.erase(reinterpret_cast<uintptr_t>(addr));
    return erased == 1 ? 0 : -1;
}

int
neuronMemcpy(void *dest, const void *src, size_t count, neuronMemcpyKind kind) {
    std::lock_guard<std::mutex> lock{allocation_tracker_mutex};
    const void *device_addr = (kind == neuronMemcpyHostToDevice) ? dest : src;
    auto [tensor, offset, alloc_size] = findTensorForVA(device_addr);
    if (tensor == nullptr) {
        std::cerr << "neuronMemcpy: no allocation found for VA " << device_addr
                  << " (kind=" << (kind == neuronMemcpyHostToDevice ? "H2D" : "D2H")
                  << ", count=" << count << ")" << std::endl;
        return -1;
    }
    if (count > alloc_size - offset) {
        std::cerr << "neuronMemcpy: access out of bounds (offset=" << offset << ", count=" << count
                  << ", alloc_size=" << alloc_size << ")" << std::endl;
        return -1;
    }

    int status;
    if (kind == neuronMemcpyHostToDevice) {
        status = nrt_tensor_write(tensor, src, offset, count);
    } else {
        status = nrt_tensor_read(tensor, dest, offset, count);
    }
    if (status != 0) {
        std::cerr << "neuronMemcpy: "
                  << (kind == neuronMemcpyHostToDevice ? "nrt_tensor_write" : "nrt_tensor_read")
                  << " failed with status " << status << " (offset=" << offset
                  << ", count=" << count << ")" << std::endl;
    }
    return status;
}

int
neuronMemset(void *addr, int val, size_t count) {
    std::lock_guard<std::mutex> lock{allocation_tracker_mutex};
    auto [tensor, offset, alloc_size] = findTensorForVA(addr);
    if (tensor == nullptr) {
        return -1;
    }
    if (count > alloc_size - offset) {
        std::cerr << "neuronMemset: access out of bounds (offset=" << offset << ", count=" << count
                  << ", alloc_size=" << alloc_size << ")" << std::endl;
        return -1;
    }

    constexpr size_t kMaxChunkSize = 1UL << 21; // 2MB
    std::vector<unsigned char> buf(kMaxChunkSize, static_cast<unsigned char>(val));
    int status = 0;
    size_t pos = 0;
    while (pos < count && status == 0) {
        const size_t write_len = std::min(kMaxChunkSize, count - pos);
        status = nrt_tensor_write(tensor, buf.data(), offset + pos, write_len);
        pos += write_len;
    }
    return status;
}
