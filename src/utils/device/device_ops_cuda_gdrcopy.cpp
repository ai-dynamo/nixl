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
#include "device/device_ops_cuda_gdrcopy.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "common/nixl_log.h"
#include "device/host_published_mem.h"

namespace nixl {
namespace {

    // GDRCopy ABI, unchanged since 2.0, declared from gdrapi.h
    // (Copyright (c) 2014-2021, NVIDIA CORPORATION, MIT).
    struct gdr;
    using gdr_t = gdr *;

    struct gdr_mh_t {
        unsigned long h;
    };

    struct gdrApi {
        gdr_t (*open)();
        int (*close)(gdr_t);
        int (*pin_buffer)(gdr_t, unsigned long, size_t, uint64_t, uint32_t, gdr_mh_t *);
        int (*unpin_buffer)(gdr_t, gdr_mh_t);
        int (*map)(gdr_t, gdr_mh_t, void **, size_t);
        int (*unmap)(gdr_t, gdr_mh_t, void *, size_t);
        int (*copy_to_mapping)(gdr_mh_t, void *, const void *, size_t);
    };

    /** GDRCopy pins and maps whole GPU pages. */
    constexpr size_t kGpuPageSize = size_t{1} << 16;

    template<typename Fn>
    bool
    loadSymbol(void *lib, const char *name, Fn &fn) noexcept {
        fn = reinterpret_cast<Fn>(dlsym(lib, name));
        return fn != nullptr;
    }

    std::optional<gdrApi>
    loadGdrApi() noexcept {
        // The versioned name first, as NVSHMEM does: runtime-only packages ship just
        // libgdrapi.so.2, and the unversioned name is a development symlink.
        void *lib = dlopen("libgdrapi.so.2", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            lib = dlopen("libgdrapi.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (lib == nullptr) {
            NIXL_INFO << "GDRCopy is unavailable: " << dlerror();
            return std::nullopt;
        }
        gdrApi api{};
        if (!loadSymbol(lib, "gdr_open", api.open) || !loadSymbol(lib, "gdr_close", api.close) ||
            !loadSymbol(lib, "gdr_pin_buffer", api.pin_buffer) ||
            !loadSymbol(lib, "gdr_unpin_buffer", api.unpin_buffer) ||
            !loadSymbol(lib, "gdr_map", api.map) || !loadSymbol(lib, "gdr_unmap", api.unmap) ||
            !loadSymbol(lib, "gdr_copy_to_mapping", api.copy_to_mapping)) {
            NIXL_WARN << "GDRCopy is unusable: " << dlerror();
            dlclose(lib);
            return std::nullopt;
        }
        // A missing driver fails every gdr_open, so detect it once rather than per allocation.
        const gdr_t probe = api.open();
        if (probe == nullptr) {
            NIXL_INFO << "GDRCopy is unavailable: gdr_open failed";
            dlclose(lib);
            return std::nullopt;
        }
        api.close(probe);
        return api;
    }

    /** Loads GDRCopy on first use and keeps it for the process; nullptr when unusable. */
    const gdrApi *
    gdrcopy() noexcept {
        static const std::optional<gdrApi> api = loadGdrApi();
        return api ? &*api : nullptr;
    }

    /**
     * Device memory the CPU writes through a GDRCopy BAR mapping.
     *
     * Each instance opens its own GDRCopy handle because libgdrapi does not lock the handle's
     * pin and map bookkeeping.
     */
    class gdrPublishedMem final : public hostPublishedDeviceMem {
    public:
        gdrPublishedMem(const gdrApi &api,
                        void *storage,
                        void *dev_ptr,
                        size_t size,
                        size_t mapped_size) noexcept
            : hostPublishedDeviceMem(dev_ptr, size, backing::device),
              api_(api),
              storage_(storage),
              mappedSize_(mapped_size) {}

        ~gdrPublishedMem() override {
            if (cpu_ != nullptr) {
                api_.unmap(gdr_, mh_, cpu_, mappedSize_);
            }
            if (pinned_) {
                api_.unpin_buffer(gdr_, mh_);
            }
            if (gdr_ != nullptr) {
                api_.close(gdr_);
            }
            const cudaError_t error = cudaFree(storage_);
            if (error != cudaSuccess) {
                NIXL_WARN << "cudaFree failed: " << cudaGetErrorString(error);
            }
        }

        /** @brief Map the storage for CPU writes and zero it. */
        [[nodiscard]] nixl_status_t
        map() noexcept {
            gdr_ = api_.open();
            if (gdr_ == nullptr) {
                NIXL_WARN << "GDRCopy cannot back host-published memory: gdr_open failed";
                return NIXL_ERR_NOT_SUPPORTED;
            }
            if (api_.pin_buffer(gdr_,
                                reinterpret_cast<unsigned long>(devicePointer()),
                                mappedSize_,
                                0,
                                0,
                                &mh_) != 0) {
                NIXL_WARN << "GDRCopy cannot back host-published memory: gdr_pin_buffer failed";
                return NIXL_ERR_NOT_SUPPORTED;
            }
            pinned_ = true;
            void *cpu = nullptr;
            if (api_.map(gdr_, mh_, &cpu, mappedSize_) != 0) {
                NIXL_WARN << "GDRCopy cannot back host-published memory: gdr_map failed";
                return NIXL_ERR_NOT_SUPPORTED;
            }
            cpu_ = static_cast<uint8_t *>(cpu);

            // Zero through the mapping: an asynchronous cudaMemset could land after a later
            // publish().
            constexpr uint64_t zero = 0;
            for (size_t offset = 0; offset < size(); offset += sizeof(zero)) {
                if (api_.copy_to_mapping(mh_, cpu_ + offset, &zero, sizeof(zero)) != 0) {
                    NIXL_WARN << "GDRCopy cannot back host-published memory: zeroing failed";
                    return NIXL_ERR_BACKEND;
                }
            }
            return NIXL_SUCCESS;
        }

    private:
        nixl_status_t
        doPublish(size_t offset, uint64_t value) noexcept override {
            // gdr_copy_to_mapping fences only after its store, so order earlier host writes here.
            std::atomic_thread_fence(std::memory_order_release);
            return api_.copy_to_mapping(mh_, cpu_ + offset, &value, sizeof(value)) == 0 ?
                NIXL_SUCCESS :
                NIXL_ERR_BACKEND;
        }

        const gdrApi &api_;
        void *storage_;
        size_t mappedSize_;
        gdr_t gdr_ = nullptr;
        gdr_mh_t mh_{};
        bool pinned_ = false;
        uint8_t *cpu_ = nullptr;
    };

} // namespace

nixl_status_t
allocGdrCopyPublishedMem(size_t size, std::unique_ptr<hostPublishedDeviceMem> &out) noexcept {
    const gdrApi *api = gdrcopy();
    if (api == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (size > std::numeric_limits<size_t>::max() - 2 * kGpuPageSize) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Over-allocate by a page so the mapped range can start on a GPU page boundary.
    const size_t mapped_size = (size + kGpuPageSize - 1) & ~(kGpuPageSize - 1);
    void *storage;
    const cudaError_t error = cudaMalloc(&storage, mapped_size + kGpuPageSize - 1);
    if (error != cudaSuccess) {
        NIXL_WARN << "GDRCopy cannot back host-published memory: cudaMalloc failed: "
                  << cudaGetErrorString(error);
        return NIXL_ERR_BACKEND;
    }
    const auto aligned =
        (reinterpret_cast<uintptr_t>(storage) + kGpuPageSize - 1) & ~uintptr_t{kGpuPageSize - 1};

    auto mem = std::make_unique<gdrPublishedMem>(
        *api, storage, reinterpret_cast<void *>(aligned), size, mapped_size);
    const nixl_status_t status = mem->map();
    if (status != NIXL_SUCCESS) {
        return status;
    }
    out = std::move(mem);
    return NIXL_SUCCESS;
}

} // namespace nixl
