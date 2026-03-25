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
#ifndef NIXL_TEST_GTEST_PLUGINS_MEMORY_HANDLER_H
#define NIXL_TEST_GTEST_PLUGINS_MEMORY_HANDLER_H

#include <absl/strings/str_format.h>
#include "backend/backend_aux.h"
#include "backend_engine.h"
#include "common/nixl_log.h"
#include "nixl.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gtest::plugins {

template<nixl_mem_t memType> class memoryHandler;

template<> class memoryHandler<DRAM_SEG> {
public:
    memoryHandler(size_t len, int dev_id) : buf_(len), len_(len), devId_(dev_id), md_(nullptr) {}

    ~memoryHandler() = default;

    void
    setIncreasing(uint8_t start_byte) {
        for (auto &entry : buf_)
            entry = start_byte++;
    }

    bool
    checkIncreasing(uint8_t start_byte) {
        for (auto &entry : buf_) {
            uint8_t expected_byte = start_byte++;
            if (entry != expected_byte) {
                NIXL_ERROR << "Verification failed! local: " << entry
                           << ", expected: " << expected_byte;
                return false;
            }
        }
        return true;
    }

    void
    reset() {
        std::fill(buf_.begin(), buf_.end(), 0x00);
    }

    void
    populateBlobDesc(nixlBlobDesc *desc, int buf_index = 0) {
        desc->addr = reinterpret_cast<uintptr_t>(buf_.data());
        desc->len = len_;
        desc->devId = devId_;
    }

    void
    populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
        desc->addr = reinterpret_cast<uintptr_t>(buf_.data()) + entry_index * entry_size;
        desc->len = entry_size;
        desc->devId = devId_;
        desc->metadataP = md_;
    }

    void
    setMD(nixlBackendMD *md) {
        md_ = md;
    }

    nixlBackendMD *
    getMD() {
        return md_;
    }

private:
    std::vector<uint8_t> buf_;
    size_t len_;
    int devId_;
    nixlBackendMD *md_;
};

template<> class memoryHandler<OBJ_SEG> {
public:
    memoryHandler(size_t len, int dev_id) : len_(len), devId_(dev_id), md_(nullptr) {}

    ~memoryHandler() = default;

    void
    populateBlobDesc(nixlBlobDesc *desc, int buf_index = 0) {
        desc->addr = 0;
        desc->len = len_;
        desc->devId = devId_;
        desc->metaInfo = absl::StrFormat("test-obj-key-%d", buf_index);
    }

    void
    populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
        desc->addr = 0;
        desc->len = len_;
        desc->devId = devId_;
        desc->metadataP = md_;
    }

    void
    setMD(nixlBackendMD *md) {
        md_ = md;
    }

    nixlBackendMD *
    getMD() {
        return md_;
    }

private:
    size_t len_;
    int devId_;
    nixlBackendMD *md_;
};

#ifdef HAVE_CUDA
/**
 * @brief Memory handler specialization for GPU (VRAM) memory segments.
 *
 * Manages CUDA device memory allocation and transfers. Uses host-side staging
 * buffers for data initialization (setIncreasing) and verification (checkIncreasing),
 * copying between host and device via cudaMemcpy.
 */
template<> class memoryHandler<VRAM_SEG> {
public:
    memoryHandler(size_t len, int dev_id) : buf_(nullptr), len_(len), devId_(dev_id), md_(nullptr) {
        cudaError_t err = cudaSetDevice(dev_id);
        if (err != cudaSuccess) {
            NIXL_ERROR << "cudaSetDevice(" << dev_id << ") failed: " << cudaGetErrorString(err);
            throw std::runtime_error("cudaSetDevice failed");
        }
        err = cudaMalloc(&buf_, len_);
        if (err != cudaSuccess) {
            buf_ = nullptr;
            NIXL_ERROR << "cudaMalloc(" << len_ << ") failed: " << cudaGetErrorString(err);
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    ~memoryHandler() {
        if (buf_) {
            cudaError_t err = cudaSetDevice(devId_);
            if (err != cudaSuccess) {
                NIXL_ERROR << "cudaSetDevice(" << devId_
                           << ") failed in destructor: " << cudaGetErrorString(err);
            }
            err = cudaFree(buf_);
            if (err != cudaSuccess) {
                NIXL_ERROR << "cudaFree failed: " << cudaGetErrorString(err);
            }
        }
    }

    void
    setIncreasing(uint8_t start_byte) {
        std::vector<uint8_t> host(len_);
        for (auto &entry : host)
            entry = start_byte++;
        cudaError_t err = cudaSetDevice(devId_);
        if (err != cudaSuccess) {
            NIXL_ERROR << "cudaSetDevice(" << devId_ << ") failed: " << cudaGetErrorString(err);
            throw std::runtime_error("cudaSetDevice failed");
        }
        err = cudaMemcpy(buf_, host.data(), len_, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            NIXL_ERROR << "cudaMemcpy H2D failed: " << cudaGetErrorString(err);
            throw std::runtime_error("cudaMemcpy failed");
        }
    }

    bool
    checkIncreasing(uint8_t start_byte) {
        std::vector<uint8_t> host(len_);
        cudaError_t err = cudaSetDevice(devId_);
        if (err != cudaSuccess) {
            NIXL_ERROR << "cudaSetDevice(" << devId_ << ") failed: " << cudaGetErrorString(err);
            return false;
        }
        err = cudaMemcpy(host.data(), buf_, len_, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            NIXL_ERROR << "cudaMemcpy D2H failed: " << cudaGetErrorString(err);
            return false;
        }
        for (auto &entry : host) {
            uint8_t expected_byte = start_byte++;
            if (entry != expected_byte) {
                NIXL_ERROR << "Verification failed! local: " << entry
                           << ", expected: " << expected_byte;
                return false;
            }
        }
        return true;
    }

    void
    reset() {
        cudaError_t err = cudaSetDevice(devId_);
        if (err != cudaSuccess) {
            NIXL_ERROR << "cudaSetDevice(" << devId_ << ") failed: " << cudaGetErrorString(err);
            throw std::runtime_error("cudaSetDevice failed");
        }
        err = cudaMemset(buf_, 0x00, len_);
        if (err != cudaSuccess) {
            NIXL_ERROR << "cudaMemset failed: " << cudaGetErrorString(err);
            throw std::runtime_error("cudaMemset failed");
        }
    }

    void
    populateBlobDesc(nixlBlobDesc *desc, int buf_index = 0) {
        desc->addr = reinterpret_cast<uintptr_t>(buf_);
        desc->len = len_;
        desc->devId = devId_;
    }

    void
    populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
        desc->addr = reinterpret_cast<uintptr_t>(buf_) + entry_index * entry_size;
        desc->len = entry_size;
        desc->devId = devId_;
        desc->metadataP = md_;
    }

    void
    setMD(nixlBackendMD *md) {
        md_ = md;
    }

    nixlBackendMD *
    getMD() {
        return md_;
    }

private:
    void *buf_;
    size_t len_;
    int devId_;
    nixlBackendMD *md_;
};
#endif // HAVE_CUDA

} // namespace gtest::plugins
#endif // NIXL_TEST_GTEST_PLUGINS_MEMORY_HANDLER_H
