/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cuda_warn.hpp"
#include "kernels/exception.cuh"

#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <memory>

namespace nixl_ep {

class cudaEvent {
public:
    cudaEvent() : event_{create()} {}

    cudaEvent(const cudaEvent &) = delete;
    cudaEvent &
    operator=(const cudaEvent &) = delete;

    cudaEvent(cudaEvent &&other) noexcept : event_{other.event_} {
        other.event_ = nullptr;
    }

    cudaEvent &
    operator=(cudaEvent &&other) noexcept {
        if (this != &other) {
            destroy();
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    ~cudaEvent() noexcept {
        destroy();
    }

    void
    record(const at::cuda::CUDAStream &stream) {
        CUDA_CHECK(cudaEventRecord(event_, stream.stream()));
    }

    void
    block(const at::cuda::CUDAStream &stream) const {
        CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), event_, 0));
    }

private:
    static cudaEvent_t
    create() {
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return event;
    }

    void
    destroy() noexcept {
        if (event_ == nullptr) {
            return;
        }
        warnCuda(cudaEventDestroy(event_), "Event::destroy()", "cudaEventDestroy");
        event_ = nullptr;
    }

    cudaEvent_t event_;
};

struct EventHandle {
    EventHandle() : event_{std::make_shared<cudaEvent>()} {
        event_->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const at::cuda::CUDAStream &stream)
        : event_{std::make_shared<cudaEvent>()} {
        event_->record(stream);
    }

    EventHandle(const EventHandle &other) = default;

    void
    currentStreamWait() const {
        event_->block(at::cuda::getCurrentCUDAStream());
    }

    void streamWait(const at::cuda::CUDAStream &stream) const {
        event_->block(stream);
    }

private:
    std::shared_ptr<cudaEvent> event_;
};

inline void
streamWait(const at::cuda::CUDAStream &s_0, const at::cuda::CUDAStream &s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    cudaEvent e;
    e.record(s_1);
    e.block(s_0);
}

} // namespace nixl_ep
