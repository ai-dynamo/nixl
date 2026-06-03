/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

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

    [[nodiscard]] cudaEvent_t
    get() const noexcept {
        return event_;
    }

    void
    record(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
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
} // namespace nixl_ep
