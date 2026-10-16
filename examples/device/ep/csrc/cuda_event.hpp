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

#pragma once

#include "cuda_warn.hpp"
#include "kernels/exception.cuh"

#include <cuda_runtime.h>

namespace nixl_ep::cuda {
class Event {
public:
    Event() : event(create()) {}

    Event(const Event &) = delete;
    Event &
    operator=(const Event &) = delete;

    Event(Event &&other) noexcept : event(other.event) {
        other.event = nullptr;
    }

    Event &
    operator=(Event &&other) noexcept {
        if (this != &other) {
            destroy();
            event = other.event;
            other.event = nullptr;
        }
        return *this;
    }

    ~Event() noexcept {
        destroy();
    }

    [[nodiscard]] cudaEvent_t
    get() const noexcept {
        return event;
    }

    void
    record(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(event, stream));
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
        if (event == nullptr) {
            return;
        }
        warn(cudaEventDestroy(event), "cuda::Event::destroy()", "cudaEventDestroy");
        event = nullptr;
    }

    cudaEvent_t event;
};
} // namespace nixl_ep::cuda
