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

#include "cuda_event.hpp"
#include "cuda_stream.hpp"
#include "cuda_warn.hpp"
#include "kernels/exception.cuh"

#include <ATen/cuda/CUDAContext.h>

#include <memory>

namespace nixl_ep {

class EventHandle {
public:
    EventHandle() : event{std::make_shared<cuda::Event>()} {
        event->record(cuda_stream::getCurrent());
    }

    explicit EventHandle(cudaStream_t stream)
        : event{std::make_shared<cuda::Event>()} {
        event->record(stream);
    }

    EventHandle(const EventHandle &other) = default;

    void
    current_stream_wait() const {
        stream_wait(cuda_stream::getCurrent());
    }

    void
    stream_wait(cudaStream_t stream) const {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event->get(), 0));
    }

private:
    std::shared_ptr<cuda::Event> event;
};

inline void
stream_wait(cudaStream_t stream_0, cudaStream_t stream_1) {
    EP_HOST_ASSERT(stream_0 != stream_1);
    cuda::Event event;
    event.record(stream_1);
    CUDA_CHECK(cudaStreamWaitEvent(stream_0, event.get(), 0));
}
} // namespace nixl_ep
