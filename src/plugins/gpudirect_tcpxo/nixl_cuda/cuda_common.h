/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#ifndef GPUDIRECT_TCPXO_NIXL_CUDA_CUDA_COMMON_H_
#define GPUDIRECT_TCPXO_NIXL_CUDA_CUDA_COMMON_H_

#include <cstddef>
#include <cstdint>

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "tcpxo_common.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <driver_types.h>
#endif

namespace tcpxo {

// Initializes the CUDA driver. Must be called before any other CUDA driver functions.
absl::Status
InitCuda();

// Initializes gpu related states.
absl::StatusOr<GpuDev> InitGpuDev(absl::string_view);

// Get DMA-BUF fd from memory.
absl::StatusOr<int>
GetDmabufFd(uint64_t cuda_dev_id, void *ptr, size_t size);

// Get the base address and size of the underlying memory allocation.
absl::StatusOr<std::pair<void *, size_t>>
GetDmabufBase(uint64_t cuda_dev_id, void *ptr);

absl::StatusOr<int>
GetDeviceCount();

absl::StatusOr<std::string>
GetPciBusIdFromCudaDevId(int cuda_dev_id);

absl::Status
CheckDeviceCount(int);

#ifdef HAVE_CUDA
// Helper function to convert CUDA Driver API results to absl::Status.
absl::Status CudaDriverApiCall(CUresult);

// Helper function to convert CUDA Runtime API results to absl::Status.
absl::Status CudaRuntimeApiCall(cudaError_t);
#endif

} // namespace tcpxo

#endif // GPUDIRECT_TCPXO_NIXL_CUDA_CUDA_COMMON_H_
