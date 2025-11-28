/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "device_kernels.cuh"
#include "device_utils.cuh"

namespace {

constexpr size_t maxThreadsPerBlock = 256;

template<nixl_gpu_level_t level>
__device__ constexpr size_t threadsPerRequest() {
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        return 1;
    } else if constexpr (level == nixl_gpu_level_t::WARP) {
        return 32;
    } else {
        return maxThreadsPerBlock;
    }
}

template<nixl_gpu_level_t level>
__device__ constexpr size_t sharedRequestCount() {
    return maxThreadsPerBlock / threadsPerRequest<level>();
}

template<nixl_gpu_level_t level>
__device__ nixl_status_t
doOperation(const NixlDeviceKernelParams &params,
            nixlGpuXferStatusH *req_ptr) {
    nixl_status_t status;

    switch (params.operation) {
    case NixlDeviceOperation::SINGLE_WRITE:
        status = nixlGpuPostSingleWriteXferReq<level>(
            params.reqHandle,
            params.singleWrite.index,
            params.singleWrite.localOffset,
            params.singleWrite.remoteOffset,
            params.singleWrite.size,
            params.singleWrite.channelId,
            params.withNoDelay,
            req_ptr);
        break;

    case NixlDeviceOperation::PARTIAL_WRITE:
        status = nixlGpuPostPartialWriteXferReq<level>(
            params.reqHandle,
            params.partialWrite.count,
            params.partialWrite.descIndices,
            params.partialWrite.sizes,
            params.partialWrite.localOffsets,
            params.partialWrite.remoteOffsets,
            params.partialWrite.signalDescIndex,
            params.partialWrite.signalInc,
            params.partialWrite.signalOffset,
            params.partialWrite.channelId,
            params.withNoDelay,
            req_ptr);
        break;

    case NixlDeviceOperation::FULL_WRITE:
        status = nixlGpuPostWriteXferReq<level>(
            params.reqHandle,
            params.fullWrite.signalInc,
            params.fullWrite.channelId,
            params.withNoDelay,
            req_ptr);
        break;

    case NixlDeviceOperation::SIGNAL_POST:
        status = nixlGpuPostSignalXferReq<level>(
            params.reqHandle,
            params.signalPost.signalDescIndex,
            params.signalPost.signalInc,
            params.signalPost.signalOffset,
            params.signalPost.channelId,
            params.withNoDelay,
            req_ptr);
        break;

    case NixlDeviceOperation::SIGNAL_READ: {
        if (params.signalRead.signalAddr == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }

        uint64_t value;
        do {
            value = nixlGpuReadSignal<level>(params.signalRead.signalAddr);
        } while (value != params.signalRead.expectedValue);

        if (params.signalRead.resultPtr != nullptr) {
            *params.signalRead.resultPtr = value;
        }
        return NIXL_SUCCESS;
    }

    case NixlDeviceOperation::SIGNAL_WRITE:
        if (params.signalWrite.signalAddr == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }
        nixlGpuWriteSignal<level>(params.signalWrite.signalAddr,
                                  params.signalWrite.value);
        return NIXL_SUCCESS;

    default:
        return NIXL_ERR_INVALID_PARAM;
    }

    if (status != NIXL_IN_PROG) {
        return (status == NIXL_SUCCESS) ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }

    if (!params.withNoDelay || (req_ptr == nullptr)) {
        return NIXL_SUCCESS;
    }

    do {
        status = nixlGpuGetXferStatus<level>(*req_ptr);
    } while (status == NIXL_IN_PROG);

    return status;
}


template<nixl_gpu_level_t level>
__device__ void
kernelJob(const NixlDeviceKernelParams &params,
          NixlDeviceKernelResult *result_ptr) {
    if (result_ptr == nullptr) {
        return;
    }

    nixl_status_t &status = result_ptr->status;

    if (blockDim.x > maxThreadsPerBlock) {
        status = NIXL_ERR_INVALID_PARAM;
        return;
    }

    if (params.numIters == 0) {
        status = NIXL_ERR_INVALID_PARAM;
        return;
    }

    __shared__ nixlGpuXferStatusH shared_reqs[sharedRequestCount<level>()];
    nixlGpuXferStatusH *req_ptr = nullptr;
    if (params.withRequest) {
        const size_t req_index = threadIdx.x / threadsPerRequest<level>();
        req_ptr = &shared_reqs[req_index];
    }

    for (size_t i = 0; i < params.numIters - 1; i++) {
        status = doOperation<level>(params, req_ptr);
        if (status != NIXL_SUCCESS) {
            return;
        }
    }

    // Last iteration forces completion to ensure all operations are finished
    NixlDeviceKernelParams params_force_completion = params;
    params_force_completion.withNoDelay = true;
    nixlGpuXferStatusH *status_ptr = nullptr;
    if (params.withRequest) {
        const size_t req_index = threadIdx.x / threadsPerRequest<level>();
        status_ptr = &shared_reqs[req_index];
    }
    status = doOperation<level>(params_force_completion, status_ptr);
}

template<nixl_gpu_level_t level>
__global__ void
nixlTestKernel(const NixlDeviceKernelParams params,
               NixlDeviceKernelResult *result_ptr) {
    kernelJob<level>(params, result_ptr);
    __threadfence_system();
}

} // namespace

NixlDeviceKernelResult
launchNixlDeviceKernel(const NixlDeviceKernelParams &params) {
    CudaArray<NixlDeviceKernelResult> result(1);
    NixlDeviceKernelResult init_result{NIXL_ERR_INVALID_PARAM};
    result.copyFromHost(&init_result, 1);

    switch (params.level) {
    case nixl_gpu_level_t::THREAD:
        nixlTestKernel<nixl_gpu_level_t::THREAD>
            <<<params.numBlocks, params.numThreads>>>(params, result.get());
        break;
    case nixl_gpu_level_t::WARP:
        nixlTestKernel<nixl_gpu_level_t::WARP>
            <<<params.numBlocks, params.numThreads>>>(params, result.get());
        break;
    case nixl_gpu_level_t::BLOCK:
        nixlTestKernel<nixl_gpu_level_t::BLOCK>
            <<<params.numBlocks, params.numThreads>>>(params, result.get());
        break;
    default: {
        NixlDeviceKernelResult error_result{NIXL_ERR_INVALID_PARAM};
        return error_result;
    }
    }

    const nixl_status_t sync_status = checkCudaErrors();
    if (sync_status != NIXL_SUCCESS) {
        NixlDeviceKernelResult error_result{sync_status};
        return error_result;
    }

    NixlDeviceKernelResult host_result;
    result.copyToHost(&host_result, 1);
    return host_result;
}
