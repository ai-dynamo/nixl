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

#include "device_test_base.cuh"
#include "device_utils.cuh"
#include <algorithm>

namespace gtest::nixl::gpu::partial_write {

__device__ inline void
printPartialWriteError(const char *operation, int thread_id, size_t iteration, nixl_status_t status) {
    printf("Thread %d: %s failed iteration %d: status=%d\n",
           thread_id,
           operation,
           (int)iteration, // cast to int because printf issue size_t
           status);
}

template<nixl_gpu_level_t level>
__global__ void
testPartialWriteKernel(nixlGpuXferReqH req_hdnl_data,
                       size_t count,
                       const unsigned *indices,
                       const size_t *sizes,
                       void *const *addrs,
                       const uint64_t *remote_addrs,
                       uint64_t signal_inc,
                       uint64_t signal_remote_addr,
                       unsigned signal_index,
                       size_t num_iters,
                       bool is_no_delay,
                       bool use_xfer_status,
                       unsigned long long *start_time_ptr,
                       unsigned long long *end_time_ptr,
                       nixl_status_t *error_status) {
    __shared__ nixlGpuXferStatusH xfer_status[MAX_THREADS];
    nixlGpuXferStatusH *xfer_status_ptr =
        use_xfer_status ? &xfer_status[getReqIdx<level>()] : nullptr;

    if (threadIdx.x == 0) {
        *start_time_ptr = getTimeNs();
    }

    __syncthreads();

    for (size_t i = 0; i < num_iters; ++i) {
        nixlGpuSignal signal = {signal_inc, signal_remote_addr};

        nixl_status_t status = nixlGpuPostPartialWriteXferReq<level>(req_hdnl_data,
                                                                     count,
                                                                     indices,
                                                                     sizes,
                                                                     addrs,
                                                                     remote_addrs,
                                                                     signal,
                                                                     signal_index,
                                                                     is_no_delay,
                                                                     xfer_status_ptr);
        if (status != NIXL_SUCCESS) {
            printPartialWriteError("nixlGpuPostPartialWriteXferReq", threadIdx.x, i, status);
            *error_status = status;
            return;
        }

        if (use_xfer_status) {
            do {
                status = nixlGpuGetXferStatus<level>(*xfer_status_ptr);
                if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                    printProgressError(threadIdx.x, i, status);
                    *error_status = status;
                    return;
                }
            } while (status == NIXL_IN_PROG);
        }

        if (status != NIXL_SUCCESS) {
            printCompletionError(threadIdx.x, i, status);
            *error_status = status;
            return;
        }
    }

    if (threadIdx.x == 0) {
        *end_time_ptr = getTimeNs();
    }
}

template<nixl_gpu_level_t level>
nixl_status_t
launchPartialWriteTest(unsigned num_threads,
                       nixlGpuXferReqH req_hdnl_data,
                       size_t count,
                       const unsigned *indices,
                       const size_t *sizes,
                       void *const *addrs,
                       const uint64_t *remote_addrs,
                       uint64_t signal_inc,
                       uint64_t signal_remote_addr,
                       unsigned signal_index,
                       size_t num_iters,
                       bool is_no_delay,
                       bool use_xfer_status,
                       unsigned long long *start_time_ptr,
                       unsigned long long *end_time_ptr,
                       nixl_status_t *error_status) {
    testPartialWriteKernel<level><<<1, num_threads>>>(req_hdnl_data,
                                                      count,
                                                      indices,
                                                      sizes,
                                                      addrs,
                                                      remote_addrs,
                                                      signal_inc,
                                                      signal_remote_addr,
                                                      signal_index,
                                                      num_iters,
                                                      is_no_delay,
                                                      use_xfer_status,
                                                      start_time_ptr,
                                                      end_time_ptr,
                                                      error_status);

    return handleCudaErrors();
}

class PartialWriteTest : public DeviceApiTestBase {
public:
    // UCX GDAKI partial write doesn't support BLOCK level
    static const std::vector<nixl_gpu_level_t>
    getPartialWriteTestLevels() {
        static const std::vector<nixl_gpu_level_t> partialWriteLevels = {
            nixl_gpu_level_t::WARP,
            nixl_gpu_level_t::THREAD,
        };
        return partialWriteLevels;
    }
    void
    logGpuResults(size_t total_size,
                  size_t count,
                  size_t num_iters,
                  unsigned long long start_time_cpu,
                  unsigned long long end_time_cpu,
                  nixl_gpu_level_t level,
                  size_t num_threads) {
        double total_time_sec = (end_time_cpu - start_time_cpu) / static_cast<double>(NSEC_PER_SEC);
        double total_data = total_size * num_iters;
        auto bandwidth = total_data / total_time_sec / 1e9;
        const char *level_str = getGpuXferLevelStr(level);

        Logger() << "Partial Write Results [" << level_str << "]: " << count << " blocks, "
                 << total_size << " bytes total " << num_iters << " iterations, " << num_threads
                 << " threads in " << std::setprecision(4) << total_time_sec << " sec ("
                 << std::setprecision(2) << bandwidth << " GB/s)";
    }

protected:
    struct PartialWriteTestData {
        std::vector<MemBuffer> src_buffers;
        std::vector<MemBuffer> dst_buffers;
        nixlXferReqH *xfer_req_data;
        nixlGpuXferReqH gpu_req_hndl_data;
        size_t signal_size;
    };

    void
    initializePartialWriteTest(const std::vector<size_t> &sizes, nixl_mem_t mem_type, PartialWriteTestData &data) {
        size_t data_buf_count = sizes.size();

        for (size_t i = 0; i < data_buf_count; ++i) {
            data.src_buffers.emplace_back(sizes[i], mem_type);
            data.dst_buffers.emplace_back(sizes[i], mem_type);
        }

        nixl_opt_args_t signal_params = {.backends = {backend_handles[receiverAgent]}};
        nixl_status_t status =
            getAgent(receiverAgent).getGpuSignalSize(data.signal_size, &signal_params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "getGpuSignalSize failed";

        // Add a dummy signal buffer to the src buffer.
        // TODO: Remove this after implementing the new createGpuXferReq API
        data.src_buffers.emplace_back(data.signal_size, mem_type);
        data.dst_buffers.emplace_back(data.signal_size, mem_type);

        registerMem(getAgent(senderAgent), data.src_buffers, mem_type);
        registerMem(getAgent(receiverAgent), data.dst_buffers, mem_type);

        std::vector<MemBuffer> signal_only = {data.dst_buffers.back()};
        auto signal_desc_list = makeDescList<nixlBlobDesc>(signal_only, mem_type);
        status = getAgent(receiverAgent).prepGpuSignal(signal_desc_list, &signal_params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "prepGpuSignal failed";

        ASSERT_NO_FATAL_FAILURE(exchangeMD(senderAgent, receiverAgent));

        nixl_opt_args_t extra_params = {};
        extra_params.hasNotif = true;
        extra_params.notifMsg = notifMsg;
        extra_params.backends = {backend_handles[senderAgent]};

        // Create single transfer request that includes both data and signal buffers
        data.xfer_req_data = nullptr;
        status = getAgent(senderAgent)
                     .createXferReq(NIXL_WRITE,
                                   makeDescList<nixlBasicDesc>(data.src_buffers, mem_type),
                                   makeDescList<nixlBasicDesc>(data.dst_buffers, mem_type),
                                   getAgentName(receiverAgent),
                                   data.xfer_req_data,
                                   &extra_params);

        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create xfer request";
        ASSERT_NE(data.xfer_req_data, nullptr);

        status = getAgent(senderAgent).createGpuXferReq(*data.xfer_req_data, data.gpu_req_hndl_data);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create GPU xfer request";
        ASSERT_NE(data.gpu_req_hndl_data, nullptr)
            << "GPU request handle is null after createGpuXferReq";
    }

    void
    cleanupPartialWriteTest(const PartialWriteTestData &data) {
        getAgent(senderAgent).releaseGpuXferReq(data.gpu_req_hndl_data);
        nixl_status_t status = getAgent(senderAgent).releaseXferReq(data.xfer_req_data);
        ASSERT_EQ(status, NIXL_SUCCESS);
        invalidateMD();
    }

    void
    runPartialWriteTest(const PartialWriteTestData &setup_data,
                        const std::vector<size_t> &sizes,
                        size_t num_threads,
                        size_t num_iters,
                        bool is_no_delay,
                        bool use_xfer_status,
                        bool should_log_performance = true) {
        size_t data_buf_count = sizes.size();

        std::vector<unsigned> indices_host(data_buf_count);
        std::vector<void *> addrs_host(data_buf_count);
        std::vector<uint64_t> remote_addrs_host(data_buf_count);

        size_t total_size = 0;
        for (size_t i = 0; i < data_buf_count; ++i) {
            indices_host[i] = static_cast<unsigned>(i);
            addrs_host[i] = static_cast<void *>(setup_data.src_buffers[i]);
            remote_addrs_host[i] = static_cast<uintptr_t>(setup_data.dst_buffers[i]);
            total_size += sizes[i];
        }

        unsigned *indices_gpu = nullptr;
        size_t *sizes_gpu = nullptr;
        void **addrs_gpu = nullptr;
        uint64_t *remote_addrs_gpu = nullptr;

        cudaError_t cuda_err;
        cuda_err = cudaMalloc(&indices_gpu, data_buf_count * sizeof(unsigned));
        if (cuda_err != cudaSuccess) printf("Failed to allocate indices_gpu: %s\n", cudaGetErrorString(cuda_err));

        cuda_err = cudaMalloc(&sizes_gpu, data_buf_count * sizeof(size_t));
        if (cuda_err != cudaSuccess) printf("Failed to allocate sizes_gpu: %s\n", cudaGetErrorString(cuda_err));

        cuda_err = cudaMalloc(&addrs_gpu, data_buf_count * sizeof(void *));
        if (cuda_err != cudaSuccess) printf("Failed to allocate addrs_gpu: %s\n", cudaGetErrorString(cuda_err));

        cuda_err = cudaMalloc(&remote_addrs_gpu, data_buf_count * sizeof(uint64_t));
        if (cuda_err != cudaSuccess) printf("Failed to allocate remote_addrs_gpu: %s\n", cudaGetErrorString(cuda_err));

        cuda_err = cudaMemcpy(indices_gpu, indices_host.data(), data_buf_count * sizeof(unsigned), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) printf("Failed to copy indices: %s\n", cudaGetErrorString(cuda_err));

        cuda_err = cudaMemcpy(sizes_gpu, sizes.data(), data_buf_count * sizeof(size_t), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) printf("Failed to copy sizes: %s\n", cudaGetErrorString(cuda_err));

        cuda_err = cudaMemcpy(addrs_gpu, addrs_host.data(), data_buf_count * sizeof(void *), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) printf("Failed to copy addrs: %s\n", cudaGetErrorString(cuda_err));

        cuda_err = cudaMemcpy(remote_addrs_gpu, remote_addrs_host.data(), data_buf_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) printf("Failed to copy remote_addrs: %s\n", cudaGetErrorString(cuda_err));

        // The signal buffer is the last one in the memory list
        uint64_t signal_remote_addr = static_cast<uintptr_t>(setup_data.dst_buffers.back());
        unsigned signal_index = static_cast<unsigned>(setup_data.dst_buffers.size() - 1);
        uint64_t signal_inc = 42;

        unsigned long long *start_time_ptr = nullptr;
        unsigned long long *end_time_ptr = nullptr;
        nixl_status_t *error_status = nullptr;

        CudaPtr<unsigned long long> start_time_guard(&start_time_ptr);
        CudaPtr<unsigned long long> end_time_guard(&end_time_ptr);
        CudaPtr<nixl_status_t> error_guard(&error_status);

        nixl_status_t status = dispatchLaunchPartialWriteTest(GetParam(),
                                                              num_threads,
                                                              setup_data.gpu_req_hndl_data,
                                                              data_buf_count,
                                                              indices_gpu,
                                                              sizes_gpu,
                                                              addrs_gpu,
                                                              remote_addrs_gpu,
                                                              signal_inc,
                                                              signal_remote_addr,
                                                              signal_index,
                                                              num_iters,
                                                              is_no_delay,
                                                              use_xfer_status,
                                                              start_time_ptr,
                                                              end_time_ptr,
                                                              error_status);

        ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel launch failed";

        nixl_status_t kernel_error = NIXL_SUCCESS;
        cudaMemcpy(&kernel_error, error_status, sizeof(nixl_status_t), cudaMemcpyDeviceToHost);
        ASSERT_EQ(kernel_error, NIXL_SUCCESS) << "GPU kernel reported error: " << kernel_error;

        if (should_log_performance) {
            unsigned long long start_time_cpu = 0;
            unsigned long long end_time_cpu = 0;
            getTiming(start_time_ptr, end_time_ptr, start_time_cpu, end_time_cpu);
            logGpuResults(total_size,
                          data_buf_count,
                          num_iters,
                          start_time_cpu,
                          end_time_cpu,
                          GetParam(),
                          num_threads);
        }

        cudaFree(indices_gpu);
        cudaFree(sizes_gpu);
        cudaFree(addrs_gpu);
        cudaFree(remote_addrs_gpu);
    }

    nixl_status_t
    dispatchLaunchPartialWriteTest(nixl_gpu_level_t level,
                                   unsigned num_threads,
                                   nixlGpuXferReqH req_hdnl_data,
                                   size_t count,
                                   const unsigned *indices,
                                   const size_t *sizes,
                                   void *const *addrs,
                                   const uint64_t *remote_addrs,
                                   uint64_t signal_inc,
                                   uint64_t signal_remote_addr,
                                   unsigned signal_index,
                                   size_t num_iters,
                                   bool is_no_delay,
                                   bool use_xfer_status,
                                   unsigned long long *start_time_ptr,
                                   unsigned long long *end_time_ptr,
                                   nixl_status_t *error_status) {
        auto launcher = [=](auto level_tag) {
            constexpr auto L = level_tag.value;
            return launchPartialWriteTest<L>(num_threads,
                                             req_hdnl_data,
                                             count,
                                             indices,
                                             sizes,
                                             addrs,
                                             remote_addrs,
                                             signal_inc,
                                             signal_remote_addr,
                                             signal_index,
                                             num_iters,
                                             is_no_delay,
                                             use_xfer_status,
                                             start_time_ptr,
                                             end_time_ptr,
                                             error_status);
        };
        return dispatchKernelByLevel(level, launcher);
    }
};

TEST_P(PartialWriteTest, BasicPartialWriteTest) {
    std::vector<size_t> sizes(128, 1024);
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;
    const size_t num_iters = 10000;
    const bool is_no_delay = true;

    PartialWriteTestData setup_data;
    ASSERT_NO_FATAL_FAILURE(initializePartialWriteTest(sizes, mem_type, setup_data));

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::vector<uint8_t> pattern(sizes[i]);
        for (size_t j = 0; j < sizes[i]; ++j) {
            pattern[j] = static_cast<uint8_t>((i * 256 + j) % 256);
        }
        cudaMemcpy(static_cast<void *>(setup_data.src_buffers[i]),
                   pattern.data(),
                   sizes[i],
                   cudaMemcpyHostToDevice);
    }

    ASSERT_NO_FATAL_FAILURE(runPartialWriteTest(setup_data, sizes, num_threads, num_iters, is_no_delay, false));

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::vector<uint8_t> expected_pattern(sizes[i]);
        std::vector<uint8_t> received_data(sizes[i]);

        for (size_t j = 0; j < sizes[i]; ++j) {
            expected_pattern[j] = static_cast<uint8_t>((i * 256 + j) % 256);
        }

        cudaMemcpy(received_data.data(),
                   static_cast<void *>(setup_data.dst_buffers[i]),
                   sizes[i],
                   cudaMemcpyDeviceToHost);

        ASSERT_EQ(received_data, expected_pattern)
            << "Data verification failed for block " << i << " of size " << sizes[i];
    }

    cleanupPartialWriteTest(setup_data);
}
} // namespace gtest::nixl::gpu::partial_write

using gtest::nixl::gpu::partial_write::PartialWriteTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         PartialWriteTest,
                         testing::ValuesIn(PartialWriteTest::getPartialWriteTestLevels()),
                         [](const testing::TestParamInfo<nixl_gpu_level_t> &info) {
                             return std::string("UCX_") +
                                 DeviceApiTestBase::getGpuXferLevelStr(info.param);
                         });
