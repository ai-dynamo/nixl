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

namespace gtest::nixl::gpu::single_write {

__device__ inline void
printSingleWriteError(const char *operation, int thread_id, size_t iteration, nixl_status_t status) {
    printf("Thread %d: %s failed iteration %d: status=%d\n",
           thread_id,
           operation,
           (int)iteration, // cast to int because printf issue size_t
           status);
}

template<nixl_gpu_level_t level>
__global__ void
testSingleWriteKernel(nixlGpuXferReqH req_hdnl,
                      unsigned index,
                      const void *src_addr,
                      uint64_t remote_addr,
                      size_t size,
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
        nixl_status_t status = nixlGpuPostSingleWriteXferReq<level>(
            req_hdnl, index, src_addr, remote_addr, size, is_no_delay, xfer_status_ptr);
        if (status != NIXL_SUCCESS) {
            printSingleWriteError("nixlGpuPostSingleWriteXferReq", threadIdx.x, i, status);
            *error_status = status;
            return;
        }

        if (use_xfer_status) {
            status = nixlGpuGetXferStatus<level>(*xfer_status_ptr);
            if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                printProgressError(threadIdx.x, i, status);
                *error_status = status;
                return;
            }

            while (status == NIXL_IN_PROG) {
                status = nixlGpuGetXferStatus<level>(*xfer_status_ptr);
                if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                    printProgressError(threadIdx.x, i, status);
                    *error_status = status;
                    return;
                }
            }

            if (status != NIXL_SUCCESS) {
                printCompletionError(threadIdx.x, i, status);
                *error_status = status;
                return;
            }
        }
    }

    if (threadIdx.x == 0) {
        *end_time_ptr = getTimeNs();
    }
}

template<nixl_gpu_level_t level>
nixl_status_t
launchSingleWriteTest(unsigned num_threads,
                      nixlGpuXferReqH req_hdnl,
                      unsigned index,
                      const void *src_addr,
                      uint64_t remote_addr,
                      size_t size,
                      size_t num_iters,
                      bool is_no_delay,
                      bool use_xfer_status,
                      unsigned long long *start_time_ptr,
                      unsigned long long *end_time_ptr,
                      nixl_status_t *error_status) {
    testSingleWriteKernel<level><<<1, num_threads>>>(req_hdnl,
                                                     index,
                                                     src_addr,
                                                     remote_addr,
                                                     size,
                                                     num_iters,
                                                     is_no_delay,
                                                     use_xfer_status,
                                                     start_time_ptr,
                                                     end_time_ptr,
                                                     error_status);

    return handleCudaErrors();
}

class SingleWriteTest : public DeviceApiTestBase {
public:
    void
    logGpuResults(size_t size,
                  size_t count,
                  size_t num_iters,
                  unsigned long long start_time_cpu,
                  unsigned long long end_time_cpu,
                  nixl_gpu_level_t level,
                  size_t num_threads) {
        double total_time_sec = (end_time_cpu - start_time_cpu) / static_cast<double>(NSEC_PER_SEC);
        double total_size = size * count * num_iters;

        auto bandwidth = total_size / total_time_sec / 1e9;

        const char *level_str = getGpuXferLevelStr(level);

        Logger() << "Single Write Results [" << level_str << "]: " << size << "x" << count << "x"
                 << num_iters << ", " << num_threads << " threads in " << std::setprecision(4)
                 << total_time_sec << " sec (" << std::setprecision(2) << bandwidth << " GB/s)";
    }

protected:
    struct SingleWriteTestData {
        std::vector<MemBuffer> src_buffers;
        std::vector<MemBuffer> dst_buffers;
        nixlXferReqH *xfer_req;
        nixlGpuXferReqH gpu_req_hndl;
    };

    void
    initializeSingleWriteTest(size_t size, size_t count, nixl_mem_t mem_type, SingleWriteTestData &data) {
        createRegisteredMem(getAgent(senderAgent), size, count, mem_type, data.src_buffers);
        createRegisteredMem(getAgent(receiverAgent), size, count, mem_type, data.dst_buffers);

        exchangeMD(senderAgent, receiverAgent);

        nixl_opt_args_t extra_params = {};
        extra_params.hasNotif = true;
        extra_params.notifMsg = notifMsg;
        extra_params.backends = {backend_handles[senderAgent]};

        data.xfer_req = nullptr;
        nixl_status_t status =
            getAgent(senderAgent)
                .createXferReq(NIXL_WRITE,
                               makeDescList<nixlBasicDesc>(data.src_buffers, mem_type),
                               makeDescList<nixlBasicDesc>(data.dst_buffers, mem_type),
                               getAgentName(receiverAgent),
                               data.xfer_req,
                               &extra_params);

        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create xfer request";
        ASSERT_NE(data.xfer_req, nullptr);

        status = getAgent(senderAgent).createGpuXferReq(*data.xfer_req, data.gpu_req_hndl);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create GPU xfer request";
        ASSERT_NE(data.gpu_req_hndl, nullptr)
            << "GPU request handle is null after createGpuXferReq";
    }

    void
    cleanupSingleWriteTest(const SingleWriteTestData &data) {
        getAgent(senderAgent).releaseGpuXferReq(data.gpu_req_hndl);
        nixl_status_t status = getAgent(senderAgent).releaseXferReq(data.xfer_req);
        ASSERT_EQ(status, NIXL_SUCCESS);
        invalidateMD();
    }

    void
    runSingleWriteTest(const SingleWriteTestData &setup_data,
                       size_t size,
                       size_t count,
                       size_t num_threads,
                       size_t num_iters,
                       unsigned index,
                       bool is_no_delay,
                       bool use_xfer_status,
                       bool should_log_performance = true) {
        uint64_t remote_addr = static_cast<uintptr_t>(setup_data.dst_buffers[0]);
        const void *src_addr = static_cast<const void *>(setup_data.src_buffers[0]);

        unsigned long long *start_time_ptr = nullptr;
        unsigned long long *end_time_ptr = nullptr;
        nixl_status_t *result_status = nullptr;

        CudaPtr<unsigned long long> start_time_guard(&start_time_ptr);
        CudaPtr<unsigned long long> end_time_guard(&end_time_ptr);
        CudaPtr<nixl_status_t> result_guard(&result_status);

        nixl_status_t *error_status = nullptr;
        CudaPtr<nixl_status_t> error_guard(&error_status);

        nixl_status_t status = dispatchLaunchSingleWriteTest(GetParam(),
                                                             num_threads,
                                                             setup_data.gpu_req_hndl,
                                                             index,
                                                             src_addr,
                                                             remote_addr,
                                                             size,
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

        nixl_status_t gpu_result;
        cudaMemcpy(&gpu_result, result_status, sizeof(nixl_status_t), cudaMemcpyDeviceToHost);
        ASSERT_EQ(gpu_result, NIXL_SUCCESS) << "GPU kernel reported error: " << gpu_result;

        if (should_log_performance) {
            unsigned long long start_time_cpu = 0;
            unsigned long long end_time_cpu = 0;
            getTiming(start_time_ptr, end_time_ptr, start_time_cpu, end_time_cpu);
            logGpuResults(
                size, count, num_iters, start_time_cpu, end_time_cpu, GetParam(), num_threads);
        }
    }

    nixl_status_t
    dispatchLaunchSingleWriteTest(nixl_gpu_level_t level,
                                  unsigned num_threads,
                                  nixlGpuXferReqH req_hdnl,
                                  unsigned index,
                                  const void *src_addr,
                                  uint64_t remote_addr,
                                  size_t size,
                                  size_t num_iters,
                                  bool is_no_delay,
                                  bool use_xfer_status,
                                  unsigned long long *start_time_ptr,
                                  unsigned long long *end_time_ptr,
                                  nixl_status_t *error_status) {
        auto launcher = [=](auto level_tag) {
            constexpr auto L = level_tag.value;
            return launchSingleWriteTest<L>(num_threads,
                                            req_hdnl,
                                            index,
                                            src_addr,
                                            remote_addr,
                                            size,
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

TEST_P(SingleWriteTest, BasicSingleWriteTest) {
    constexpr size_t size = 4 * 1024;
    constexpr size_t count = 128;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;
    const size_t num_iters = 10000;
    constexpr unsigned index = 0;
    const bool is_no_delay = true;

    SingleWriteTestData setup_data;
    ASSERT_NO_FATAL_FAILURE(initializeSingleWriteTest(size, count, mem_type, setup_data));

    uint32_t *src_data = static_cast<uint32_t *>(static_cast<void *>(setup_data.src_buffers[0]));
    uint32_t pattern = 0xDEADBEEF;
    cudaMemset(src_data, 0, size);
    cudaMemcpy(src_data, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);

    ASSERT_NO_FATAL_FAILURE(runSingleWriteTest(setup_data, size, count, num_threads, num_iters, index, is_no_delay, true));

    uint32_t dst_data;
    cudaMemcpy(&dst_data,
               static_cast<uint32_t *>(static_cast<void *>(setup_data.dst_buffers[0])),
               sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    ASSERT_EQ(dst_data, pattern) << "Data transfer verification failed. Expected: 0x" << std::hex
                                 << pattern << ", Got: 0x" << dst_data;

    ASSERT_NO_FATAL_FAILURE(cleanupSingleWriteTest(setup_data));
}

TEST_P(SingleWriteTest, MultipleSizesTest) {
    std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384};

    for (size_t size : sizes) {
        constexpr size_t count = 128;
        nixl_mem_t mem_type = VRAM_SEG;
        size_t num_threads = 32;
        const size_t num_iters = 50000;
        constexpr unsigned index = 0;
        const bool is_no_delay = true;

        SingleWriteTestData setup_data;
        ASSERT_NO_FATAL_FAILURE(initializeSingleWriteTest(size, count, mem_type, setup_data));

        std::vector<uint8_t> pattern(size);
        for (size_t i = 0; i < size; ++i) {
            pattern[i] = static_cast<uint8_t>(i % 256);
        }
        cudaMemcpy(static_cast<void *>(setup_data.src_buffers[0]),
                   pattern.data(),
                   size,
                   cudaMemcpyHostToDevice);

        ASSERT_NO_FATAL_FAILURE(runSingleWriteTest(
            setup_data, size, count, num_threads, num_iters, index, is_no_delay, true, false));

        std::vector<uint8_t> received_data(size);
        cudaMemcpy(received_data.data(),
                   static_cast<void *>(setup_data.dst_buffers[0]),
                   size,
                   cudaMemcpyDeviceToHost);

        ASSERT_EQ(received_data, pattern) << "Data verification failed for size " << size;

        ASSERT_NO_FATAL_FAILURE(cleanupSingleWriteTest(setup_data));
    }
}

TEST_P(SingleWriteTest, NoStatusTrackingTest) {
    constexpr size_t size = 4 * 1024;
    constexpr size_t count = 128;
    nixl_mem_t mem_type = VRAM_SEG;
    constexpr size_t num_threads = 1; // UCX only supports 1 thread for no status tracking
    const size_t num_iters = 1000;
    constexpr unsigned index = 0;
    const bool is_no_delay = true;
    const bool use_xfer_status = false;

    SingleWriteTestData setup_data;
    ASSERT_NO_FATAL_FAILURE(initializeSingleWriteTest(size, count, mem_type, setup_data));

    uint32_t *src_data = static_cast<uint32_t *>(static_cast<void *>(setup_data.src_buffers[0]));
    uint32_t pattern = 0xCAFEBABE;
    cudaMemset(src_data, 0, size);
    cudaMemcpy(src_data, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);

    ASSERT_NO_FATAL_FAILURE(runSingleWriteTest(
        setup_data, size, count, num_threads, num_iters, index, is_no_delay, use_xfer_status));

    uint32_t dst_data;
    cudaMemcpy(&dst_data,
               static_cast<uint32_t *>(static_cast<void *>(setup_data.dst_buffers[0])),
               sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    ASSERT_EQ(dst_data, pattern) << "Data transfer verification failed. Expected: 0x" << std::hex
                                 << pattern << ", Got: 0x" << dst_data
                                 << " (This verifies transfer completion when xfer_status=nullptr)";

    ASSERT_NO_FATAL_FAILURE(cleanupSingleWriteTest(setup_data));
}

} // namespace gtest::nixl::gpu::single_write

using gtest::nixl::gpu::single_write::SingleWriteTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         SingleWriteTest,
                         testing::ValuesIn(DeviceApiTestBase::getTestLevels()),
                         [](const testing::TestParamInfo<nixl_gpu_level_t> &info) {
                             return std::string("UCX_") +
                                 DeviceApiTestBase::getGpuXferLevelStr(info.param);
                         });
