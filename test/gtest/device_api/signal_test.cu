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

namespace gtest::nixl::gpu::signal {

__device__ inline void
printSignalError(const char *operation, int thread_id, size_t iteration, nixl_status_t status) {
    printf("Thread %d: %s failed iteration %d: status=%d\n",
           thread_id,
           operation,
           (int)iteration, // cast to int because printf issue size_t
           status);
}

template<nixl_gpu_level_t level>
__global__ void
testSignalPostKernel(nixlGpuXferReqH req_hdnl,
                     unsigned index,
                     uint64_t signal_inc,
                     uint64_t remote_addr,
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
        nixlGpuSignal signal = {signal_inc, remote_addr};

        nixl_status_t status =
            nixlGpuPostSignalXferReq<level>(req_hdnl, index, signal, is_no_delay, xfer_status_ptr);
        if (status != NIXL_SUCCESS) {
            printSignalError("nixlGpuPostSignalXferReq", threadIdx.x, i, status);
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
__global__ void
testSignalReadKernel(const void *signal_addr,
                     uint64_t expected_signal_value,
                     uint64_t *result_ptr,
                     unsigned long long *start_time_ptr,
                     unsigned long long *end_time_ptr) {
    if (threadIdx.x == 0) {
        *start_time_ptr = getTimeNs();
    }

    __syncthreads();

    uint64_t signal_value;

    do {
        signal_value = nixlGpuReadSignal<level>(signal_addr);
    } while (signal_value != expected_signal_value);

    __syncthreads();

    if (threadIdx.x == 0) {
        *result_ptr = signal_value;
        *end_time_ptr = getTimeNs();
    }
}

template<nixl_gpu_level_t level>
nixl_status_t
launchSignalPostTest(unsigned num_threads,
                     nixlGpuXferReqH req_hdnl,
                     unsigned index,
                     uint64_t signal_inc,
                     uint64_t remote_addr,
                     size_t num_iters,
                     bool is_no_delay,
                     bool use_xfer_status,
                     unsigned long long *start_time_ptr,
                     unsigned long long *end_time_ptr,
                     nixl_status_t *error_status) {
    testSignalPostKernel<level><<<1, num_threads>>>(req_hdnl,
                                                    index,
                                                    signal_inc,
                                                    remote_addr,
                                                    num_iters,
                                                    is_no_delay,
                                                    use_xfer_status,
                                                    start_time_ptr,
                                                    end_time_ptr,
                                                    error_status);

    return handleCudaErrors();
}

template<nixl_gpu_level_t level>
nixl_status_t
launchSignalReadTest(unsigned num_threads,
                     const void *signal_addr,
                     uint64_t expected_signal_value,
                     uint64_t *result_ptr,
                     unsigned long long *start_time_ptr,
                     unsigned long long *end_time_ptr) {
    testSignalReadKernel<level><<<1, num_threads>>>(
        signal_addr, expected_signal_value, result_ptr, start_time_ptr, end_time_ptr);

    return handleCudaErrors();
}

class SignalTest : public DeviceApiTestBase {
public:
    void
    logSignalResults(size_t num_iters,
                     uint64_t signal_inc,
                     unsigned long long start_time_cpu,
                     unsigned long long end_time_cpu) const {
        double total_time_sec = (end_time_cpu - start_time_cpu) / static_cast<double>(NSEC_PER_SEC);
        Logger() << "Signal Results: " << num_iters << " iterations with increment " << signal_inc
                 << " in " << std::setprecision(4) << total_time_sec << " sec";
    }

protected:
    nixl_status_t
    dispatchLaunchSignalPostTest(nixl_gpu_level_t level,
                                 unsigned num_threads,
                                 nixlGpuXferReqH req_hdnl,
                                 unsigned index,
                                 uint64_t signal_inc,
                                 uint64_t remote_addr,
                                 size_t num_iters,
                                 bool is_no_delay,
                                 bool use_xfer_status,
                                 unsigned long long *start_time_ptr,
                                 unsigned long long *end_time_ptr,
                                 nixl_status_t *error_status) {
        auto launcher = [=](auto level_tag) {
            constexpr auto L = level_tag.value;
            return launchSignalPostTest<L>(num_threads,
                                           req_hdnl,
                                           index,
                                           signal_inc,
                                           remote_addr,
                                           num_iters,
                                           is_no_delay,
                                           use_xfer_status,
                                           start_time_ptr,
                                           end_time_ptr,
                                           error_status);
        };
        return dispatchKernelByLevel(level, launcher);
    }

    nixl_status_t
    dispatchLaunchSignalReadTest(nixl_gpu_level_t level,
                                 unsigned num_threads,
                                 const void *signal_addr,
                                 uint64_t expected_signal_value,
                                 uint64_t *result_ptr,
                                 unsigned long long *start_time_ptr,
                                 unsigned long long *end_time_ptr) {
        auto launcher = [=](auto level_tag) {
            constexpr auto L = level_tag.value;
            return launchSignalReadTest<L>(num_threads,
                                           signal_addr,
                                           expected_signal_value,
                                           result_ptr,
                                           start_time_ptr,
                                           end_time_ptr);
        };
        return dispatchKernelByLevel(level, launcher);
    }

    struct testSetupData {
        size_t signal_size;
        std::vector<MemBuffer> signal_buffers;
        std::vector<MemBuffer> src_buffers;
        nixlXferReqH *xfer_req;
        nixlGpuXferReqH gpu_req_hndl;
    };

    void
    initializeSignalTest(testSetupData &data) {
        nixl_opt_args_t extra_params = {.backends = {backend_handles[receiverAgent]}};
        nixl_status_t status =
            getAgent(receiverAgent).getGpuSignalSize(data.signal_size, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "getGpuSignalSize failed";

        createRegisteredMem(
            getAgent(receiverAgent), data.signal_size, 1, VRAM_SEG, data.signal_buffers);

        auto signal_desc_list = makeDescList<nixlBlobDesc>(data.signal_buffers, VRAM_SEG);
        status = getAgent(receiverAgent).prepGpuSignal(signal_desc_list, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "prepGpuSignal failed";

        createRegisteredMem(getAgent(senderAgent), data.signal_size, 1, VRAM_SEG, data.src_buffers);

        ASSERT_NO_FATAL_FAILURE(exchangeMD(senderAgent, receiverAgent));

        nixl_opt_args_t xfer_extra_params;
        xfer_extra_params.hasNotif = true;
        xfer_extra_params.notifMsg = notifMsg;
        xfer_extra_params.backends = {backend_handles[senderAgent]};

        data.xfer_req = nullptr;
        status = getAgent(senderAgent)
                     .createXferReq(NIXL_WRITE,
                                    makeDescList<nixlBasicDesc>(data.src_buffers, VRAM_SEG),
                                    makeDescList<nixlBasicDesc>(data.signal_buffers, VRAM_SEG),
                                    getAgentName(receiverAgent),
                                    data.xfer_req,
                                    &xfer_extra_params);

        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create xfer request";
        ASSERT_NE(data.xfer_req, nullptr);

        status = getAgent(senderAgent).createGpuXferReq(*data.xfer_req, data.gpu_req_hndl);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create GPU xfer request";
        ASSERT_NE(data.gpu_req_hndl, nullptr)
            << "GPU request handle is null after createGpuXferReq";
    }

    void
    cleanupSignalTest(const testSetupData &data) {
        getAgent(senderAgent).releaseGpuXferReq(data.gpu_req_hndl);
        nixl_status_t status = getAgent(senderAgent).releaseXferReq(data.xfer_req);
        ASSERT_EQ(status, NIXL_SUCCESS);
        invalidateMD();
    }

    void
    runSignalTest(testSetupData &setup_data,
                  size_t num_threads,
                  size_t num_iters,
                  unsigned index,
                  bool is_no_delay,
                  bool use_xfer_status,
                  uint64_t signal_inc,
                  uint64_t expected_signal_value) {
        uint64_t remote_addr = static_cast<uintptr_t>(setup_data.signal_buffers[0]);

        unsigned long long *start_time_ptr = nullptr;
        unsigned long long *end_time_ptr = nullptr;
        nixl_status_t *error_status = nullptr;
        CudaPtr<unsigned long long> start_time_guard(&start_time_ptr);
        CudaPtr<unsigned long long> end_time_guard(&end_time_ptr);
        CudaPtr<nixl_status_t> error_guard(&error_status);

        nixl_status_t status = dispatchLaunchSignalPostTest(GetParam(),
                                                            num_threads,
                                                            setup_data.gpu_req_hndl,
                                                            index,
                                                            signal_inc,
                                                            remote_addr,
                                                            num_iters,
                                                            is_no_delay,
                                                            use_xfer_status,
                                                            start_time_ptr,
                                                            end_time_ptr,
                                                            error_status);

        ASSERT_EQ(status, NIXL_SUCCESS)
            << "Signal post kernel launch failed with status: " << status;

        nixl_status_t kernel_error = NIXL_SUCCESS;
        cudaMemcpy(&kernel_error, error_status, sizeof(nixl_status_t), cudaMemcpyDeviceToHost);
        ASSERT_EQ(kernel_error, NIXL_SUCCESS) << "GPU kernel reported error: " << kernel_error;

        unsigned long long start_time_cpu = 0;
        unsigned long long end_time_cpu = 0;
        getTiming(start_time_ptr, end_time_ptr, start_time_cpu, end_time_cpu);
        logSignalResults(num_iters, signal_inc, start_time_cpu, end_time_cpu);

        uint64_t *result_ptr = nullptr;
        CudaPtr<uint64_t> result_guard(&result_ptr);

        unsigned long long *read_start_time_ptr = nullptr;
        unsigned long long *read_end_time_ptr = nullptr;
        CudaPtr<unsigned long long> read_start_time_guard(&read_start_time_ptr);
        CudaPtr<unsigned long long> read_end_time_guard(&read_end_time_ptr);

        status =
            dispatchLaunchSignalReadTest(GetParam(),
                                         num_threads,
                                         static_cast<const void *>(setup_data.signal_buffers[0]),
                                         expected_signal_value,
                                         result_ptr,
                                         read_start_time_ptr,
                                         read_end_time_ptr);

        ASSERT_EQ(status, NIXL_SUCCESS)
            << "Signal read kernel launch failed with status: " << status;

        uint64_t signal_value = 0;
        cudaMemcpy(&signal_value, result_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        ASSERT_EQ(signal_value, expected_signal_value)
            << "Signal value mismatch. Expected: " << expected_signal_value
            << ", Got: " << signal_value;
    }

private:
};

TEST_P(SignalTest, BasicSignalTest) {
#ifndef HAVE_UCX_GPU_DEVICE_API
    GTEST_SKIP() << "UCX GPU device API not available, skipping test";
#else
    testSetupData setup_data;
    ASSERT_NO_FATAL_FAILURE(initializeSignalTest(setup_data));

    constexpr size_t num_threads = 1;
    constexpr size_t num_iters = 1;
    constexpr unsigned index = 0;
    constexpr bool is_no_delay = true;
    constexpr uint64_t signal_inc = 42;
    uint64_t expected_value = signal_inc * num_iters;

    ASSERT_NO_FATAL_FAILURE(runSignalTest(
        setup_data, num_threads, num_iters, index, is_no_delay, true, signal_inc, expected_value));

    ASSERT_NO_FATAL_FAILURE(cleanupSignalTest(setup_data));
#endif
}

TEST_P(SignalTest, StressSignalTest) {
#ifndef HAVE_UCX_GPU_DEVICE_API
    GTEST_SKIP() << "UCX GPU device API not available, skipping test";
#else
    srand(time(nullptr));

    testSetupData setup_data;
    ASSERT_NO_FATAL_FAILURE(initializeSignalTest(setup_data));

    constexpr size_t num_threads = 1;
    constexpr size_t num_iters = 10000;
    constexpr unsigned index = 0;
    constexpr bool is_no_delay = true;
    uint64_t signal_inc = (rand() % 100) + 1;
    uint64_t expected_value = signal_inc * num_iters;

    ASSERT_NO_FATAL_FAILURE(runSignalTest(
        setup_data, num_threads, num_iters, index, is_no_delay, true, signal_inc, expected_value));

    ASSERT_NO_FATAL_FAILURE(cleanupSignalTest(setup_data));
#endif
}

TEST_P(SignalTest, NoStatusTrackingTest) {
#ifndef HAVE_UCX_GPU_DEVICE_API
    GTEST_SKIP() << "UCX GPU device API not available, skipping test";
#else
    testSetupData setup_data;
    ASSERT_NO_FATAL_FAILURE(initializeSignalTest(setup_data));

    constexpr size_t num_threads = 1; // UCX only supports 1 thread for no status tracking
    constexpr size_t num_iters = 10000;
    constexpr unsigned index = 0;
    constexpr bool is_no_delay = true;
    constexpr bool use_xfer_status = false;
    constexpr uint64_t signal_inc = 10;
    uint64_t expected_value = signal_inc * num_iters;

    ASSERT_NO_FATAL_FAILURE(runSignalTest(setup_data,
                            num_threads,
                            num_iters,
                            index,
                            is_no_delay,
                            use_xfer_status,
                            signal_inc,
                            expected_value));

    ASSERT_NO_FATAL_FAILURE(cleanupSignalTest(setup_data));
#endif
}

} // namespace gtest::nixl::gpu::signal

using gtest::nixl::gpu::signal::SignalTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         SignalTest,
                         testing::ValuesIn(DeviceApiTestBase::getTestLevels()),
                         [](const testing::TestParamInfo<nixl_gpu_level_t> &info) {
                             return std::string("UCX_") +
                                 DeviceApiTestBase::getGpuXferLevelStr(info.param);
                         });
