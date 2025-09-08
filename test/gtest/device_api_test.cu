#include <gtest/gtest.h>
#include "nixl.h"

#include <cuda_runtime.h>
#include <nixl_device.cuh>

namespace gtest::nixl::gpu {

class ucxDeviceApi : public ::testing::Test {};

template<nixl_gpu_level_t level>
__global__ void dummyKernel() {
    const nixlGpuSignal signal { 1, 0x1000 };
    nixlGpuXferStatusH status;

    [[maybe_unused]] auto result1 = nixlGpuPostSingleWriteXferReq<level>(nullptr, 0, nullptr, 0, 0);
    [[maybe_unused]] auto result2 = nixlGpuPostSignalXferReq<level>(nullptr, 0, signal);
    [[maybe_unused]] auto result3 = nixlGpuPostPartialWriteXferReq<level>(nullptr, 0, nullptr, nullptr, nullptr, nullptr, signal);
    [[maybe_unused]] auto result4 = nixlGpuPostWriteXferReq<level>(nullptr, nullptr, nullptr, nullptr, signal);
    [[maybe_unused]] auto result5 = nixlGpuGetXferStatus<level>(status);
    [[maybe_unused]] auto result6 = nixlGpuReadSignal<level>(nullptr);
}

TEST_F(ucxDeviceApi, compilationTest) {
    dummyKernel<nixl_gpu_level_t::THREAD><<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

} // namespace gtest::nixl::gpu
