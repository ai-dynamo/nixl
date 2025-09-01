#include <gtest/gtest.h>
#include <tuple>
#include "nixl.h"
#include "common.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAVE_UCX_GPU_DEVICE_API
#include <gpu/ucx/nixl_device.cuh>

class GpuDeviceApi : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

template<nixl_gpu_level_t level>
__global__ void nixl_gpu_api_compilation_test_kernel() {
    const nixlGpuSignal signal { 1, 0x1000 };
    nixlGpuXferStatusH status;

    [[maybe_unused]] auto result1 = nixlGpuPostSingleWriteXferReq<level>(nullptr, 0, nullptr, 0, 0);
    [[maybe_unused]] auto result2 = nixlGpuPostSignalXferReq<level>(nullptr, 0, signal);
    [[maybe_unused]] auto result3 = nixlGpuPostPartialWriteXferReq<level>(nullptr, 0, nullptr, nullptr, nullptr, nullptr, signal);
    [[maybe_unused]] auto result4 = nixlGpuPostWriteXferReq<level>(nullptr, nullptr, nullptr, nullptr, signal);
    [[maybe_unused]] auto result5 = nixlGpuGetXferStatus<level>(status);
    [[maybe_unused]] auto result6 = nixlGpuReadSignalValue<level>(nullptr);
}

TEST_F(GpuDeviceApi, NixlGpuApiCompilationTest) {
    nixl_gpu_api_compilation_test_kernel<nixl_gpu_level_t::THREAD><<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

#endif
