/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_cuda_device_guard.h"
#include <cuda.h>
#include <array>
#include <cstdio>
#include <thread>

namespace {
bool
check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

bool
allocation(int device) {
    CUdeviceptr ptr = 0;
    if (!check(cuMemAlloc(&ptr, 4096) == CUDA_SUCCESS, "guarded driver allocation")) {
        return false;
    }
    int owner = -1;
    std::array<unsigned char, 4096> source{}, destination{};
    source.fill(0x5a);
    bool ok = check(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr) ==
                            CUDA_SUCCESS &&
                        owner == device,
                    "allocation uses engine GPU");
    ok =
        check(cuMemcpyHtoD(ptr, source.data(), source.size()) == CUDA_SUCCESS, "copy to GPU") && ok;
    ok = check(cuMemcpyDtoH(destination.data(), ptr, destination.size()) == CUDA_SUCCESS,
               "copy from GPU") &&
        ok;
    ok = check(source == destination, "payload round trip") && ok;
    ok = check(cuMemFree(ptr) == CUDA_SUCCESS, "free allocation") && ok;
    return ok;
}

bool
coldThread(int engine_device) {
    CUcontext context = nullptr;
    if (!check(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context == nullptr,
               "new listener has no current context")) {
        return false;
    }
    CUdeviceptr ptr = 0;
    CUresult unbound = cuMemAlloc(&ptr, 4096);
    if (unbound == CUDA_SUCCESS) {
        cuMemFree(ptr);
    }
    if (!check(unbound == CUDA_ERROR_INVALID_CONTEXT, "unbound allocation reproduces CUDA 201")) {
        return false;
    }
    {
        nixl::doca::cudaDeviceGuard guard(engine_device);
        if (!check(guard.status() == cudaSuccess, "select engine GPU")) {
            return false;
        }
        if (!allocation(engine_device)) {
            return false;
        }
    }
    int restored = -1;
    return check(cudaGetDevice(&restored) == cudaSuccess && restored == 0,
                 "restore default runtime device");
}

bool
warmThread(int engine_device) {
    if (!check(cudaSetDevice(0) == cudaSuccess, "initialize caller device")) {
        return false;
    }
    {
        nixl::doca::cudaDeviceGuard guard(engine_device);
        if (!check(guard.status() == cudaSuccess, "switch from caller to engine")) {
            return false;
        }
        if (!allocation(engine_device)) {
            return false;
        }
    }
    int restored = -1;
    return check(cudaGetDevice(&restored) == cudaSuccess && restored == 0,
                 "restore caller runtime device");
}
} // namespace

int
main() {
    CUresult init = cuInit(0);
    if (init == CUDA_ERROR_NO_DEVICE) {
        return 77;
    }
    if (!check(init == CUDA_SUCCESS, "cuInit")) {
        return 1;
    }
    int count = 0;
    if (!check(cuDeviceGetCount(&count) == CUDA_SUCCESS, "device count")) {
        return 1;
    }
    if (count == 0) {
        return 77;
    }
    bool ok = true;
    // Test default-device initialization and, when available, non-default GPU routing.
    for (int device = 0; device < (count > 1 ? 2 : 1); ++device) {
        std::thread cold([&] { ok = coldThread(device) && ok; });
        cold.join();
        std::thread warm([&] { ok = warmThread(device) && ok; });
        warm.join();
    }
    // Use an ordinal outside the visible range: selection failure must be observable.
    {
        nixl::doca::cudaDeviceGuard invalid(static_cast<uint32_t>(count));
        ok = check(invalid.status() == cudaErrorInvalidDevice, "invalid device rejected") && ok;
    }
    if (!ok) {
        return 1;
    }
    std::puts(
        "PASS: CUDA 201 reproduced; guarded allocation and runtime-device restoration passed");
    return 0;
}
