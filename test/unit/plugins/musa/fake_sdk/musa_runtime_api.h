// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Test double only. Does not certify the proprietary MUSA SDK ABI.
#ifndef NIXL_TEST_UNIT_PLUGINS_MUSA_FAKE_SDK_MUSA_RUNTIME_API_H
#define NIXL_TEST_UNIT_PLUGINS_MUSA_FAKE_SDK_MUSA_RUNTIME_API_H
#include <cstddef>

enum musaError_t { musaSuccess = 0, musaErrorInvalidValue = 1, musaErrorUnknown = 7 };

enum musaMemoryType { musaMemoryTypeHost, musaMemoryTypeDevice, musaMemoryTypeManaged };

struct musaPointerAttributes {
    musaMemoryType type;
    int device;
};

extern "C" {
musaError_t
musaGetDeviceCount(int *);
musaError_t
musaPointerGetAttributes(musaPointerAttributes *, const void *);
const char *musaGetErrorString(musaError_t);

// Declarations used only to syntax-check the hardware harness, not to emulate GPU transfers.
enum musaMemcpyKind { musaMemcpyHostToDevice, musaMemcpyDeviceToHost };

musaError_t
musaSetDevice(int);
musaError_t
musaDeviceSynchronize();
musaError_t
musaMalloc(void **, size_t);
musaError_t
musaFree(void *);
musaError_t
musaMemcpy(void *, const void *, size_t, musaMemcpyKind);
}

extern int fake_device_count;
extern musaError_t fake_status;
extern musaPointerAttributes fake_attributes;

#endif
