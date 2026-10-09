// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Test double only. Does not certify the proprietary MUSA SDK ABI.
#pragma once
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
