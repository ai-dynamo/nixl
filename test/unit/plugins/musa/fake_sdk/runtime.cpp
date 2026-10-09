// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "musa_runtime_api.h"

int fake_device_count = 2;
musaError_t fake_status = musaSuccess;
musaPointerAttributes fake_attributes{musaMemoryTypeDevice, 1};

extern "C" musaError_t
musaGetDeviceCount(int *count) {
    *count = fake_device_count;
    return fake_status;
}

extern "C" musaError_t
musaPointerGetAttributes(musaPointerAttributes *attributes, const void *) {
    *attributes = fake_attributes;
    return fake_status;
}

extern "C" const char *
musaGetErrorString(musaError_t) {
    return "injected SDK error";
}
