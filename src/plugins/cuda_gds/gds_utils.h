/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_PLUGINS_CUDA_GDS_GDS_UTILS_H
#define NIXL_SRC_PLUGINS_CUDA_GDS_GDS_UTILS_H

#include <fcntl.h>
#include <unistd.h>

#include <nixl.h>
#include <cufile.h>

#include "file/file_path_mode.h"

// RAII cuFile file handle. Registers a CUfileHandle_t for the owned fd on
// construction and deregisters it on destruction. Shared (via shared_ptr) so a
// single underlying handle is refcounted across multiple registrations of the
// same fd.
class gdsFileHandle {
public:
    explicit gdsFileHandle(nixl::FileFd &&fd);
    ~gdsFileHandle();

    gdsFileHandle(const gdsFileHandle &) = delete;
    gdsFileHandle &
    operator=(const gdsFileHandle &) = delete;
    gdsFileHandle(gdsFileHandle &&) = delete;
    gdsFileHandle &
    operator=(gdsFileHandle &&) = delete;

    nixl::FileFd file_fd;
    CUfileHandle_t cu_fhandle{nullptr};
};

// RAII cuFile buffer registration. Registration failure is non-fatal (falls
// back to compat mode); in that case the buffer is not deregistered.
class gdsMemBuf {
public:
    gdsMemBuf(void *ptr, size_t sz, int flags = 0);
    ~gdsMemBuf();

    gdsMemBuf(const gdsMemBuf &) = delete;
    gdsMemBuf &
    operator=(const gdsMemBuf &) = delete;
    gdsMemBuf(gdsMemBuf &&) = delete;
    gdsMemBuf &
    operator=(gdsMemBuf &&) = delete;

private:
    void *base_{nullptr};
    bool registered_{false};
};

// RAII cuFile driver handle. Opens the driver on construction (throws on
// failure) and closes it on destruction.
class gdsDriverHandle {
public:
    gdsDriverHandle();
    ~gdsDriverHandle();

    gdsDriverHandle(const gdsDriverHandle &) = delete;
    gdsDriverHandle &
    operator=(const gdsDriverHandle &) = delete;
    gdsDriverHandle(gdsDriverHandle &&) = delete;
    gdsDriverHandle &
    operator=(gdsDriverHandle &&) = delete;
};

#endif // NIXL_SRC_PLUGINS_CUDA_GDS_GDS_UTILS_H
