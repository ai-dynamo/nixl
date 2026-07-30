/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef NIXL_SRC_PLUGINS_ROCM_AIS_AIS_UTILS_H
#define NIXL_SRC_PLUGINS_ROCM_AIS_AIS_UTILS_H

#include <cstddef>

#include <fcntl.h>
#include <unistd.h>

#include <hipfile.h>

#include <nixl.h>

#include "file/file_path_mode.h"

// RAII wrappers around the hipFile resources shared by the AIS engines. Each
// type registers in its constructor and deregisters in its destructor, so the
// engines never hand-manage hipFile lifetimes.

// Owns the registered hipFile handle together with the nixl::FileFd it was
// built from. FileFd covers both registration modes: it closes the fd on
// destruction when path-mode opened it, and leaves a caller-supplied
// fd-in-devId alone.
class aisFileHandle {
public:
    explicit aisFileHandle(nixl::FileFd &&fd);
    ~aisFileHandle();

    aisFileHandle(const aisFileHandle &) = delete;
    aisFileHandle &
    operator=(const aisFileHandle &) = delete;
    aisFileHandle(aisFileHandle &&) = delete;
    aisFileHandle &
    operator=(aisFileHandle &&) = delete;

    nixl::FileFd file_fd;
    hipFileHandle_t hip_fhandle{nullptr};
};

class aisMemBuf {
public:
    aisMemBuf(void *ptr, size_t sz, int flags = 0);
    ~aisMemBuf();

    aisMemBuf(const aisMemBuf &) = delete;
    aisMemBuf &
    operator=(const aisMemBuf &) = delete;
    aisMemBuf(aisMemBuf &&) = delete;
    aisMemBuf &
    operator=(aisMemBuf &&) = delete;

private:
    void *base_{nullptr};
    bool registered_{false};
};

class aisDriverHandle {
public:
    aisDriverHandle();
    ~aisDriverHandle();

    aisDriverHandle(const aisDriverHandle &) = delete;
    aisDriverHandle &
    operator=(const aisDriverHandle &) = delete;
};

#endif // NIXL_SRC_PLUGINS_ROCM_AIS_AIS_UTILS_H
