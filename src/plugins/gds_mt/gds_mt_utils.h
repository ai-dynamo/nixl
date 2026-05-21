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
#ifndef __GDS_MT_UTILS_H
#define __GDS_MT_UTILS_H

#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>
#include <cufile.h>

#include "file/file_path_mode.h"

// RAII wrapper: ctor cuFileHandleRegister, dtor cuFileHandleDeregister.
// fd ownership lives in the nixl::fdHandle base.
class gdsMtFileHandle : public nixl::fdHandle {
public:
    explicit gdsMtFileHandle(nixl::fdHandle &&h);
    ~gdsMtFileHandle() override;

    CUfileHandle_t cu_fhandle{nullptr};
};

class gdsMtMemBuf {
public:
    gdsMtMemBuf (void *ptr, size_t sz, int flags = 0);
    ~gdsMtMemBuf();

    gdsMtMemBuf (const gdsMtMemBuf &) = delete;
    gdsMtMemBuf &
    operator= (const gdsMtMemBuf &) = delete;
    gdsMtMemBuf (gdsMtMemBuf &&) = delete;
    gdsMtMemBuf &
    operator= (gdsMtMemBuf &&) = delete;

private:
    void *base_{nullptr};
    bool registered_{false};
};

class gdsMtUtil {
public:
    gdsMtUtil();
    ~gdsMtUtil();
};
#endif
