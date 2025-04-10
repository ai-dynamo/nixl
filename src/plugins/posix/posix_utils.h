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

#ifndef POSIX_UTILS_H
#define POSIX_UTILS_H

#include <string>
#include <unistd.h>
#include "posix_backend.h"
#define POSIX_MEM_BUF_SIZE 4096

struct posixFileHandle : public nixlBackendMD {
    int         fd;
    size_t      size;
    std::string metadata;
};

// Utility class for POSIX operations
class posixUtil {
public:
    posixUtil() {}
    ~posixUtil() {}
    
    // Register a file handle
    nixl_status_t fillHandle(int fd, size_t size,
                                    std::string metaInfo,
                                    posixFileHandle& handle);
                                    
    unsigned int getWorkId(nixlPosixBackendReqH& handle, unsigned int idx);
};

#endif // POSIX_UTILS_H 