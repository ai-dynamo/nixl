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

#include <cassert>
#include <iostream>
#include "file/file_utils.h"

int main() {
    const std::string test_file = "/tmp/nixl_test_file";
    int fd;
    nixl_status_t status;

    // Test file creation
    status = nixlFileUtils::openFile(test_file, nixlFileMode::CREATE, fd);
    assert(status == NIXL_SUCCESS);
    
    // Test file close
    status = nixlFileUtils::closeFile(fd);
    assert(status == NIXL_SUCCESS);

    // Test file exists
    status = nixlFileUtils::fileExists(test_file);
    assert(status == NIXL_SUCCESS);

    // Test file unlink
    status = nixlFileUtils::unlinkFile(test_file);
    assert(status == NIXL_SUCCESS);

    // Verify file no longer exists
    status = nixlFileUtils::fileExists(test_file);
    assert(status == NIXL_ERR_NOT_FOUND);

    std::cout << "All file utils tests passed!" << std::endl;
    return 0;
} 