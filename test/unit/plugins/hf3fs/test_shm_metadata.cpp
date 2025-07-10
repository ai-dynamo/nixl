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

#include <iostream>
#include <cassert>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "hf3fs_backend.h"
#include "hf3fs_utils.h"

int main() {
    std::cout << "Testing nixlHf3fsShmMetadata constructor with IOV wrapper..." << std::endl;
    
    const size_t test_size = 4096; // 4KB
    const char test_data[] = "Hello, POSIX Shared Memory with IOV!";
    const size_t test_data_len = strlen(test_data);
    
    // Allocate memory for testing
    void* test_addr = mmap(nullptr, test_size, PROT_READ | PROT_WRITE, 
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (test_addr == MAP_FAILED) {
        std::cerr << "Failed to allocate test memory" << std::endl;
        return 1;
    }
    
    // Write test data to the memory
    memcpy(test_addr, test_data, test_data_len);
    
    std::cout << "Original memory address: " << test_addr << std::endl;
    std::cout << "Original data: " << (char*)test_addr << std::endl;
    
    try {
        // Create hf3fsUtil instance
        hf3fsUtil utils;
        utils.mount_point = "/mnt/3fs/"; // Set mount point
        
        // Create shared memory metadata
        nixlHf3fsShmMetadata* shm_metadata = new nixlHf3fsShmMetadata(test_addr, test_size, utils);
        
        std::cout << "Shared memory name: " << shm_metadata->shm_name << std::endl;
        std::cout << "IOV base address: " << (void*)shm_metadata->iov.base << std::endl;
        std::cout << "IOV size: " << shm_metadata->iov.size << std::endl;
        
        // Verify that the shared memory was created successfully
        assert(shm_metadata->iov.base != nullptr);
        assert(shm_metadata->iov.size == test_size);
        
        // Verify that the data is accessible through the IOV
        std::cout << "Data through IOV: " << (char*)shm_metadata->iov.base << std::endl;
        assert(memcmp(test_addr, shm_metadata->iov.base, test_data_len) == 0);
        
        // Test writing to shared memory through IOV
        const char new_data[] = "Modified via IOV wrapper!";
        memcpy(shm_metadata->iov.base, new_data, strlen(new_data));
        
        std::cout << "Modified data in original memory: " << (char*)test_addr << std::endl;
        std::cout << "Modified data in IOV: " << (char*)shm_metadata->iov.base << std::endl;
        
        // Verify that changes are reflected in both locations
        assert(memcmp(test_addr, shm_metadata->iov.base, strlen(new_data)) == 0);
        
        // Test that the shared memory object exists in the file system
        struct stat st;
        if (stat(shm_metadata->shm_name.c_str(), &st) == 0) {
            std::cout << "Shared memory object exists in file system" << std::endl;
            std::cout << "File size: " << st.st_size << " bytes" << std::endl;
        } else {
            std::cout << "Warning: Shared memory object not found in file system" << std::endl;
        }
        
        // Clean up
        delete shm_metadata;
        
    } catch (const nixlHf3fsShmException& e) {
        std::cerr << "Shared memory exception caught: " << e.what() << std::endl;
        std::cerr << "Error type: " << static_cast<int>(e.getErrorType()) << std::endl;
        munmap(test_addr, test_size);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        munmap(test_addr, test_size);
        return 1;
    }
    
    // Clean up the test memory
    munmap(test_addr, test_size);
    
    std::cout << "IOV wrapper test completed successfully!" << std::endl;
    return 0;
} 