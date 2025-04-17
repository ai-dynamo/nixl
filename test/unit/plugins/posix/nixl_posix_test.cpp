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
#include <filesystem>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <string>
#include "nixl.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"
#include "common/nixl_time.h"

// Default values
#define DEFAULT_NUM_TRANSFERS 250
#define DEFAULT_TRANSFER_SIZE (10 * 1024 * 1024)  // 10MB
#define TEST_PHRASE "NIXL Storage Test Pattern 2025 POSIX"
#define TEST_PHRASE_LEN (sizeof(TEST_PHRASE) - 1)  // -1 to exclude null terminator
#define TEST_FILE_NAME "testfile"
#define STD_FILE_PERMISSIONS 0744

// Sizes
#define KB_SIZE (1024)
#define MB_SIZE (1024 * 1024)
#define GB_SIZE (1024 * 1024 * 1024)
#define US_TO_S(us) ((us) / 1000000.0)

// Get system page size
static size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);

// Print constants
#define LINE_WIDTH 60
#define PROGRESS_BAR_WIDTH (LINE_WIDTH - 2) // -2 for the brackets
#define LINE_STR (std::string(LINE_WIDTH, '='))
#define CENTER_STR(str) (std::string((LINE_WIDTH - (std::string(str).length())) / 2, ' ') + str)


// Helper function to parse size strings like "1K", "2M", "3G"
size_t parse_size(const char* size_str) {
    char* end;
    size_t size = strtoull(size_str, &end, 10);
    if (end == size_str) {
        return 0;  // Invalid number
    }

    if (*end) {
        switch (toupper(*end)) {
            case 'K': size *= KB_SIZE; break;
            case 'M': size *= MB_SIZE; break;
            case 'G': size *= GB_SIZE; break;
            default: return 0;  // Invalid suffix
        }
    }
    return size;
}

// Helper function to fill buffer with repeating pattern
void fill_test_pattern(void* buffer, size_t size) {
    char* buf = (char*)buffer;
    size_t phrase_len = TEST_PHRASE_LEN;
    size_t offset = 0;

    while (offset < size) {
        size_t remaining = size - offset;
        size_t copy_len = (remaining < phrase_len) ? remaining : phrase_len;
        memcpy(buf + offset, TEST_PHRASE, copy_len);
        offset += copy_len;
    }
}

// Helper function to format duration
std::string format_duration(nixlTime::us_t us) {
    nixlTime::ms_t ms = us/1000.0;
    if (ms < 1000) {
        return std::to_string(ms) + " ms";
    }
    double seconds = ms / 1000.0;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << seconds << " sec";
    return ss.str();
}

// Helper function to generate timestamped filename
std::string generate_timestamped_filename(const std::string& base_name) {
    std::time_t t = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp),
                  "%Y%m%d%H%M%S", std::localtime(&t));
    return base_name + std::string(timestamp);
}

void printProgress(float progress) {
    int barWidth = PROGRESS_BAR_WIDTH;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% ";

    // Add completion indicator
    if (progress >= 1.0) {
        std::cout << "DONE!" << std::endl;
    } else {
        std::cout << "\r";
        std::cout.flush();
    }
}

int main(int argc, char *argv[])
{
    std::cout << "NIXL POSIX Plugin Test" << std::endl;

    int         num_transfers   = DEFAULT_NUM_TRANSFERS;
    size_t      transfer_size   = DEFAULT_TRANSFER_SIZE;
    std::string dir_path        = ".";  // Default to current directory
    void        **dram_addr     = new void*[num_transfers];
    int         *fd             = new int[num_transfers];
    int         file_open_flags = O_RDWR|O_CREAT;
    // TODO: Check if we need to add O_DIRECT flag
    // if (use_direct) {
    //     file_open_flags |= O_DIRECT;
    // }
    
    // Convert directory path to absolute path using std::filesystem
    std::filesystem::path path_obj(dir_path);
    std::string abs_path = std::filesystem::absolute(path_obj).string();

    // Initialize NIXL components
    nixlAgentConfig   cfg(true);
    nixl_b_params_t   params;
    nixlBlobDesc      *dram_buf = new nixlBlobDesc[num_transfers];
    nixlBlobDesc      *ftrans = new nixlBlobDesc[num_transfers];
    nixlBackendH      *posix;
    nixl_reg_dlist_t  dram_for_posix(DRAM_SEG);
    nixl_reg_dlist_t  file_for_posix(FILE_SEG);
    nixlXferReqH      *treq;
    std::string       name;
    nixl_xfer_dlist_t src_list(DRAM_SEG);
    nixl_xfer_dlist_t file_for_posix_list(FILE_SEG);

    // Control variables
    nixl_status_t    status;
    int              ret = 0;
    int              i = 0;
    nixlTime::us_t   write_start;
    nixlTime::us_t   write_end;
    nixlTime::us_t   write_duration;
    nixlTime::us_t   total_time(0);
    double           total_data_gb(0);
    double           gbps;
    double           seconds;
    double           data_gb;


    // TODO: Implement POSIX plugin test
    // =====================================================================
    // TEST CONFIGURATION DISPLAY
    // =====================================================================
    // Print test configuration information
    std::cout << "\n" << LINE_STR << std::endl;
    std::cout << CENTER_STR("NIXL STORAGE TEST STARTING (POSIX PLUGIN)") << std::endl;
    std::cout << LINE_STR << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Number of transfers: " << num_transfers << std::endl;
    std::cout << "- Transfer size: " << transfer_size << " bytes" << std::endl;
    std::cout << "- Total data: " << std::fixed << std::setprecision(2)
              << ((transfer_size * num_transfers) / (GB_SIZE)) << " GB" << std::endl;
    std::cout << "- Directory: " << abs_path << std::endl;
    std::cout << std::endl;
    std::cout << LINE_STR << "\n" << std::endl;

    nixlAgent agent("POSIXTester", cfg);

    // Create POSIX backend
    status = agent.createBackend("POSIX", params, posix);
    if (status != NIXL_SUCCESS) { // TODO should we check for NULL? i see gds test does, but i dont think it is needed
        std::cerr << "Error creating POSIX backend: " << nixlEnumStrings::statusStr(status) << std::endl;
        ret = 1;
        goto out_err;
    }

    std::cout << "\n" << LINE_STR << std::endl;
    std::cout << CENTER_STR("PHASE 1: Allocating and initializing buffers") << std::endl;
    std::cout << LINE_STR << std::endl;

    // Allocate and initialize DRAM buffer
    for (i = 0; i < num_transfers; ++i) {
        if (posix_memalign(&dram_addr[i], PAGE_SIZE, transfer_size) != 0) {
            std::cerr << "DRAM allocation failed\n";
            ret = 1;
            goto out_err;
        }
        fill_test_pattern(dram_addr[i], transfer_size);
    
        // Create test file
        name = generate_timestamped_filename(TEST_FILE_NAME);
        name = dir_path + "/" + name + "_" + std::to_string(i);

        fd[i] = open(name.c_str(), file_open_flags, 0744);
        if (fd[i] < 0) {
            std::cerr << "Failed to open file: " << name << " - " << strerror(errno) << std::endl;
            goto out_err;
            ret = 1;
        }
        dram_buf[i].addr   = (uintptr_t)(dram_addr[i]);
        dram_buf[i].len    = transfer_size;
        dram_buf[i].devId  = 0;
        dram_for_posix.addDesc(dram_buf[i]);

        ftrans[i].addr  = 0;
        ftrans[i].len   = transfer_size;
        ftrans[i].devId = fd[i];
        file_for_posix.addDesc(ftrans[i]);

        printProgress(float(i + 1) / num_transfers);
    }

    std::cout << "\n" << LINE_STR << std::endl;
    std::cout << CENTER_STR("PHASE 2: Registering memory with NIXL") << std::endl;
    std::cout << LINE_STR << std::endl;

    i = 0;
    ret = agent.registerMem(dram_for_posix);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register DRAM memory with NIXL" << std::endl;
        ret = 1;
        goto out_err;
    }
    printProgress(float(i + 1) / 2);
    ++i;
    ret = agent.registerMem(file_for_posix);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register file memory with NIXL" << std::endl;
        ret = 1;
        goto out_err;
    }
    printProgress(float(i + 1) / 2);

    std::cout << "\n" << LINE_STR << std::endl;
    std::cout << CENTER_STR("PHASE 3: Memory to File Transfer (Write Test)") << std::endl;
    std::cout << LINE_STR << std::endl;

    file_for_posix_list = file_for_posix.trim();
    src_list = dram_for_posix.trim();

    status = agent.createXferReq(NIXL_WRITE, src_list, file_for_posix_list,
                                 "POSIXTester", treq);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Failed to create write transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
        ret = 1;
        goto out_err;
    }

    write_start = nixlTime::getUs();
    status = agent.postXferReq(treq);
    if (status < 0) {
        std::cerr << "Failed to post write transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
        ret = 1;
        goto out_err;
    }

    // Wait for transfer to complete
    while (status == NIXL_IN_PROG) {
        status = agent.getXferStatus(treq);
        if (status < 0) {
            std::cerr << "Error during write transfer - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            ret = 1;
            goto out_err;
        }
    }
    write_end = nixlTime::getUs();
    agent.releaseXferReq(treq);
    write_duration = write_end - write_start;
    total_time += write_duration;

    data_gb = (transfer_size * num_transfers) / (GB_SIZE);
    total_data_gb += data_gb;
    seconds = US_TO_S(write_duration);
    gbps = data_gb / seconds;

    std::cout << "Write completed:" << std::endl;
    std::cout << "- Time: " << format_duration(write_duration) << std::endl;
    std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB" << std::endl;
    std::cout << "- Speed: " << gbps << " GB/s" << std::endl;

out_err:
    return ret;
} 