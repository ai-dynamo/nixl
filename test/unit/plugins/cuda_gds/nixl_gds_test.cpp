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
#include <string>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <cstring>
#include <getopt.h>
#include "nixl_descriptors.h"
#include "nixl_params.h"
#include "nixl.h"
#include "common/nixl_time.h"

// Default values
#define DEFAULT_NUM_TRANSFERS 250
#define DEFAULT_TRANSFER_SIZE (10 * 1024 * 1024)  // 10MB
#define TEST_PHRASE "NIXL Storage Test Pattern 2025"
#define TEST_PHRASE_LEN (sizeof(TEST_PHRASE) - 1)  // -1 to exclude null terminator

// Get system page size
static size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);

// Progress bar configuration
#define PROGRESS_WIDTH 50

// Helper function to parse size strings like "1K", "2M", "3G"
size_t parse_size(const char* size_str) {
    char* end;
    size_t size = strtoull(size_str, &end, 10);
    if (end == size_str) {
        return 0;  // Invalid number
    }

    if (*end) {
        switch (toupper(*end)) {
            case 'K': size *= 1024; break;
            case 'M': size *= 1024 * 1024; break;
            case 'G': size *= 1024 * 1024 * 1024; break;
            default: return 0;  // Invalid suffix
        }
    }
    return size;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options] <directory_path>\n"
              << "Options:\n"
              << "  -d, --dram              Use DRAM for memory operations\n"
              << "  -v, --vram              Use VRAM for memory operations (default)\n"
              << "  -n, --num-transfers N   Number of transfers to perform (default: " << DEFAULT_NUM_TRANSFERS << ")\n"
              << "  -s, --size SIZE         Size of each transfer (default: " << DEFAULT_TRANSFER_SIZE << " bytes)\n"
              << "                          Can use K, M, or G suffix (e.g., 1K, 2M, 3G)\n"
              << "  -r, --no-read           Skip read test\n"
              << "  -w, --no-write          Skip write test\n"
              << "  -p, --pool-size SIZE    Size of batch pool (default: 8, range: 1-32)\n"
              << "  -b, --batch-limit SIZE  Maximum requests per batch (default: 128, range: 1-1024)\n"
              << "  -m, --max-req-size SIZE Maximum size per request (default: 16M, range: 1M-1G)\n"
              << "                          Can use K, M, or G suffix (e.g., 1K, 2M, 3G)\n"
              << "  -D, --direct            Use O_DIRECT for file operations (bypass page cache)\n"
              << "  -h, --help              Show this help message\n"
              << "\nExample:\n"
              << "  " << program_name << " -d -n 100 -s 2M -p 16 -b 256 -m 32M -D /path/to/dir\n";
}

void printProgress(float progress) {
    int barWidth = PROGRESS_WIDTH;

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

std::string generate_timestamped_filename(const std::string& base_name) {
    std::time_t t = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp),
                  "%Y%m%d%H%M%S", std::localtime(&t));
    return base_name + std::string(timestamp);
}

void validateBuffer(void* expected, void* actual, size_t size, const char* operation) {
    if (memcmp(expected, actual, size) != 0) {
        std::cerr << "Data validation failed for " << operation << std::endl;
        exit(-1);
    }
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

// Helper function to fill GPU buffer with repeating pattern
cudaError_t fill_gpu_test_pattern(void* gpu_buffer, size_t size) {
    char* host_buffer = (char*)malloc(size);
    if (!host_buffer) {
        return cudaErrorMemoryAllocation;
    }

    fill_test_pattern(host_buffer, size);
    cudaError_t err = cudaMemcpy(gpu_buffer, host_buffer, size, cudaMemcpyHostToDevice);
    free(host_buffer);
    return err;
}

void clear_buffer(void* buffer, size_t size) {
    memset(buffer, 0, size);
}

cudaError_t clear_gpu_buffer(void* gpu_buffer, size_t size) {
    return cudaMemset(gpu_buffer, 0, size);
}

// Helper function to validate GPU buffer
bool validate_gpu_buffer(void* gpu_buffer, size_t size) {
    char* host_buffer = (char*)malloc(size);
    char* expected_buffer = (char*)malloc(size);
    if (!host_buffer || !expected_buffer) {
        free(host_buffer);
        free(expected_buffer);
        return false;
    }

    // Copy GPU data to host
    cudaError_t err = cudaMemcpy(host_buffer, gpu_buffer, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(host_buffer);
        free(expected_buffer);
        return false;
    }

    // Create expected pattern
    fill_test_pattern(expected_buffer, size);

    // Compare
    bool match = (memcmp(host_buffer, expected_buffer, size) == 0);

    free(host_buffer);
    free(expected_buffer);
    return match;
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

int main(int argc, char *argv[])
{
    nixl_status_t               ret = NIXL_SUCCESS;
    void                        **vram_addr = NULL;
    void                        **dram_addr = NULL;
    std::string                 role;
    int                         status = 0;
    int                         i;
    int                         *fd = NULL;
    bool                        use_dram = false;
    bool                        use_vram = false;
    int                         opt;
    std::string                 dir_path;
    size_t                      transfer_size = DEFAULT_TRANSFER_SIZE;
    int                         num_transfers = DEFAULT_NUM_TRANSFERS;
    bool                        skip_read = false;
    bool                        skip_write = false;
    unsigned int                pool_size = 8;
    unsigned int                batch_limit = 128;
    size_t                      max_request_size = 16 * 1024 * 1024;
    nixlTime::us_t              total_time(0);
    double                      total_data_gb = 0;
    bool                        use_direct = false;

    // Parse command line options
    static struct option long_options[] = {
        {"dram",            no_argument,       0, 'd'},
        {"vram",            no_argument,       0, 'v'},
        {"num-transfers",   required_argument, 0, 'n'},
        {"size",           required_argument, 0, 's'},
        {"no-read",        no_argument,       0, 'r'},
        {"no-write",       no_argument,       0, 'w'},
        {"pool-size",      required_argument, 0, 'p'},
        {"batch-limit",    required_argument, 0, 'b'},
        {"max-req-size",   required_argument, 0, 'm'},
        {"direct",         no_argument,       0, 'D'},
        {"help",           no_argument,       0, 'h'},
        {0,                0,                 0,  0}
    };

    while ((opt = getopt_long(argc, argv, "dvn:s:rwp:b:m:Dh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd':
                use_dram = true;
                break;
            case 'v':
                use_vram = true;
                break;
            case 'n':
                num_transfers = atoi(optarg);
                if (num_transfers <= 0) {
                    std::cerr << "Error: Number of transfers must be positive\n";
                    return 1;
                }
                break;
            case 's':
                transfer_size = parse_size(optarg);
                if (transfer_size == 0) {
                    std::cerr << "Error: Invalid transfer size format\n";
                    return 1;
                }
                break;
            case 'r':
                skip_read = true;
                break;
            case 'w':
                skip_write = true;
                break;
            case 'p':
                pool_size = atoi(optarg);
                if (pool_size < 1 || pool_size > 32) {
                    std::cerr << "Error: Pool size must be between 1 and 32\n";
                    return 1;
                }
                break;
            case 'b':
                batch_limit = atoi(optarg);
                if (batch_limit < 1 || batch_limit > 1024) {
                    std::cerr << "Error: Batch limit must be between 1 and 1024\n";
                    return 1;
                }
                break;
            case 'm':
                max_request_size = parse_size(optarg);
                if (max_request_size < 1024*1024 || max_request_size > 1024*1024*1024) {
                    std::cerr << "Error: Max request size must be between 1M and 1G\n";
                    return 1;
                }
                break;
            case 'D':
                use_direct = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (skip_read && skip_write) {
        std::cerr << "Error: Cannot skip both read and write tests\n";
        return 1;
    }

    // Check if directory path is provided
    if (optind >= argc) {
        std::cerr << "Error: Directory path is required\n";
        print_usage(argv[0]);
        return 1;
    }
    dir_path = argv[optind];

    // If neither is specified, default to VRAM
    if (!use_dram && !use_vram) {
        use_vram = true;
    }

    // Check if both DRAM and VRAM are specified
    if (use_dram && use_vram) {
        std::cerr << "Error: Cannot specify both DRAM (-d) and VRAM (-v) options\n";
        print_usage(argv[0]);
        return 1;
    }

    // Allocate arrays based on num_transfers
    if (use_vram) {
        vram_addr = new void*[num_transfers];
    }
    if (use_dram) {
        dram_addr = new void*[num_transfers];
    }
    fd = new int[num_transfers];

    // Initialize NIXL components
    nixlAgentConfig             cfg(true);
    nixl_b_params_t             params;
    nixlBlobDesc                *vram_buf = use_vram ? new nixlBlobDesc[num_transfers] : NULL;
    nixlBlobDesc                *dram_buf = use_dram ? new nixlBlobDesc[num_transfers] : NULL;
    nixlBlobDesc                *ftrans = new nixlBlobDesc[num_transfers];
    nixlBackendH                *gds;
    nixl_reg_dlist_t            vram_for_gds(VRAM_SEG);
    nixl_reg_dlist_t            dram_for_gds(DRAM_SEG);
    nixl_reg_dlist_t            file_for_gds(FILE_SEG);
    nixlXferReqH                *treq;
    std::string                 name;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "                 NIXL STORAGE TEST STARTING (GDS PLUGIN)                     " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Mode: " << (use_dram ? "DRAM" : "VRAM") << std::endl;
    std::cout << "- Number of transfers: " << num_transfers << std::endl;
    std::cout << "- Transfer size: " << transfer_size << " bytes" << std::endl;
    std::cout << "- Total data: " << std::fixed << std::setprecision(2)
              << ((transfer_size * num_transfers) / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "- Directory: " << dir_path << std::endl;
    std::cout << "- Batch pool size: " << pool_size << std::endl;
    std::cout << "- Batch limit: " << batch_limit << std::endl;
    std::cout << "- Max request size: " << max_request_size << " bytes" << std::endl;
    std::cout << "- Use O_DIRECT: " << (use_direct ? "Yes" : "No") << std::endl;
    std::cout << "- Operation: ";
    if (!skip_read && !skip_write) {
        std::cout << "Read and Write";
    } else if (skip_read) {
        std::cout << "Write Only";
    } else {  // skip_write
        std::cout << "Read Only";
    }
    std::cout << std::endl;
    std::cout << "============================================================\n" << std::endl;

    nixlAgent agent("GDSTester", cfg);

    // Set GDS backend parameters
    params["batch_pool_size"] = std::to_string(pool_size);
    params["batch_limit"] = std::to_string(batch_limit);
    params["max_request_size"] = std::to_string(max_request_size);

    // Create backends
    ret = agent.createBackend("GDS", params, gds);
    if (ret != NIXL_SUCCESS || gds == NULL) {
        std::cerr << "Error creating GDS backend: "
                  << (ret != NIXL_SUCCESS ? "Failed to create backend" : "Backend handle is NULL")
                  << std::endl;
        goto cleanup;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 1: Allocating and initializing buffers" << std::endl;
    std::cout << "============================================================" << std::endl;
    for (i = 0; i < num_transfers; i++) {
        if (use_vram) {
            // Allocate and initialize VRAM buffer
            if (cudaMalloc(&vram_addr[i], transfer_size) != cudaSuccess) {
                std::cerr << "CUDA malloc failed\n";
                goto cleanup;
            }
            if (fill_gpu_test_pattern(vram_addr[i], transfer_size) != cudaSuccess) {
                std::cerr << "CUDA buffer initialization failed\n";
                goto cleanup;
            }
        }

        if (use_dram) {
            // Allocate and initialize DRAM buffer
            if (posix_memalign(&dram_addr[i], PAGE_SIZE, transfer_size) != 0) {
                std::cerr << "DRAM allocation failed\n";
                goto cleanup;
            }
            fill_test_pattern(dram_addr[i], transfer_size);
        }

        // Create test file
        name = generate_timestamped_filename("testfile");
        name = dir_path + "/" + name + "_" + std::to_string(i);

        int flags = O_RDWR|O_CREAT;
        if (use_direct) {
            flags |= O_DIRECT;
        }
        fd[i] = open(name.c_str(), flags, 0744);
        if (fd[i] < 0) {
            std::cerr << "Failed to open file: " << name << " - " << strerror(errno) << std::endl;
            goto cleanup;
        }

        // Set up descriptors
        if (use_vram) {
            vram_buf[i].addr   = (uintptr_t)(vram_addr[i]);
            vram_buf[i].len    = transfer_size;
            vram_buf[i].devId  = 0;
            vram_for_gds.addDesc(vram_buf[i]);
        }

        if (use_dram) {
            dram_buf[i].addr   = (uintptr_t)(dram_addr[i]);
            dram_buf[i].len    = transfer_size;
            dram_buf[i].devId  = 0;
            dram_for_gds.addDesc(dram_buf[i]);
        }

        ftrans[i].addr  = 0;
        ftrans[i].len   = transfer_size;
        ftrans[i].devId = fd[i];
        file_for_gds.addDesc(ftrans[i]);

        printProgress(float(i + 1) / num_transfers);
    }

    std::cout << "\n=== Registering memory ===" << std::endl;
    ret = agent.registerMem(file_for_gds);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register file memory\n";
        goto cleanup;
    }

    if (use_vram) {
        ret = agent.registerMem(vram_for_gds);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register VRAM memory\n";
            goto cleanup;
        }
    }

    if (use_dram) {
        ret = agent.registerMem(dram_for_gds);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM memory\n";
            goto cleanup;
        }
    }

    // Prepare transfer lists
    nixl_xfer_dlist_t file_for_gds_list = file_for_gds.trim();
    nixl_xfer_dlist_t src_list = use_dram ? dram_for_gds.trim() : vram_for_gds.trim();

    using namespace nixlTime;

    // Perform write test if not skipped
    if (!skip_write) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 2: Memory to File Transfer (Write Test)" << std::endl;
        std::cout << "============================================================" << std::endl;

        ret = agent.createXferReq(NIXL_WRITE, src_list, file_for_gds_list,
                                 "GDSTester", treq);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to create write transfer request\n";
            goto cleanup;
        }

        us_t write_start = getUs();
        status = agent.postXferReq(treq);
        if (status < 0) {
            std::cerr << "Failed to post write transfer request\n";
            goto cleanup;
        }

        while (status == NIXL_IN_PROG) {
            status = agent.getXferStatus(treq);
            if (status < 0) {
                std::cerr << "Error during write transfer\n";
                goto cleanup;
            }
        }
        us_t write_end = getUs();

        agent.releaseXferReq(treq);
        us_t write_duration = write_end - write_start;
        total_time += write_duration;

        double data_gb = (transfer_size * num_transfers) / (1024.0 * 1024.0 * 1024.0);
        total_data_gb += data_gb;
        double seconds = write_duration / 1000000.0;
        double gbps = data_gb / seconds;

        std::cout << "Write completed:" << std::endl;
        std::cout << "- Time: " << format_duration(write_duration) << std::endl;
        std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB" << std::endl;
        std::cout << "- Speed: " << gbps << " GB/s" << std::endl;
    }

    // Clear buffers before read if doing both operations
    if (!skip_read && !skip_write) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 3: Clearing buffers for read test" << std::endl;
        std::cout << "============================================================" << std::endl;
        for (i = 0; i < num_transfers; i++) {
            if (use_vram && vram_addr[i]) {
                if (clear_gpu_buffer(vram_addr[i], transfer_size) != cudaSuccess) {
                    std::cerr << "Failed to clear VRAM buffer " << i << std::endl;
                    goto cleanup;
                }
            }
            if (use_dram && dram_addr[i]) {
                clear_buffer(dram_addr[i], transfer_size);
            }
            printProgress(float(i + 1) / num_transfers);
        }
    }

    // Perform read test if not skipped
    if (!skip_read) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 4: File to Memory Transfer (Read Test)" << std::endl;
        std::cout << "============================================================" << std::endl;

        ret = agent.createXferReq(NIXL_READ, src_list, file_for_gds_list,
                                 "GDSTester", treq);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to create read transfer request\n";
            goto cleanup;
        }

        us_t read_start = getUs();
        status = agent.postXferReq(treq);
        if (status < 0) {
            std::cerr << "Failed to post read transfer request\n";
            goto cleanup;
        }

        while (status == NIXL_IN_PROG) {
            status = agent.getXferStatus(treq);
            if (status < 0) {
                std::cerr << "Error during read transfer\n";
                goto cleanup;
            }
        }
        us_t read_end = getUs();

        agent.releaseXferReq(treq);
        us_t read_duration = read_end - read_start;
        total_time += read_duration;

        double data_gb = (transfer_size * num_transfers) / (1024.0 * 1024.0 * 1024.0);
        total_data_gb += data_gb;
        // Ensure we don't divide by zero and use microseconds for more precision
        double seconds = std::max(read_duration / 1000000.0, 0.000001); // minimum 1 microsecond
        double gbps = data_gb / seconds;

        std::cout << "Read completed:" << std::endl;
        std::cout << "- Time: " << format_duration(read_duration) << std::endl;
        std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB" << std::endl;
        std::cout << "- Speed: " << std::fixed << std::setprecision(2) << gbps << " GB/s" << std::endl;

        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 5: Validating read data" << std::endl;
        std::cout << "============================================================" << std::endl;
        for (i = 0; i < num_transfers; i++) {
            if (use_vram) {
                if (!validate_gpu_buffer(vram_addr[i], transfer_size)) {
                    std::cerr << "VRAM buffer " << i << " validation failed\n";
                    goto cleanup;
                }
            }
            if (use_dram) {
                char* expected_buffer = (char*)malloc(transfer_size);
                if (!expected_buffer) {
                    std::cerr << "Failed to allocate validation buffer\n";
                    goto cleanup;
                }
                fill_test_pattern(expected_buffer, transfer_size);
                if (memcmp(dram_addr[i], expected_buffer, transfer_size) != 0) {
                    std::cerr << "DRAM buffer " << i << " validation failed\n";
                    free(expected_buffer);
                    goto cleanup;
                }
                free(expected_buffer);
            }
            printProgress(float(i + 1) / num_transfers);
        }
        std::cout << "\nVerification completed successfully!" << std::endl;
    }

cleanup:
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 6: Cleanup" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Delete test files
    std::cout << "Deleting test files..." << std::endl;
    for (i = 0; i < num_transfers; i++) {
        if (fd[i] > 0) {
            // Get the filename from the file descriptor
            char proc_path[64];
            char filename[PATH_MAX];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd[i]);
            ssize_t len = readlink(proc_path, filename, sizeof(filename) - 1);
            if (len != -1) {
                filename[len] = '\0';
                close(fd[i]);
                if (unlink(filename) != 0) {
                    std::cerr << "Warning: Failed to delete file " << filename << ": " << strerror(errno) << std::endl;
                }
            }
        }
    }
    printProgress(1.0);

    // Cleanup resources
    agent.deregisterMem(file_for_gds);
    if (use_vram) {
        agent.deregisterMem(vram_for_gds);
        for (i = 0; i < num_transfers; i++) {
            if (vram_addr[i]) cudaFree(vram_addr[i]);
        }
        delete[] vram_addr;
        delete[] vram_buf;
    }
    if (use_dram) {
        agent.deregisterMem(dram_for_gds);
        for (i = 0; i < num_transfers; i++) {
            if (dram_addr[i]) free(dram_addr[i]);
        }
        delete[] dram_addr;
        delete[] dram_buf;
    }
    if (fd) {
        delete[] fd;
    }
    delete[] ftrans;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "                    TEST SUMMARY                             " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Total time: " << format_duration(total_time) << std::endl;
    std::cout << "Total data: " << std::fixed << std::setprecision(2) << total_data_gb << " GB" << std::endl;
    std::cout << "============================================================" << std::endl;
    return 0;
}
