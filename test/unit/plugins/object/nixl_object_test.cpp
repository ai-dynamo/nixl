/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define DEFAULT_NUM_TRANSFERS 64
#define DEFAULT_TRANSFER_SIZE (16 * 1024 * 1024) // 16MB
#define DEFAULT_ITERATIONS 1 // Default number of iterations
#define DEFAULT_BACKEND "OBJ"
#define TEST_PHRASE "NIXL Storage Test Pattern 2026"
#define TEST_PHRASE_LEN (sizeof(TEST_PHRASE) - 1) // -1 to exclude null terminator

// Get system page size
static size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);

// Progress bar configuration
#define PROGRESS_WIDTH 50

// Helper function to parse size strings like "1K", "2M", "3G"
size_t
parse_size(const char *size_str) {
    char *end;
    size_t size = strtoull(size_str, &end, 10);
    if (end == size_str) {
        return 0; // Invalid number
    }

    if (*end) {
        switch (toupper(*end)) {
        case 'K':
            size *= 1024;
            break;
        case 'M':
            size *= 1024 * 1024;
            break;
        case 'G':
            size *= 1024 * 1024 * 1024;
            break;
        default:
            return 0; // Invalid suffix
        }
    }
    return size;
}

void
print_usage(const char *program_name) {
    std::cerr << "Usage: " << program_name << " [options] <directory_path>\n"
              << "Options:\n"
              << "  -d, --dram              Use DRAM for memory operations (default)\n"
              << "  -n, --num-transfers N   Number of transfers to perform (default: "
              << DEFAULT_NUM_TRANSFERS << ")\n"
              << "  -s, --size SIZE         Size of each transfer (default: "
              << DEFAULT_TRANSFER_SIZE << " bytes)\n"
              << "                          Can use K, M, or G suffix (e.g., 1K, 2M, 3G)\n"
              << "  -r, --no-read           Skip read test\n"
              << "  -w, --no-write          Skip write test\n"
              << "  -t, --iterations N      Number of iterations for each transfer (default: "
              << DEFAULT_ITERATIONS << ")\n"
              << "  -e, --endpoint ENDPOINT S3 Endpoint URL\n"
              << "  -u, --bucket BUCKET     S3 Bucket name\n"
              << "  -h, --help              Show this help message\n"
              << "\nExamples:\n"
              << "  " << program_name << " -d -n 100 -s 16M -t 5\n"
              << "  " << program_name << " -a default -d -n 100 -s 16M -t 5\n";
}

void
printProgress(float progress) {
    int barWidth = PROGRESS_WIDTH;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
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

void
validateBuffer(void *expected, void *actual, size_t size, const char *operation) {
    if (memcmp(expected, actual, size) != 0) {
        std::cerr << "Data validation failed for " << operation << std::endl;
        exit(-1);
    }
}

// Helper function to fill buffer with repeating pattern
void
fill_test_pattern(void *buffer, size_t size) {
    char *buf = (char *)buffer;
    size_t phrase_len = TEST_PHRASE_LEN;
    size_t offset = 0;

    while (offset < size) {
        size_t remaining = size - offset;
        size_t copy_len = (remaining < phrase_len) ? remaining : phrase_len;
        memcpy(buf + offset, TEST_PHRASE, copy_len);
        offset += copy_len;
    }
}

void
clear_buffer(void *buffer, size_t size) {
    memset(buffer, 0, size);
}

// Helper function to format duration
std::string
format_duration(nixlTime::us_t us) {
    nixlTime::ms_t ms = us / 1000.0;
    if (ms < 1000) {
        return std::to_string(ms) + " ms";
    }
    double seconds = ms / 1000.0;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << seconds << " sec";
    return ss.str();
}

int
main(int argc, char *argv[]) {
    nixl_status_t ret = NIXL_SUCCESS;
    void **dram_addr = NULL;
    int status = 0;
    int i;
    bool use_dram = false;
    int opt;
    size_t transfer_size = DEFAULT_TRANSFER_SIZE;
    int num_transfers = DEFAULT_NUM_TRANSFERS;
    bool skip_read = false;
    bool skip_write = false;
    nixlTime::us_t total_time(0);
    nixlTime::us_t reg_time(0);
    double total_data_gb = 0;
    unsigned int iterations = DEFAULT_ITERATIONS;
    std::string endpoint;
    std::string bucket;

    // Parse command line options
    static struct option long_options[] = {{"dram", no_argument, 0, 'd'},
                                           {"num-transfers", required_argument, 0, 'n'},
                                           {"size", required_argument, 0, 's'},
                                           {"no-read", no_argument, 0, 'r'},
                                           {"no-write", no_argument, 0, 'w'},
                                           {"iterations", required_argument, 0, 't'},
                                           {"endpoint", required_argument, 0, 'e'},
                                           {"bucket", required_argument, 0, 'u'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "dn:s:rwt:he:u:", long_options, NULL)) != -1) {
        switch (opt) {
        case 'e':
            endpoint = optarg;
            break;
        case 'u':
            bucket = optarg;
            break;
        case 'd':
            use_dram = true;
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
        case 't':
            iterations = atoi(optarg);
            if (iterations <= 0) {
                std::cerr << "Error: Number of iterations must be positive\n";
                return 1;
            }
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

    // Default to DRAM
    use_dram = true;

    // Allocate DRAM array
    dram_addr = new void *[num_transfers];

    // Initialize NIXL components
    nixlAgentConfig cfg(true);
    nixlBlobDesc *dram_buf = new nixlBlobDesc[num_transfers];
    nixlBlobDesc *objects = new nixlBlobDesc[num_transfers];
    nixlBackendH *obj;
    nixl_reg_dlist_t dram_for_obj(DRAM_SEG);
    nixl_reg_dlist_t obj_for_obj(OBJ_SEG);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "                 NIXL STORAGE TEST STARTING ( OBJ PLUGIN)                     "
              << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Mode: DRAM" << std::endl;
    std::cout << "- Number of transfers: " << num_transfers << std::endl;
    std::cout << "- Transfer size: " << transfer_size << " bytes" << std::endl;
    std::cout << "- Total data: " << std::fixed << std::setprecision(2)
              << ((transfer_size * num_transfers) / (1024.0 * 1024.0 * 1024.0)) << " GB"
              << std::endl;
    std::cout << "- Number of iterations: " << iterations << std::endl;
    std::cout << "- Operation: ";
    if (!skip_read && !skip_write) {
        std::cout << "Read and Write";
    } else if (skip_read) {
        std::cout << "Write Only";
    } else { // skip_write
        std::cout << "Read Only";
    }
    std::cout << std::endl;
    std::cout << "============================================================\n" << std::endl;

    nixlAgent agent("ObjTester", cfg);

    nixl_b_params_t params = {{"bucket", bucket},
                              {"endpoint_override", endpoint},
                              {"scheme", "http"},
                              {"use_virtual_addressing", "false"},
                              {"req_checksum", "required"}};


    // Create backends
    ret = agent.createBackend(DEFAULT_BACKEND, params, obj);
    if (ret != NIXL_SUCCESS || obj == NULL) {
        std::cerr << "Error creating " << DEFAULT_BACKEND << " backend: "
                  << (ret != NIXL_SUCCESS ? "Failed to create backend" : "Backend handle is NULL")
                  << std::endl;
        goto cleanup;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 1: Allocating and initializing buffers" << std::endl;
    std::cout << "============================================================" << std::endl;
    for (i = 0; i < num_transfers; i++) {
        // Allocate and initialize DRAM buffer
        if (posix_memalign(&dram_addr[i], PAGE_SIZE, transfer_size) != 0) {
            std::cerr << "DRAM allocation failed\n";
            goto cleanup;
        }
        fill_test_pattern(dram_addr[i], transfer_size);

        // Set up DRAM descriptor
        dram_buf[i].addr = (uintptr_t)(dram_addr[i]);
        dram_buf[i].len = transfer_size;
        dram_buf[i].devId = 0;
        dram_for_obj.addDesc(dram_buf[i]);

        objects[i].addr = 0;
        objects[i].len = transfer_size;
        objects[i].devId = i;
        objects[i].metaInfo = "test-write-key" + std::to_string(i);
        obj_for_obj.addDesc(objects[i]);

        printProgress(float(i + 1) / num_transfers);
    }
    using namespace nixlTime;
    us_t reg_start = getUs();

    std::cout << "\n=== Registering memory ===" << std::endl;

    ret = agent.registerMem(obj_for_obj);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register file memory\n";
        goto cleanup;
    }

    if (use_dram) {
        ret = agent.registerMem(dram_for_obj);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM memory\n";
            goto cleanup;
        }
    }

    us_t reg_end = getUs();

    reg_time = (reg_end - reg_start);

    std::cout << "Registration completed:" << std::endl;
    std::cout << "- Time: " << format_duration(reg_time) << std::endl;

    // Prepare transfer lists
    nixl_xfer_dlist_t obj_for_obj_list = obj_for_obj.trim();
    nixl_xfer_dlist_t src_list = dram_for_obj.trim();


    // Perform write test if not skipped
    if (!skip_write) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 2: Memory to Object Transfer (Write Test)" << std::endl;
        std::cout << "============================================================" << std::endl;

        us_t write_duration(0);
        nixlXferReqH *write_req = nullptr;

        // Create descriptor lists for all transfers
        nixl_reg_dlist_t src_reg(DRAM_SEG);
        nixl_reg_dlist_t obj_reg(OBJ_SEG);

        // Add all descriptors
        for (int transfer_idx = 0; transfer_idx < num_transfers; transfer_idx++) {
            src_reg.addDesc(dram_buf[transfer_idx]);
            obj_reg.addDesc(objects[transfer_idx]);
            printProgress(float(transfer_idx + 1) / num_transfers);
        }
        std::cout << "\nAll descriptors added." << std::endl;

        // Create transfer lists
        nixl_xfer_dlist_t src_list = src_reg.trim();
        nixl_xfer_dlist_t obj_list = obj_reg.trim();

        // Create single transfer request for all transfers
        ret = agent.createXferReq(NIXL_WRITE, src_list, obj_list, "ObjTester", write_req);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to create write transfer request" << std::endl;
            goto cleanup;
        }
        std::cout << "Write transfer request created." << std::endl;

        // Now do the iterations
        for (unsigned int iter = 0; iter < iterations; iter++) {
            us_t iter_start = getUs();

            status = agent.postXferReq(write_req);
            if (status < 0) {
                std::cerr << "Failed to post write transfer request" << std::endl;
                goto cleanup;
            }

            // Wait for completion
            while (status == NIXL_IN_PROG) {
                status = agent.getXferStatus(write_req);
                if (status < 0) {
                    std::cerr << "Error during write transfer" << std::endl;
                    goto cleanup;
                }
            }

            us_t iter_end = getUs();
            write_duration += (iter_end - iter_start);

            if (iterations > 1) {
                printProgress(float(iter + 1) / iterations);
            }
        }

        agent.releaseXferReq(write_req);
        total_time += write_duration;

        double data_gb = (transfer_size * num_transfers * iterations) / (1024.0 * 1024.0 * 1024.0);
        total_data_gb += data_gb;
        double seconds = write_duration / 1000000.0;
        double gbps = data_gb / seconds;

        std::cout << "Write completed:" << std::endl;
        std::cout << "- Time: " << format_duration(write_duration) << std::endl;
        std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB"
                  << std::endl;
        std::cout << "- Speed: " << gbps << " GB/s" << std::endl;
    }

    // Clear buffers before read if doing both operations
    if (!skip_read && !skip_write) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 3: Clearing buffers for read test" << std::endl;
        std::cout << "============================================================" << std::endl;
        for (i = 0; i < num_transfers; i++) {
            clear_buffer(dram_addr[i], transfer_size);
            printProgress(float(i + 1) / num_transfers);
        }
    }

    // Create extra params with backend
    nixl_opt_args_t extra_params;
    extra_params.backends = {obj};
    std::vector<nixl_query_resp_t> resp;
    status = agent.queryMem(obj_for_obj, resp, &extra_params);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Failed to query object memory status\n";
        goto cleanup;
    }
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 4: Querying Objects" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "\nQueryMem Results:" << std::endl;
    std::cout << "response count: " << resp.size() << std::endl;
    for (const auto &r : resp) {
        if (!r.has_value()) {
            std::cerr << "Error: QueryMem response has no value\n";
            goto cleanup;
        }
    }
    std::cout << "All queried objects are valid." << std::endl;


    // Perform read test if not skipped
    if (!skip_read) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 5: Object to Memory Transfer (Read Test)" << std::endl;
        std::cout << "============================================================" << std::endl;

        us_t read_duration(0);
        nixlXferReqH *read_req = nullptr;

        // Create descriptor lists for all transfers
        nixl_reg_dlist_t src_reg(DRAM_SEG);
        nixl_reg_dlist_t obj_reg(OBJ_SEG);

        // Add all descriptors
        for (int transfer_idx = 0; transfer_idx < num_transfers; transfer_idx++) {
            src_reg.addDesc(dram_buf[transfer_idx]);
            obj_reg.addDesc(objects[transfer_idx]);
            printProgress(float(transfer_idx + 1) / num_transfers);
        }
        std::cout << "\nAll descriptors added." << std::endl;

        // Create transfer lists
        nixl_xfer_dlist_t src_list = src_reg.trim();
        nixl_xfer_dlist_t obj_list = obj_reg.trim();

        // Create single transfer request for all transfers
        ret = agent.createXferReq(NIXL_READ, src_list, obj_list, "ObjTester", read_req);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to create read transfer request" << std::endl;
            goto cleanup;
        }
        std::cout << "Read transfer request created." << std::endl;

        // Now do the iterations
        for (unsigned int iter = 0; iter < iterations; iter++) {
            us_t iter_start = getUs();

            status = agent.postXferReq(read_req);
            if (status < 0) {
                std::cerr << "Failed to post read transfer request" << std::endl;
                goto cleanup;
            }

            // Wait for completion
            while (status == NIXL_IN_PROG) {
                status = agent.getXferStatus(read_req);
                if (status < 0) {
                    std::cerr << "Error during read transfer" << std::endl;
                    goto cleanup;
                }
            }

            us_t iter_end = getUs();
            read_duration += (iter_end - iter_start);

            if (iterations > 1) {
                printProgress(float(iter + 1) / iterations);
            }
        }

        agent.releaseXferReq(read_req);
        total_time += read_duration;

        double data_gb = (transfer_size * num_transfers * iterations) / (1024.0 * 1024.0 * 1024.0);
        total_data_gb += data_gb;
        double seconds = read_duration / 1000000.0;
        double gbps = data_gb / seconds;

        std::cout << "Read completed:" << std::endl;
        std::cout << "- Time: " << format_duration(read_duration) << std::endl;
        std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB"
                  << std::endl;
        std::cout << "- Speed: " << gbps << " GB/s" << std::endl;

        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 6: Validating read data" << std::endl;
        std::cout << "============================================================" << std::endl;
        for (i = 0; i < num_transfers; i++) {
            char *expected_buffer = (char *)malloc(transfer_size);
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
            printProgress(float(i + 1) / num_transfers);
        }
        std::cout << "\nVerification completed successfully!" << std::endl;
    }

cleanup:
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 6: Cleanup" << std::endl;
    std::cout << "============================================================" << std::endl;

    printProgress(1.0);

    // Cleanup resources
    agent.deregisterMem(obj_for_obj);
    agent.deregisterMem(dram_for_obj);
    for (i = 0; i < num_transfers; i++) {
        if (dram_addr[i]) free(dram_addr[i]);
    }
    delete[] dram_addr;
    delete[] dram_buf;

    delete[] objects;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "                    TEST SUMMARY                             " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Total time: " << format_duration(total_time) << std::endl;
    std::cout << "Total data: " << std::fixed << std::setprecision(2) << total_data_gb << " GB"
              << std::endl;
    std::cout << "============================================================" << std::endl;
    return 0;
}
