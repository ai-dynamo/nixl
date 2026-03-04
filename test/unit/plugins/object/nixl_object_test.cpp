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
/**
 * @brief Parse size strings with suffixes like "1K", "2M", "3G"
 * @param size_str The size string to parse (e.g., "16M", "1G", "1024K")
 * @return The parsed size in bytes, or 0 if invalid format
 */
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

/**
 * @brief Print usage information for the program
 * @param program_name The name of the program executable
 */
void
print_usage(const char *program_name) {
    std::cerr << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -n, --num-transfers N   Number of transfers to perform (default: "
              << DEFAULT_NUM_TRANSFERS << ")\n"
              << "  -s, --size SIZE         Size of each transfer (default: "
              << DEFAULT_TRANSFER_SIZE << " bytes)\n"
              << "                          Can use K, M, or G suffix (e.g., 1K, 2M, 3G)\n"
              << "  -t, --iterations N      Number of iterations for each transfer (default: "
              << DEFAULT_ITERATIONS << ")\n"
              << "  -e, --endpoint ENDPOINT S3 Endpoint URL\n"
              << "  -u, --bucket BUCKET     S3 Bucket name\n"
              << "  -h, --help              Show this help message\n"
              << "\nExamples:\n"
              << "  " << program_name << " -n 100 -s 16M -t 5\n"
              << "  " << program_name
              << " -n 100 -s 16M -t 5 -e http://s3.example.com:9000 -u my-bucket\n";
}

/**
 * @brief Print a progress bar to the console
 * @param progress Progress value between 0.0 and 1.0
 */
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

/**
 * @brief Generate a timestamped prefix for object keys
 * @param base_name Base name for the object key
 * @return Timestamped prefix string
 */
std::string
generate_timestamped_object_prefix(const std::string &base_name) {
    std::time_t t = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", std::localtime(&t));
    return base_name + std::string(timestamp);
}

// Helper function to fill buffer with repeating pattern
/**
 * @brief Fill buffer with a repeating test pattern
 * @param buffer Pointer to the buffer to fill
 * @param size Size of the buffer in bytes
 */
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

/**
 * @brief Clear a buffer by setting all bytes to zero
 * @param buffer Pointer to the buffer to clear
 * @param size Size of the buffer in bytes
 */
void
clear_buffer(void *buffer, size_t size) {
    memset(buffer, 0, size);
}

// Helper function to format duration
/**
 * @brief Format duration in microseconds to a human-readable string
 * @param us Duration in microseconds
 * @return Formatted duration string (e.g., "500 ms", "1.234 sec")
 */
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

/**
 * @brief Main function for NIXL object storage performance testing
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Exit code (0 for success, non-zero for failure)
 */
int
main(int argc, char *argv[]) {
    nixl_status_t ret = NIXL_SUCCESS;
    void **dram_addr = NULL;
    int status = 0;
    int i;
    int opt;
    size_t transfer_size = DEFAULT_TRANSFER_SIZE;
    int num_transfers = DEFAULT_NUM_TRANSFERS;
    nixlTime::us_t total_time(0);
    nixlTime::us_t reg_time(0);
    double total_data_gb = 0;
    int iterations = DEFAULT_ITERATIONS;
    std::string endpoint;
    std::string bucket;
    int ret_code = 0;
    nixlXferReqH *write_req = nullptr;
    nixlXferReqH *read_req = nullptr;
    bool obj_registered = false;
    bool dram_registered = false;

    // Parse command line options
    static struct option long_options[] = {{"num-transfers", required_argument, 0, 'n'},
                                           {"size", required_argument, 0, 's'},
                                           {"iterations", required_argument, 0, 't'},
                                           {"endpoint", required_argument, 0, 'e'},
                                           {"bucket", required_argument, 0, 'u'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "n:s:t:he:u:", long_options, NULL)) != -1) {
        switch (opt) {
        case 'e':
            endpoint = optarg;
            break;
        case 'u':
            bucket = optarg;
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
        case 't': {
            int parsed = atoi(optarg);
            if (parsed <= 0) {
                std::cerr << "Error: Number of iterations must be positive\n";
                return 1;
            }
            iterations = parsed;
            break;
        }
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }


    // Allocate DRAM array
    dram_addr = new void *[num_transfers]();

    // Initialize NIXL components
    nixlAgentConfig cfg(true);
    nixlBlobDesc *dram_buf = new nixlBlobDesc[num_transfers];
    nixlBlobDesc *objects = new nixlBlobDesc[num_transfers];
    nixlBackendH *obj;
    nixl_reg_dlist_t dram_for_obj(DRAM_SEG);
    nixl_reg_dlist_t obj_for_obj(OBJ_SEG);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "                 NIXL STORAGE TEST STARTING (OBJ PLUGIN)                     "
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
    std::cout << "- Operation: Read and Write" << std::endl;
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
        ret_code = 1;
        goto cleanup;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 1: Allocating and initializing buffers" << std::endl;
    std::cout << "============================================================" << std::endl;

    std::string object_prefix = generate_timestamped_object_prefix("test-key-");
    for (i = 0; i < num_transfers; i++) {
        // Allocate and initialize DRAM buffer
        if (posix_memalign(&dram_addr[i], PAGE_SIZE, transfer_size) != 0) {
            std::cerr << "DRAM allocation failed\n";
            ret_code = 1;
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
        objects[i].metaInfo = object_prefix + "-" + std::to_string(i);
        obj_for_obj.addDesc(objects[i]);

        printProgress(float(i + 1) / num_transfers);
    }
    using namespace nixlTime;
    us_t reg_start = getUs();

    std::cout << "\n=== Registering memory ===" << std::endl;

    ret = agent.registerMem(obj_for_obj);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register file memory\n";
        ret_code = 1;
        goto cleanup;
    }
    obj_registered = true;

    ret = agent.registerMem(dram_for_obj);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register DRAM memory\n";
        ret_code = 1;
        goto cleanup;
    }
    dram_registered = true;

    us_t reg_end = getUs();

    reg_time = (reg_end - reg_start);

    std::cout << "Registration completed:" << std::endl;
    std::cout << "- Time: " << format_duration(reg_time) << std::endl;


    // Perform write test
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 2: Memory to Object Transfer (Write Test)" << std::endl;
    std::cout << "============================================================" << std::endl;

    us_t write_duration(0);

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
        ret_code = 1;
        goto cleanup;
    }
    std::cout << "Write transfer request created." << std::endl;

    // Now do the iterations
    for (int iter = 0; iter < iterations; iter++) {
        us_t iter_start = getUs();

        status = agent.postXferReq(write_req);
        if (status < 0) {
            std::cerr << "Failed to post write transfer request" << std::endl;
            ret_code = 1;
            goto cleanup;
        }

        // Wait for completion
        while (status == NIXL_IN_PROG) {
            status = agent.getXferStatus(write_req);
            if (status < 0) {
                std::cerr << "Error during write transfer" << std::endl;
                ret_code = 1;
                goto cleanup;
            }
        }

        us_t iter_end = getUs();
        write_duration += (iter_end - iter_start);

        if (iterations > 1) {
            printProgress(float(iter + 1) / iterations);
        }
    }

    total_time += write_duration;

    double data_gb = (transfer_size * num_transfers * iterations) / (1024.0 * 1024.0 * 1024.0);
    total_data_gb += data_gb;
    double seconds = write_duration / 1000000.0;
    double gbps = data_gb / seconds;

    std::cout << "Write completed:" << std::endl;
    std::cout << "- Time: " << format_duration(write_duration) << std::endl;
    std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB" << std::endl;
    std::cout << "- Speed: " << gbps << " GB/s" << std::endl;

    // Clear buffers before read test
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 3: Clearing buffers for read test" << std::endl;
    std::cout << "============================================================" << std::endl;
    for (i = 0; i < num_transfers; i++) {
        clear_buffer(dram_addr[i], transfer_size);
        printProgress(float(i + 1) / num_transfers);
    }

    // Create extra params with backend
    nixl_opt_args_t extra_params;
    extra_params.backends = {obj};
    std::vector<nixl_query_resp_t> resp;
    status = agent.queryMem(obj_for_obj, resp, &extra_params);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Failed to query object memory status\n";
        ret_code = 1;
        goto cleanup;
    }
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 4: Querying Objects" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "\nQueryMem Results:" << std::endl;
    std::cout << "response count: " << resp.size() << std::endl;
    if (resp.size() != static_cast<size_t>(num_transfers)) {
        std::cerr << "Error: Expected " << num_transfers << " responses, got " << resp.size()
                  << std::endl;
        ret_code = 1;
        goto cleanup;
    }
    for (const auto &r : resp) {
        if (!r.has_value()) {
            std::cerr << "Error: QueryMem response has no value\n";
            ret_code = 1;
            goto cleanup;
        }
    }
    std::cout << "All queried objects are valid." << std::endl;


    // Perform read test
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 5: Object to Memory Transfer (Read Test)" << std::endl;
    std::cout << "============================================================" << std::endl;

    us_t read_duration(0);

    // Create descriptor lists for all transfers
    nixl_reg_dlist_t read_src_reg(DRAM_SEG);
    nixl_reg_dlist_t read_obj_reg(OBJ_SEG);

    // Add all descriptors
    for (int transfer_idx = 0; transfer_idx < num_transfers; transfer_idx++) {
        read_src_reg.addDesc(dram_buf[transfer_idx]);
        read_obj_reg.addDesc(objects[transfer_idx]);
        printProgress(float(transfer_idx + 1) / num_transfers);
    }
    std::cout << "\nAll descriptors added." << std::endl;

    // Create transfer lists
    nixl_xfer_dlist_t read_src_list = read_src_reg.trim();
    nixl_xfer_dlist_t read_obj_list = read_obj_reg.trim();

    // Create single transfer request for all transfers
    ret = agent.createXferReq(NIXL_READ, read_src_list, read_obj_list, "ObjTester", read_req);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to create read transfer request" << std::endl;
        ret_code = 1;
        goto cleanup;
    }
    std::cout << "Read transfer request created." << std::endl;

    // Now do the iterations
    for (int iter = 0; iter < iterations; iter++) {
        us_t iter_start = getUs();

        status = agent.postXferReq(read_req);
        if (status < 0) {
            std::cerr << "Failed to post read transfer request" << std::endl;
            ret_code = 1;
            goto cleanup;
        }

        // Wait for completion
        while (status == NIXL_IN_PROG) {
            status = agent.getXferStatus(read_req);
            if (status < 0) {
                std::cerr << "Error during read transfer" << std::endl;
                ret_code = 1;
                goto cleanup;
            }
        }

        us_t iter_end = getUs();
        read_duration += (iter_end - iter_start);

        if (iterations > 1) {
            printProgress(float(iter + 1) / iterations);
        }
    }

    total_time += read_duration;

    double read_data_gb = (transfer_size * num_transfers * iterations) / (1024.0 * 1024.0 * 1024.0);
    total_data_gb += read_data_gb;
    double read_seconds = read_duration / 1000000.0;
    double read_gbps = read_data_gb / read_seconds;

    std::cout << "Read completed:" << std::endl;
    std::cout << "- Time: " << format_duration(read_duration) << std::endl;
    std::cout << "- Data: " << std::fixed << std::setprecision(2) << read_data_gb << " GB"
              << std::endl;
    std::cout << "- Speed: " << read_gbps << " GB/s" << std::endl;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 6: Validating read data" << std::endl;
    std::cout << "============================================================" << std::endl;
    char *expected_buffer = (char *)malloc(transfer_size);

    for (i = 0; i < num_transfers; i++) {
        if (!expected_buffer) {
            std::cerr << "Failed to allocate validation buffer\n";
            ret_code = 1;
            goto cleanup;
        }
        fill_test_pattern(expected_buffer, transfer_size);
        if (memcmp(dram_addr[i], expected_buffer, transfer_size) != 0) {
            std::cerr << "DRAM buffer " << i << " validation failed\n";
            free(expected_buffer);
            ret_code = 1;
            goto cleanup;
        }

        printProgress(float(i + 1) / num_transfers);
    }
    std::cout << "\nVerification completed successfully!" << std::endl;


cleanup:
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 7: Cleanup" << std::endl;
    std::cout << "============================================================" << std::endl;

    printProgress(1.0);

    // Cleanup transfer requests
    if (write_req) {
        agent.releaseXferReq(write_req);
    }
    if (read_req) {
        agent.releaseXferReq(read_req);
    }

    // Cleanup resources
    free(expected_buffer);

    if (obj_registered) {
        agent.deregisterMem(obj_for_obj);
        obj_registered = false;
    }
    if (dram_registered) {
        agent.deregisterMem(dram_for_obj);
        dram_registered = false;
    }
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
    return ret_code;
}
