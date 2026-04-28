/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DataDirect Networks, Inc.
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
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include "common/nixl_time.h"
#include <cassert>
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <cstring>
#include <getopt.h>
#include <random>
#include <fstream>
#include <sys/sysinfo.h>

using namespace nixlTime;

// Default values
#define DEFAULT_NUM_TRANSFERS 250
#define DEFAULT_TRANSFER_SIZE (10 * 1024 * 1024)  // 10MB
#define DEFAULT_ITERATIONS 1  // Default number of iterations
#define TEST_PHRASE "NIXL Storage Test Pattern 2025."
#define TEST_PHRASE_LEN (sizeof(TEST_PHRASE) - 1)  // -1 to exclude null

#define KEY_SIZE      32    // Use UUID

// Get system page size
static size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);

// Progress bar configuration
#define PROGRESS_WIDTH 50

// Forward declaration of format_data_size (defined later)
std::string format_data_size(size_t bytes);

// Helper structure to hold system memory information
struct MemoryInfo {
    size_t total_ram;      // Total physical RAM in bytes
    size_t available_ram;  // Available RAM in bytes
    size_t free_ram;       // Free RAM in bytes
    size_t buffers;        // Buffer cache in bytes
    size_t cached;         // Page cache in bytes
};

// Get system memory information from /proc/meminfo
bool get_memory_info(MemoryInfo& info) {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        // Fallback to sysinfo if /proc/meminfo is not available
        struct sysinfo si;
        if (sysinfo(&si) != 0) {
            return false;
        }
        info.total_ram = si.totalram * si.mem_unit;
        info.free_ram = si.freeram * si.mem_unit;
        info.available_ram = info.free_ram;
        info.buffers = si.bufferram * si.mem_unit;
        info.cached = 0;
        return true;
    }

    // Parse /proc/meminfo for more accurate information
    std::string line;
    info.total_ram = 0;
    info.available_ram = 0;
    info.free_ram = 0;
    info.buffers = 0;
    info.cached = 0;

    while (std::getline(meminfo, line)) {
        size_t value;
        if (sscanf(line.c_str(), "MemTotal: %zu kB", &value) == 1) {
            info.total_ram = value * 1024;
        } else if (sscanf(line.c_str(), "MemAvailable: %zu kB", &value) == 1) {
            info.available_ram = value * 1024;
        } else if (sscanf(line.c_str(), "MemFree: %zu kB", &value) == 1) {
            info.free_ram = value * 1024;
        } else if (sscanf(line.c_str(), "Buffers: %zu kB", &value) == 1) {
            info.buffers = value * 1024;
        } else if (sscanf(line.c_str(), "Cached: %zu kB", &value) == 1) {
            info.cached = value * 1024;
        }
    }

    // If MemAvailable is not present (older kernels), estimate it
    if (info.available_ram == 0) {
        info.available_ram = info.free_ram + info.buffers + info.cached;
    }

    return info.total_ram > 0;
}

// Check if there's enough memory available for the test
// Returns true if sufficient memory, false otherwise
bool check_memory_requirements(size_t required_bytes, double safety_factor = 0.9) {
    MemoryInfo mem_info;
    if (!get_memory_info(mem_info)) {
        std::cerr << "Warning: Could not retrieve system memory information\n";
        std::cerr << "Proceeding anyway, but be aware of potential OOM issues\n";
        return true;  // Proceed with caution if we can't check
    }

    // Use available memory as the most accurate measure
    size_t usable_memory = static_cast<size_t>(mem_info.available_ram * safety_factor);

    std::cout << "\n=== Memory Check ===" << std::endl;
    std::cout << "- Total RAM:       " << format_data_size(mem_info.total_ram) << std::endl;
    std::cout << "- Available RAM:   " << format_data_size(mem_info.available_ram) << std::endl;
    std::cout << "- Required:        " << format_data_size(required_bytes) << std::endl;
    std::cout << "- Usable (90%):    " << format_data_size(usable_memory) << std::endl;

    if (required_bytes > usable_memory) {
        std::cerr << "\n*** ERROR: Insufficient memory! ***\n";
        std::cerr << "Required:  " << format_data_size(required_bytes) << std::endl;
        std::cerr << "Available: " << format_data_size(usable_memory) << " (90% of available RAM)\n";
        std::cerr << "\nSuggestions:\n";
        std::cerr << "1. Reduce number of transfers (-n flag)\n";
        std::cerr << "2. Reduce transfer size (-s flag)\n";
        std::cerr << "3. Close other applications to free memory\n";
        std::cerr << "4. Add more RAM to the system\n";
        return false;
    }

    std::cout << "- Status:          OK (sufficient memory)\n";
    return true;
}

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
    std::cerr << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
#ifdef HAVE_CUDA
              << "  -d, --dram                  Use DRAM for memory operations\n"
              << "  -v, --vram                  Use VRAM for memory operations (default)\n"
#else
              << "  -d, --dram                  Use DRAM for memory operations (default)\n"
#endif
              << "  -n, --num-transfers N       Number of transfers to perform (default: " << DEFAULT_NUM_TRANSFERS << ")\n"
              << "  -s, --size SIZE             Size of each transfer (default: " << DEFAULT_TRANSFER_SIZE << " bytes)\n"
              << "                              Can use K, M, or G suffix (e.g., 1K, 2M, 3G)\n"
              << "  -r, --no-read               Skip read test\n"
              << "  -w, --no-write              Skip write test\n"
              << "  -t, --iterations N          Number of iterations for each transfer (default: " << DEFAULT_ITERATIONS << ")\n"
              << "\n  INFINIA Backend Parameters:\n"
              << "  -T, --sthreads N            Number of RED FS service threads (default: 8, range: 1-64)\n"
              << "  -B, --num-buffers N         Pre-allocated buffers for async ops (default: 512, range: 1-4096)\n"
              << "  -R, --num-ring-entries N    Depth of async I/O ring buffer (default: 512, range: 1-4096)\n"
              << "  -C, --coremask MASK         CPU affinity: hex (\"0x0F\") or list (\"0-3,8\") (default: \"0x2\")\n"
              << "  -M, --max-retries N         Max retry attempts for operations (default: 3, range: 0-100)\n"
              << "\n  Other:\n"
              << "  -S, --seed N                Random seed for reproducibility (default: 0, use -1 to skip validation).\n"
              << "  -h, --help                  Show this help message\n"
              << "\nExample:\n"
              << "  " << program_name << " -d -n 100 -s 2M -t 5 -T 16 -B 1024 -R 1024\n";
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

void validateBuffer(void* expected, void* actual, size_t size, const char* operation) {
    if (memcmp(expected, actual, size) != 0) {
        std::cerr << "Data validation failed for " << operation << std::endl;
        exit(-1);
    }
}

// Global random number generator
std::random_device random_device;
std::mt19937 generator;

// Helper function to fill buffer with repeating pattern
void fill_test_pattern(unsigned char* buffer, size_t size) {
#if 0
    char* buf = (char*)buffer;
    size_t phrase_len = TEST_PHRASE_LEN;
    std::uniform_int_distribution<> distribution(0, TEST_PHRASE_LEN);
    std::string random_string;
    size_t offset = 0;
    size_t remaining = distribution(generator);
    remaining = (remaining < size) ? remaining : size;

    while (offset < size) {
        size_t copy_len = (remaining < phrase_len) ? remaining : phrase_len;
        remaining = size - offset - copy_len;
        memcpy(buf + offset, TEST_PHRASE, copy_len);
        offset += copy_len;
    }
#else
    std::uniform_int_distribution<> distribution(0, 255);

    for (size_t i = 0; i < size; ++i) {
        buffer[i] = distribution(generator);
    }
#endif
}

void clear_buffer(void* buffer, size_t size) {
    memset(buffer, 0, size);
}

#ifdef HAVE_CUDA
// Helper function to fill GPU buffer with repeating pattern
cudaError_t fill_gpu_test_pattern(void* gpu_buffer, size_t size) {
    unsigned char* host_buffer = (unsigned char*)malloc(size);
    if (!host_buffer) {
        return cudaErrorMemoryAllocation;
    }

    fill_test_pattern(host_buffer, size);
    cudaError_t err = cudaMemcpy(gpu_buffer, host_buffer, size, cudaMemcpyHostToDevice);
    free(host_buffer);
    return err;
}

cudaError_t clear_gpu_buffer(void* gpu_buffer, size_t size) {
    return cudaMemset(gpu_buffer, 0, size);
}

// Helper function to validate GPU buffer
bool validate_gpu_buffer(void* gpu_buffer, size_t size) {
    unsigned char* host_buffer = (unsigned char*)malloc(size);
    unsigned char* expected_buffer = (unsigned char*)malloc(size);
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
#endif

bool fill_test_key(char *key) {
    std::uniform_int_distribution<unsigned char> distribution(0, 255);

    std::stringstream output;
    for (size_t i = 0; i < 32; ++i) {
        output << std::setw(2) << std::setfill('0') << static_cast<int>(distribution(generator));
    }

    strncpy(key, output.str().c_str(), 32);
    key[32] = '\0';

    return true;
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

// Helper function to format data size in MiB
// Always uses MiB for consistency (1 MiB = 1024 * 1024 bytes)
std::string format_data_size(size_t bytes) {
    double mib = bytes / (1024.0 * 1024.0);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << mib << " MiB";
    return ss.str();
}

// Helper function to format throughput in MiB/s
std::string format_throughput(size_t bytes, nixlTime::us_t duration_us) {
    if (duration_us == 0) {
        return "0.00 MiB/s";
    }
    double seconds = duration_us / 1000000.0;
    double mib = bytes / (1024.0 * 1024.0);
    double mibps = mib / seconds;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << mibps << " MiB/s";
    return ss.str();
}

int main(int argc, char *argv[])
{
    nixl_status_t           ret = NIXL_SUCCESS;
#ifdef HAVE_CUDA
    void                    **vram_addr = NULL;
#endif
    void                    **dram_addr = NULL;
    char                    **test_keys = NULL;
    std::string             role;
    int                     status = 0;
    int                     i;
    bool                    use_dram = false;
    bool                    use_vram = false;
    int                     opt;
    std::string             dir_path;
    size_t                  transfer_size = DEFAULT_TRANSFER_SIZE;
    int                     num_transfers = DEFAULT_NUM_TRANSFERS;
    bool                    skip_read = false;
    bool                    skip_write = false;
    unsigned int            sthreads = 8;
    unsigned int            num_buffers = 512;
    unsigned int            num_ring_entries = 512;
    std::string             coremask = "0x2";
    unsigned int            max_retries = 3;
    nixlTime::us_t          total_time(0);
    nixlTime::us_t          alloc_duration(0);
    nixlTime::us_t          write_duration_total(0);
    nixlTime::us_t          query_duration_total(0);
    nixlTime::us_t          clear_duration_total(0);
    nixlTime::us_t          read_duration_total(0);
    nixlTime::us_t          validate_duration_total(0);
    nixlTime::us_t          cleanup_duration_total(0);
    double                  total_data_gb = 0;
    bool                    use_direct __attribute__((unused)) = false;
    unsigned int            iterations = DEFAULT_ITERATIONS;
    int                     seed = 0;
    // -1: arg error, -2: creatBackend, -3: initialize, -4: registerMem
    // -5: createXferReq, -6: postXferReq, -7: getXferStatus, -15: validate
    int                     rc = 0;
    nixlXferReqH*           write_req = nullptr;
    nixlXferReqH*           read_req = nullptr;

    // Parse command line options
    static struct option long_options[] = {
        {"dram",            no_argument,       0, 'd'},
        {"vram",            no_argument,       0, 'v'},
        {"num-transfers",   required_argument, 0, 'n'},
        {"size",           required_argument, 0, 's'},
        {"no-read",        no_argument,       0, 'r'},
        {"no-write",       no_argument,       0, 'w'},
        {"sthreads",       required_argument, 0, 'T'},
        {"num-buffers",    required_argument, 0, 'B'},
        {"num-ring-entries", required_argument, 0, 'R'},
        {"coremask",       required_argument, 0, 'C'},
        {"max-retries",    required_argument, 0, 'M'},
        {"iterations",     required_argument, 0, 't'},
        {"direct",         no_argument,       0, 'D'},
        {"seed",           required_argument, 0, 'S'},
        {"help",           no_argument,       0, 'h'},
        {0,                0,                 0,  0}
    };

#ifdef HAVE_CUDA
    while ((opt = getopt_long(argc, argv, "dvn:s:rwT:B:R:C:M:t:DS:h", long_options, NULL)) != -1) {
#else
    while ((opt = getopt_long(argc, argv, "dn:s:rwT:B:R:C:M:t:DS:h", long_options, NULL)) != -1) {
#endif
        switch (opt) {
            case 'd':
                use_dram = true;
                break;
#ifdef HAVE_CUDA
            case 'v':
                use_vram = true;
                break;
#endif
            case 'n':
                num_transfers = atoi(optarg);
                if (num_transfers <= 0) {
                    std::cerr << "Error: Number of transfers must be positive\n";
                    return -1;
                }
                break;
            case 's':
                transfer_size = parse_size(optarg);
                if (transfer_size == 0) {
                    std::cerr << "Error: Invalid transfer size format\n";
                    return -1;
                }
                // TODO set max transfer_size to 10M at now
                if (transfer_size > 10*1024*1024) {
                    transfer_size = 10*1024*1024;
                    std::cout << "Warning: set transfer_size to 10M\n";
                }
                break;
            case 'r':
                skip_read = true;
                break;
            case 'w':
                skip_write = true;
                break;
            case 'T':
                sthreads = atoi(optarg);
                if (sthreads < 1 || sthreads > 64) {
                    std::cerr << "Error: Service threads must be between 1 and 64\n";
                    return -1;
                }
                break;
            case 'B':
                num_buffers = atoi(optarg);
                if (num_buffers < 1 || num_buffers > 4096) {
                    std::cerr << "Error: Number of buffers must be between 1 and 4096\n";
                    return -1;
                }
                break;
            case 'R':
                num_ring_entries = atoi(optarg);
                if (num_ring_entries < 1 || num_ring_entries > 4096) {
                    std::cerr << "Error: Number of ring entries must be between 1 and 4096\n";
                    return -1;
                }
                break;
            case 'C':
                coremask = optarg;
                break;
            case 'M':
                max_retries = atoi(optarg);
                if (max_retries < 0 || max_retries > 100) {
                    std::cerr << "Error: Max retries must be between 0 and 100\n";
                    return -1;
                }
                break;
            case 't':
                iterations = atoi(optarg);
                if (iterations <= 0) {
                    std::cerr << "Error: Number of iterations must be positive\n";
                    return -1;
                }
                break;
            case 'D':
                use_direct = true;
                break;
            case 'S':
                seed = atoi(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
        }
    }

    if (skip_read && skip_write) {
        std::cerr << "Error: Cannot skip both read and write tests\n";
        return -1;
    }

    // If neither is specified, default to VRAM
    if (!use_dram && !use_vram) {
#ifdef HAVE_CUDA
        use_vram = true;
#else
        use_dram = true;
#endif
    }

    // Check if both DRAM and VRAM are specified
    if (use_dram && use_vram) {
        std::cerr << "Error: Cannot specify both DRAM (-d) and VRAM (-v) options\n";
        print_usage(argv[0]);
        return -1;
    }

    // Allocate arrays based on num_transfers
    if (use_vram) {
#ifdef HAVE_CUDA
        vram_addr = new void*[num_transfers]();
#endif
    }
    if (use_dram) {
        dram_addr = new void*[num_transfers]();
    }

    test_keys = new char*[num_transfers]();

    // Initialize NIXL components
    nixlAgentConfig             cfg(true);
    nixl_b_params_t             params;
    nixlBlobDesc                *vram_buf = use_vram ? new nixlBlobDesc[num_transfers] : NULL;
    nixlBlobDesc                *dram_buf = use_dram ? new nixlBlobDesc[num_transfers] : NULL;
    nixlBlobDesc                *ftrans = new nixlBlobDesc[num_transfers];
    // TODO: rename infinia to something?
    nixlBackendH                *infinia;
    nixl_reg_dlist_t            vram_for_infinia(VRAM_SEG);
    nixl_reg_dlist_t            dram_for_infinia(DRAM_SEG);
    nixl_reg_dlist_t            obj_for_infinia(OBJ_SEG);
    std::string                 name;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "               NIXL STORAGE TEST STARTING (INFINIA PLUGIN)                   " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Mode: " << (use_dram ? "DRAM" : "VRAM") << std::endl;
    std::cout << "- Number of transfers: " << num_transfers << std::endl;
    std::cout << "- Transfer size: " << transfer_size << " bytes" << std::endl;
    std::cout << "- Total data: " << format_data_size(transfer_size * num_transfers) << std::endl;
    std::cout << "- Number of iterations: " << iterations << std::endl;
    std::cout << "- Operation: ";
    if (!skip_read && !skip_write) {
      std::cout << "Read and Write";
    } else if (skip_read) {
      std::cout << "Write Only";
    } else {  // skip_write
      std::cout << "Read Only";
    }
    std::cout << std::endl;
    std::cout << "\nINFINIA Backend Configuration:" << std::endl;
    std::cout << "- Service threads: " << sthreads << std::endl;
    std::cout << "- Async buffers: " << num_buffers << std::endl;
    std::cout << "- Ring entries: " << num_ring_entries << std::endl;
    std::cout << "- Core mask: " << coremask << std::endl;
    std::cout << "- Max retries: " << max_retries << std::endl;
    std::cout << "============================================================\n" << std::endl;

    // Check memory requirements before starting
    if (use_dram) {
        // Estimate total memory needed:
        // - Main buffers: num_transfers * transfer_size
        // - Backend internal buffers: num_buffers * transfer_size (estimate)
        // - Overhead for descriptors, keys, etc: ~10%
        size_t buffer_memory = static_cast<size_t>(num_transfers) * transfer_size;
        size_t backend_memory = static_cast<size_t>(num_buffers) * transfer_size;
        size_t overhead = (buffer_memory + backend_memory) / 10;
        size_t total_required = buffer_memory + backend_memory + overhead;

        if (!check_memory_requirements(total_required)) {
            std::cerr << "\nTest aborted due to insufficient memory.\n";
            return -1;
        }
        std::cout << std::endl;
    }

    nixlAgent agent("INFINIA_Tester", cfg);

    // Set INFINIA backend parameters
    params["sthreads"] = std::to_string(sthreads);
    params["num_buffers"] = std::to_string(num_buffers);
    params["num_ring_entries"] = std::to_string(num_ring_entries);
    params["coremasks"] = coremask;
    params["max_retries"] = std::to_string(max_retries);

    // To also test the decision making of createXferReq
    ret = agent.createBackend("INFINIA", params, infinia);

    if (ret != NIXL_SUCCESS || infinia == NULL) {
      std::cerr << "Error creating Infinia backend: "
                << (ret != NIXL_SUCCESS ? "Failed to create backend" : "Backend handle is NULL")
                << std::endl;
      rc = -2;
      goto cleanup;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 1: Allocating and initializing buffers" << std::endl;
    std::cout << "============================================================" << std::endl;

    us_t alloc_start = getUs();

    // Seed the generator for the Key
    generator.seed(seed == -1? random_device() : (seed ^ 0x12345));

    for (i = 0; i < num_transfers; i++) {
        // set keys
        test_keys[i] = new char[KEY_SIZE + 1];
        if ( !fill_test_key(test_keys[i])) {
            rc = -3;
            goto cleanup;
        }
    }

    // Seed the global generator
    generator.seed(seed == -1? random_device() : seed);

    for (i = 0; i < num_transfers; i++) {
#ifdef HAVE_CUDA
        if (use_vram) {
            // Allocate and initialize VRAM buffer
            cudaError_t cuda_err = cudaMalloc(&vram_addr[i], transfer_size);
            if (cuda_err != cudaSuccess) {
                std::cerr << "\n*** CUDA malloc failed at buffer " << i << "/" << num_transfers << " ***\n";
                std::cerr << "Error: " << cudaGetErrorString(cuda_err) << "\n";
                std::cerr << "Successfully allocated " << i << " buffers of "
                          << format_data_size(transfer_size) << " each\n";
                std::cerr << "Total allocated before failure: " << format_data_size(i * transfer_size) << "\n";
                std::cerr << "\nThis typically indicates:\n";
                std::cerr << "1. Insufficient GPU memory\n";
                std::cerr << "2. GPU memory fragmentation\n";
                std::cerr << "\nPlease reduce test parameters and try again.\n";
                rc = -3;
                goto cleanup;
            }
            cuda_err = fill_gpu_test_pattern(vram_addr[i], transfer_size);
            if (cuda_err != cudaSuccess) {
                std::cerr << "\n*** CUDA buffer initialization failed at buffer " << i << " ***\n";
                std::cerr << "Error: " << cudaGetErrorString(cuda_err) << "\n";
                rc = -3;
                goto cleanup;
            }
        }
#endif

        if (use_dram) {
            // Allocate and initialize DRAM buffer
            int alloc_result = posix_memalign(&dram_addr[i], PAGE_SIZE, transfer_size);
            if (alloc_result != 0) {
                std::cerr << "\n*** DRAM allocation failed at buffer " << i << "/" << num_transfers << " ***\n";
                std::cerr << "Error: " << strerror(alloc_result) << " (errno=" << alloc_result << ")\n";
                std::cerr << "Successfully allocated " << i << " buffers of "
                          << format_data_size(transfer_size) << " each\n";
                std::cerr << "Total allocated before failure: " << format_data_size(i * transfer_size) << "\n";
                std::cerr << "\nThis typically indicates:\n";
                std::cerr << "1. Insufficient system memory (OOM condition)\n";
                std::cerr << "2. Memory fragmentation preventing large allocations\n";
                std::cerr << "3. System limits (ulimit) restricting memory usage\n";
                std::cerr << "\nPlease reduce test parameters and try again.\n";
                rc = -3;
                goto cleanup;
            }
            fill_test_pattern((unsigned char*)dram_addr[i], transfer_size);
        }

        // Set up descriptors
#ifdef HAVE_CUDA
        if (use_vram) {
            vram_buf[i].addr   = (uintptr_t)(vram_addr[i]);
            vram_buf[i].len    = transfer_size;
            vram_buf[i].devId  = 0;
            vram_for_infinia.addDesc(vram_buf[i]);
        }
#endif

        if (use_dram) {
            dram_buf[i].addr   = (uintptr_t)(dram_addr[i]);
            dram_buf[i].len    = transfer_size;
            dram_buf[i].devId  = 0;
            dram_for_infinia.addDesc(dram_buf[i]);
        }

        // For OBJ_SEG: metaInfo contains the key, devId is unique identifier, addr is offset
        ftrans[i].addr     = 0;                         // Object offset (0 = whole object)
        ftrans[i].len      = transfer_size;             // Object size
        ftrans[i].devId    = i;                         // Unique ID per KV pair
        ftrans[i].metaInfo = std::string(test_keys[i]); // ACTUAL KEY STRING
        obj_for_infinia.addDesc(ftrans[i]);

        printProgress(float(i + 1) / num_transfers);
    }

    std::cout << "\n=== Registering memory ===" << std::endl;
    ret = agent.registerMem(obj_for_infinia);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register file memory\n";
        rc = -4;
        goto cleanup;
    }

#ifdef HAVE_CUDA
    if (use_vram) {
        ret = agent.registerMem(vram_for_infinia);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register VRAM memory\n";
            rc = -4;
            goto cleanup;
        }
    }
#endif

    if (use_dram) {
        ret = agent.registerMem(dram_for_infinia);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM memory\n";
            rc = -4;
            goto cleanup;
        }
    }

    us_t alloc_end = getUs();
    alloc_duration = alloc_end - alloc_start;
    std::cout << "- Time:         " << format_duration(alloc_duration) << std::endl;
    total_time += alloc_duration;

    // Perform write test if not skipped
    if (!skip_write) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 2: Memory to Object Transfer (Write Test)" << std::endl;
        std::cout << "============================================================" << std::endl;

        us_t write_duration(0);

        // Create descriptor lists for all transfers
        nixl_reg_dlist_t src_reg(use_dram ? DRAM_SEG : VRAM_SEG);
        nixl_reg_dlist_t obj_reg(OBJ_SEG);

        // Add all descriptors
        for (int transfer_idx = 0; transfer_idx < num_transfers; transfer_idx++) {
            if (use_dram) {
                src_reg.addDesc(dram_buf[transfer_idx]);
            } else {
                src_reg.addDesc(vram_buf[transfer_idx]);
            }
            obj_reg.addDesc(ftrans[transfer_idx]);
            printProgress(float(transfer_idx + 1) / num_transfers);
        }
        std::cout << "\nAll descriptors added." << std::endl;

        // Create transfer lists
        nixl_xfer_dlist_t src_list = src_reg.trim();
        nixl_xfer_dlist_t obj_list = obj_reg.trim();

        // Create single transfer request for all transfers
        ret = agent.createXferReq(NIXL_WRITE, src_list, obj_list,
                                  "INFINIA_Tester", write_req);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to create write transfer request " << ret << std::endl;
            rc = -5;
            goto cleanup;
        }
        std::cout << "Write transfer request created." << std::endl;

        std::cout << " Post NIXL_WRITE request to Infinia KV dataset (INFINIA backend)\n";
        for (unsigned int iter = 0; iter < iterations; iter++) {
            us_t iter_start = getUs();

            status = agent.postXferReq(write_req);
            if (status < 0) {
                std::cerr << "Failed to post write transfer request " << status << std::endl;
                rc = -6;
                goto cleanup;
            }

            std::cout << " Infinia NIXL_WRITE has been posted\n";
            std::cout << " Waiting for completion\n";

            while (status == NIXL_IN_PROG) {
                status = agent.getXferStatus(write_req);
                if (status < 0) {
                    std::cerr << "Error during write transfer " << status << std::endl;
                    rc = -7;
                    goto cleanup;
                }
            }

            us_t iter_end = getUs();
            write_duration += (iter_end - iter_start);

            if (iterations > 1) {
                printProgress(float(iter + 1) / iterations);
            }
        }

        std::cout <<" Completed writing data to Infinia KV dataset.\n";
        agent.releaseXferReq(write_req);
        write_req = nullptr;
        total_time += write_duration;

        size_t total_bytes = transfer_size * num_transfers * iterations;
        total_data_gb += total_bytes / (1024.0 * 1024.0);  // Track in MiB

        write_duration_total = write_duration;
        std::cout << "Write completed:" << std::endl;
        std::cout << "- Time: " << format_duration(write_duration) << std::endl;
        std::cout << "- Data: " << format_data_size(total_bytes) << std::endl;
        std::cout << "- Speed: " << format_throughput(total_bytes, write_duration) << std::endl;
    }

    // Verify keys after write
    if (!skip_write) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 3: Verify keys" << std::endl;
        std::cout << "============================================================" << std::endl;

        std::vector<nixl_query_resp_t> query_resp;
        nixl_opt_args_t query_params;
        query_params.backends.push_back(infinia);

        us_t query_start = getUs();
        ret = agent.queryMem(obj_for_infinia, query_resp, &query_params);
        us_t query_end = getUs();
        us_t query_duration = query_end - query_start;
        if (ret == NIXL_ERR_NOT_SUPPORTED) {
            std::cout << "Note: queryMem() is not supported by the Infinia backend" << std::endl;
            std::cout << "Skipping key verification phase." << std::endl;
        } else if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to query memory (error code: " << ret << ")" << std::endl;
            rc = -8;
            goto cleanup;
        } else {
            // queryMem succeeded - verify results
            int found_count = 0;
            int missing_count = 0;
            std::vector<std::string> missing_keys;

            for (size_t i = 0; i < query_resp.size(); ++i) {
                if (query_resp[i].has_value()) {
                    found_count++;
                } else {
                    missing_count++;
                    missing_keys.push_back(std::string(test_keys[i]));
                }
            }

            query_duration_total = query_duration;
            std::cout << "Query results:" << std::endl;
            std::cout << "- Total keys:   " << query_resp.size() << std::endl;
            std::cout << "- Keys found:   " << found_count << std::endl;
            std::cout << "- Keys missing: " << missing_count << std::endl;
            std::cout << "- Time:         " << format_duration(query_duration) << std::endl;

            if (missing_count > 0) {
                std::cout << "\nMissing keys:" << std::endl;
                for (const auto& key : missing_keys) {
                    std::cout << "  - " << key << std::endl;
                }
                std::cerr << "Warning: Some keys were not found in the Infinia backend" << std::endl;
            }

            total_time += query_duration;
        }
    }

    // Clear buffers before read if doing both operations
    if (!skip_read && !skip_write) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 4: Clearing buffers for read test" << std::endl;
        std::cout << "============================================================" << std::endl;
        us_t clear_start = getUs();
        for (i = 0; i < num_transfers; i++) {
#ifdef HAVE_CUDA
            if (use_vram && vram_addr[i]) {
                if (clear_gpu_buffer(vram_addr[i], transfer_size) != cudaSuccess) {
                    std::cerr << "Failed to clear VRAM buffer " << i << std::endl;
                    rc = -3;
                    goto cleanup;
                }
            }
#endif
            if (use_dram && dram_addr[i]) {
                clear_buffer(dram_addr[i], transfer_size);
            }
            printProgress(float(i + 1) / num_transfers);
            // No cleanup of OBJ_SEG buffers required
        }
        us_t clear_end = getUs();
        clear_duration_total = clear_end - clear_start;
        std::cout << "\n- Time:         " << format_duration(clear_duration_total) << std::endl;
        total_time += clear_duration_total;
    }

    // Perform read test if not skipped
    if (!skip_read) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "PHASE 5: Object to Memory Transfer (Read Test)" << std::endl;
        std::cout << "============================================================" << std::endl;

        us_t read_duration(0);

        // Create descriptor lists for all transfers
        nixl_reg_dlist_t src_reg(use_dram ? DRAM_SEG : VRAM_SEG);
        nixl_reg_dlist_t obj_reg(OBJ_SEG);

        // Add all descriptors
        for (int transfer_idx = 0; transfer_idx < num_transfers; transfer_idx++) {
            if (use_dram) {
                src_reg.addDesc(dram_buf[transfer_idx]);
            } else {
                src_reg.addDesc(vram_buf[transfer_idx]);
            }
            obj_reg.addDesc(ftrans[transfer_idx]);
            printProgress(float(transfer_idx + 1) / num_transfers);
        }
        std::cout << "\nAll descriptors added." << std::endl;

        // Create transfer lists
        nixl_xfer_dlist_t src_list = src_reg.trim();
        nixl_xfer_dlist_t obj_list = obj_reg.trim();

        // Create single transfer request for all transfers
        ret = agent.createXferReq(NIXL_READ, src_list, obj_list,
                "INFINIA_Tester", read_req);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to create read transfer request " << ret << std::endl;
            rc = -5;
            goto cleanup;
        }
        std::cout << "Read transfer request created." << std::endl;

        for (unsigned int iter = 0; iter < iterations; iter++) {
            us_t iter_start = getUs();

            status = agent.postXferReq(read_req);
            if (status < 0) {
                std::cerr << "Failed to post read transfer request " << status << std::endl;
                rc = -6;
                goto cleanup;
            }

            std::cout << " Infinia NIXL_READ has been posted\n";
            std::cout << " Waiting for completion\n";

            while (status != NIXL_SUCCESS) {
                status = agent.getXferStatus(read_req);
                if (status < 0) {
                    std::cerr << "Error during read transfer " << status << std::endl;
                    rc = -7;
                    goto cleanup;
                }
            }

            us_t iter_end = getUs();
            read_duration += (iter_end - iter_start);

            if (iterations > 1) {
                printProgress(float(iter + 1) / iterations);
            }
        }

        std::cout <<" Completed reading data from Infinia KV dataset.\n";
        agent.releaseXferReq(read_req);
        read_req = nullptr;
        total_time += read_duration;

        size_t total_bytes = transfer_size * num_transfers * iterations;
        total_data_gb += total_bytes / (1024.0 * 1024.0);  // Track in MiB

        read_duration_total = read_duration;
        std::cout << "Read completed:" << std::endl;
        std::cout << "- Time: " << format_duration(read_duration) << std::endl;
        std::cout << "- Data: " << format_data_size(total_bytes) << std::endl;
        std::cout << "- Speed: " << format_throughput(total_bytes, read_duration) << std::endl;

        if (!skip_write && seed != -1) {
            // Seed the global generator
            generator.seed(seed);

            std::cout << "\n============================================================" << std::endl;
            std::cout << "PHASE 6: Validating read data" << std::endl;
            std::cout << "============================================================" << std::endl;
            us_t validate_start = getUs();
            for (i = 0; i < num_transfers; i++) {
#ifdef HAVE_CUDA
                if (use_vram) {
                    if (!validate_gpu_buffer(vram_addr[i], transfer_size)) {
                        std::cerr << "VRAM buffer " << i << " validation failed\n";
                        rc = -15;
                        goto cleanup;
                    }
                }
#endif
                if (use_dram) {
                    unsigned char* expected_buffer = (unsigned char*)malloc(transfer_size);
                    if (!expected_buffer) {
                        std::cerr << "Failed to allocate validation buffer\n";
                        rc = -15;
                        goto cleanup;
                    }
                    fill_test_pattern(expected_buffer, transfer_size);
                    if (memcmp(dram_addr[i], expected_buffer, transfer_size) != 0) {
                        std::cerr << "DRAM buffer " << i << " validation failed\n";
                        free(expected_buffer);
                        rc = -15;
                        goto cleanup;
                    }
                    free(expected_buffer);
                }
                printProgress(float(i + 1) / num_transfers);
            }
            us_t validate_end = getUs();
            validate_duration_total = validate_end - validate_start;
            std::cout << "\nVerification completed successfully!" << std::endl;
            std::cout << "- Time:         " << format_duration(validate_duration_total) << std::endl;
            total_time += validate_duration_total;
        }
    }

cleanup:
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PHASE 7: Cleanup" << std::endl;
    std::cout << "============================================================" << std::endl;

    us_t cleanup_start = getUs();
    printProgress(1.0);

    // Cleanup transfer request handles
    if (write_req != nullptr) {
        agent.releaseXferReq(write_req);
    }
    if (read_req != nullptr) {
        agent.releaseXferReq(read_req);
    }

    // Cleanup resources
    agent.deregisterMem(obj_for_infinia);
#ifdef HAVE_CUDA
    if (use_vram) {
       agent.deregisterMem(vram_for_infinia);
        for (i = 0; i < num_transfers; i++) {
            if (vram_addr[i]) cudaFree(vram_addr[i]);
        }
        delete[] vram_addr;
        delete[] vram_buf;
    }
#endif
    if (use_dram) {
        agent.deregisterMem(dram_for_infinia);
        for (i = 0; i < num_transfers; i++) {
            if (dram_addr[i]) free(dram_addr[i]);
        }
        delete[] dram_addr;
        delete[] dram_buf;
    }
    if (test_keys) {
        for (i = 0; i < num_transfers; i++) {
            delete[] test_keys[i];
        }
        delete[] test_keys;
    }
    delete[] ftrans;

    us_t cleanup_end = getUs();
    cleanup_duration_total = cleanup_end - cleanup_start;
    std::cout << "\n- Time:         " << format_duration(cleanup_duration_total) << std::endl;
    total_time += cleanup_duration_total;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "                    TEST SUMMARY                             " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Total keys:   " << num_transfers << std::endl;
    std::cout << "Object size:  " << transfer_size << " bytes" << std::endl;
    std::cout << "Total time:   " << format_duration(total_time) << std::endl;
    std::cout << "Total data:   " << std::fixed << std::setprecision(2) << total_data_gb << " MiB" << std::endl;
    std::cout << "\nPhase breakdown:" << std::endl;
    std::cout << "1. Allocate:  " << format_duration(alloc_duration) << std::endl;
    std::cout << "2. Write:     " << format_duration(write_duration_total) << std::endl;
    std::cout << "3. Verify:    " << format_duration(query_duration_total) << std::endl;
    std::cout << "4. Clear:     " << format_duration(clear_duration_total) << std::endl;
    std::cout << "5. Read:      " << format_duration(read_duration_total) << std::endl;
    std::cout << "6. Validate:  " << format_duration(validate_duration_total) << std::endl;
    std::cout << "7. Cleanup:   " << format_duration(cleanup_duration_total) << std::endl;
    std::cout << "============================================================" << std::endl;
    return rc;
}
