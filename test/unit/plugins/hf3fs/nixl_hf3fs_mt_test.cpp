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
#include <filesystem>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <nixl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <getopt.h>
#include <absl/strings/str_format.h>
#include "common/nixl_time.h"
#include "temp_file.h"

#define NIXL_3FS_VALIDATION_MODE 0

namespace {
    constexpr int default_num_threads = 4;
    constexpr int default_transfers_per_thread = 10;
    constexpr size_t default_transfer_size = 1024 * 1024;  // 1MB
    constexpr int default_write_iterations = 1;
    constexpr int default_read_iterations = 1;
    constexpr int default_iopool_size = 64;
    constexpr char test_phrase[] = "NIXL HF3FS Multi-Thread Test Pattern 2025";
    constexpr char test_file_name[] = "mt_testfile";
    constexpr mode_t std_file_permissions = 0744;
    constexpr char default_test_files_dir_path[] = "/mnt/3fs";

    constexpr size_t kb_size = 1024;
    constexpr size_t mb_size = 1024 * 1024;
    constexpr size_t gb_size = 1024 * 1024 * 1024;

    std::mutex cout_mutex;
    std::atomic<int> total_completed_write_transfers{0};
    std::atomic<int> total_completed_read_transfers{0};
    std::atomic<int> total_failed_write_transfers{0};
    std::atomic<int> total_failed_read_transfers{0};
    std::atomic<int> finished_write_threads{0};

    // Barrier to synchronize between write and read stages
    std::mutex barrier_mutex;
    std::condition_variable barrier_cv;
    bool write_stage_complete = false;

    void print_protected(const std::string& message) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        absl::PrintF("%s\n", message);
    }

    std::string
    format_data_size(size_t bytes) {
        if (bytes >= gb_size) {
            return absl::StrFormat("%.2f GB", (double)bytes / gb_size);
        } else if (bytes >= mb_size) {
            return absl::StrFormat("%.2f MB", (double)bytes / mb_size);
        } else if (bytes >= kb_size) {
            return absl::StrFormat("%.2f KB", (double)bytes / kb_size);
        } else {
            return absl::StrFormat("%zu bytes", bytes);
        }
    }

    void fill_test_pattern(void* buffer, size_t size, int thread_id) {
        char* buf = (char*)buffer;
        size_t offset = 0;
        while (offset < size) {
            size_t remaining = size - offset;
            size_t copy_len = std::min(remaining, strlen(test_phrase));
            memcpy(buf + offset, test_phrase, copy_len);
            offset += copy_len;
        }
        // Add thread ID at the end for validation
        if (size >= sizeof(int)) {
            memcpy(buf + size - sizeof(int), &thread_id, sizeof(int));
        }
    }

#if NIXL_3FS_VALIDATION_MODE
    bool validate_buffer(void* buffer, size_t size, int thread_id) {
        char* buf = (char*)buffer;
        // Check thread ID at the end
        if (size >= sizeof(int)) {
            int stored_thread_id;
            memcpy(&stored_thread_id, buf + size - sizeof(int), sizeof(int));
            if (stored_thread_id != thread_id) {
                print_protected("Thread ID mismatch in validation");
                return false;
            }
        }
        return true;
    }
#endif

    class ThreadStats {
    public:
        nixlTime::us_t total_time{0};
        size_t total_data{0};
        int completed_transfers{0};
        int failed_transfers{0};

        void add_transfer(size_t data_size, nixlTime::us_t duration) {
            total_time += duration;
            total_data += data_size;
            completed_transfers++;
        }

        void add_failure() {
            failed_transfers++;
        }

        double get_throughput_mbps() const {
            if (total_time == 0) {
                return 0.0;
            }
            double seconds = total_time / 1000000.0;
            return ((double)total_data / mb_size) / seconds;
        }
    };

    // Modify struct definitions to avoid default constructors
    struct PreparedWriteRequest {
        nixlXferReqH* req = nullptr;
        // Remove the member variables that require non-default constructors
    };

    struct PreparedReadRequest {
        nixlXferReqH* req = nullptr;
        // Remove the member variables that require non-default constructors
    };

    void
    execute_transfer_iterations(nixlAgent &agent,
                                nixlXferReqH *req,
                                int iterations,
                                const std::string &operation_type,
                                int thread_id) {
        for (int iter = 0; iter < iterations; iter++) {
            auto status = agent.postXferReq(req);
            if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                throw std::runtime_error("Failed to post " + operation_type + " request (iter " +
                                         std::to_string(iter) +
                                         "), err: " + nixlEnumStrings::statusStr(status));
            }

            // Wait for completion
            status = agent.getXferStatus(req);
            while (status == NIXL_IN_PROG) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                status = agent.getXferStatus(req);
            }

            if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                throw std::runtime_error("Failed to wait for " + operation_type +
                                         " completion (iter " + std::to_string(iter) +
                                         "), err: " + nixlEnumStrings::statusStr(status));
            }

#if NIXL_3FS_VALIDATION_MODE
            print_protected(absl::StrFormat("thread_id: %d %s_iter: %d %s completed",
                                            thread_id,
                                            operation_type,
                                            iter,
                                            operation_type));
#endif
        }
    }

    // Update worker functions to accept all parameters directly
    void
    worker_write_thread(int thread_id,
                        nixlAgent &agent,
                        const std::string &test_dir,
                        size_t transfer_size,
                        int num_transfers,
                        int write_iterations,
                        ThreadStats &write_stats,
                        const nixl_reg_dlist_t &dram_list,
                        const nixl_reg_dlist_t &file_list,
                        size_t start_idx,
                        nixlXferReqH *write_req) {
        try {
#if NIXL_3FS_VALIDATION_MODE
            auto write_start = nixlTime::getUs();
#endif
            execute_transfer_iterations(agent, write_req, write_iterations, "write", thread_id);

#if NIXL_3FS_VALIDATION_MODE
            auto write_end = nixlTime::getUs();
            write_stats.add_transfer(transfer_size * num_transfers * write_iterations,
                                     write_end - write_start);

            print_protected(absl::StrFormat(
                "thread_id: %d total write data: %s duration: %lld µs write_iterations: %d",
                thread_id,
                format_data_size(transfer_size * num_transfers * write_iterations),
                write_end - write_start,
                write_iterations));
#endif
            total_completed_write_transfers += num_transfers * write_iterations;

            // Notify that this thread has completed its write stage
            {
                std::lock_guard<std::mutex> lock(barrier_mutex);
                finished_write_threads++;
                if (finished_write_threads.load() == thread_id + 1) {
                    write_stage_complete = true;
                    barrier_cv.notify_all();
                }
            }
        }
        catch (const std::exception &e) {
            print_protected(absl::StrFormat("Thread %d write failed: %s", thread_id, e.what()));
            total_failed_write_transfers += num_transfers * write_iterations;
            write_stats.failed_transfers += num_transfers * write_iterations;

            // Notify that this thread has completed (with errors)
            {
                std::lock_guard<std::mutex> lock(barrier_mutex);
                finished_write_threads++;
                if (finished_write_threads.load() == thread_id + 1) {
                    write_stage_complete = true;
                    barrier_cv.notify_all();
                }
            }
        }
    }

    void
    worker_read_thread(int thread_id,
                       nixlAgent &agent,
                       const std::string &test_dir,
                       size_t transfer_size,
                       int num_transfers,
                       int read_iterations,
                       ThreadStats &read_stats,
                       const nixl_reg_dlist_t &dram_list,
                       const nixl_reg_dlist_t &file_list,
                       size_t start_idx,
                       nixlXferReqH *read_req) {
        try {
#if NIXL_3FS_VALIDATION_MODE
            // Clear DRAM buffers for this thread's range
            for (int i = 0; i < num_transfers; i++) {
                void* ptr = (void*)dram_list[start_idx + i].addr;
                memset(ptr, 0, transfer_size);
            }

            auto read_start = nixlTime::getUs();
#endif
            execute_transfer_iterations(agent, read_req, read_iterations, "read", thread_id);

#if NIXL_3FS_VALIDATION_MODE
            auto read_end = nixlTime::getUs();
            read_stats.add_transfer(transfer_size * num_transfers * read_iterations,
                                    read_end - read_start);
            print_protected(absl::StrFormat(
                "thread_id: %d total read data: %zu duration: %lld µs read_iterations: %d",
                thread_id,
                format_data_size(transfer_size * num_transfers * read_iterations),
                read_end - read_start,
                read_iterations));

            // Validate read data (only validate after the last iteration)
            for (int i = 0; i < num_transfers; i++) {
                void* ptr = (void*)dram_list[start_idx + i].addr;
                if (!validate_buffer(ptr, transfer_size, thread_id)) {
                    read_stats.add_failure();
                    total_failed_read_transfers++;
                }
            }
#endif

            total_completed_read_transfers += num_transfers * read_iterations;
        } catch (const std::exception& e) {
            print_protected(absl::StrFormat("Thread %d read failed: %s", thread_id, e.what()));
            total_failed_read_transfers += num_transfers * read_iterations;
            read_stats.failed_transfers += num_transfers * read_iterations;
        }
    }
}

int main(int argc, char *argv[]) {
    // Default parameters
    int num_threads = default_num_threads;
    int transfers_per_thread = default_transfers_per_thread;
    size_t transfer_size = default_transfer_size;
    int write_iterations = default_write_iterations;
    int read_iterations = default_read_iterations;
    std::string test_dir = default_test_files_dir_path;
    int iopool_size = default_iopool_size;

    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "t:n:s:w:r:d:h:i:")) != -1) {
        switch (opt) {
            case 't':
                num_threads = std::stoi(optarg);
                break;
            case 'n':
                transfers_per_thread = std::stoi(optarg);
                break;
            case 's':
                transfer_size = std::stoul(optarg);
                break;
            case 'w':
                write_iterations = std::stoi(optarg);
                break;
            case 'r':
                read_iterations = std::stoi(optarg);
                break;
            case 'd':
                test_dir = optarg;
                break;
            case 'i':
                iopool_size = std::stoi(optarg);
                break;
            case 'h':
            default:
                std::cout << absl::StrFormat(
                                 "Usage: %s [-t num_threads] [-n transfers_per_thread] [-s "
                                 "transfer_size] [-w write_iterations] [-r read_iterations] [-d "
                                 "test_dir]",
                                 argv[0])
                          << std::endl;
                std::cout << absl::StrFormat("  -t: Number of threads (default: %d)",
                                             default_num_threads)
                          << std::endl;
                std::cout << absl::StrFormat("  -n: Transfers per thread (default: %d)",
                                             default_transfers_per_thread)
                          << std::endl;
                std::cout << absl::StrFormat("  -s: Transfer size in bytes (default: %zu)",
                                             default_transfer_size)
                          << std::endl;
                std::cout << absl::StrFormat("  -w: Number of write iterations (default: %d)",
                                             default_write_iterations)
                          << std::endl;
                std::cout << absl::StrFormat("  -r: Number of read iterations (default: %d)",
                                             default_read_iterations)
                          << std::endl;
                std::cout << absl::StrFormat("  -d: Test directory (default: %s)",
                                             default_test_files_dir_path)
                          << std::endl;
                std::cout << absl::StrFormat("  -i: IO Pool Size (default: %s)",
                                             default_iopool_size)
                          << std::endl;
                return (opt == 'h') ? 0 : 1;
        }
    }

    // Create test directory
    try {
        std::filesystem::create_directories(test_dir);
    } catch (const std::exception& e) {
        std::cerr << absl::StrFormat("Failed to create test directory: %s", e.what()) << std::endl;
        return 1;
    }

    // Initialize NIXL
    nixlAgentConfig cfg(true, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
    nixlAgent agent("HF3FSMultiThreadTester", cfg);

    // Create HF3FS backend
    nixlBackendH* hf3fs = nullptr;
    nixl_b_params_t params;
    params["iopool_size"] = std::to_string(iopool_size);
    nixl_status_t ret = agent.createBackend("HF3FS", params, hf3fs);
    if (ret != NIXL_SUCCESS) {
        std::cerr << absl::StrFormat("Error creating HF3FS backend: %d", ret) << std::endl;
        return 1;
    }

    if (hf3fs == nullptr) {
        std::cerr << "Error creating a new backend" << std::endl;
        return 1;
    }

    // Pre-allocate and register all DRAM buffers
    std::vector<std::unique_ptr<void, void(*)(void*)>> dram_buffers;
    nixl_reg_dlist_t dram_list(DRAM_SEG);
    nixl_reg_dlist_t file_list(FILE_SEG);

    // Pre-create all test files
    std::vector<tempFile> files;
    for (int i = 0; i < num_threads * transfers_per_thread; i++) {
        // Allocate DRAM buffer
        void* ptr = malloc(transfer_size);
        if (!ptr) {
            std::cerr << "Failed to allocate DRAM buffer" << std::endl;
            return 1;
        }
        dram_buffers.emplace_back(ptr, free);

        // Create test file
        std::string filename = test_dir + "/" + test_file_name +
                             "_" + std::to_string(i);
        files.emplace_back(filename, O_RDWR|O_CREAT|O_TRUNC, std_file_permissions);
        if (files.back() < 0) {
            std::cerr << "Failed to create file" << std::endl;
            return 1;
        }

        // Setup DRAM descriptor
        nixlBlobDesc dram_desc;
        dram_desc.addr = (uintptr_t)ptr;
        dram_desc.len = transfer_size;
        dram_desc.devId = 0;
        dram_list.addDesc(dram_desc);

        // Setup file descriptor
        nixlBlobDesc file_desc;
        file_desc.addr = 0;  // offset
        file_desc.len = transfer_size;
        file_desc.devId = files.back();
        file_list.addDesc(file_desc);
    }

    // Register all memory at once
    if (agent.registerMem(dram_list) != NIXL_SUCCESS ||
        agent.registerMem(file_list) != NIXL_SUCCESS) {
        std::cerr << "Failed to register memory" << std::endl;
        return 1;
    }

    // Fill test pattern for each thread's DRAM buffers
    for (int thread_id = 0; thread_id < num_threads; thread_id++) {
        size_t start_idx = thread_id * transfers_per_thread;
        for (int i = 0; i < transfers_per_thread; i++) {
            void* ptr = (void*)dram_list[start_idx + i].addr;
            fill_test_pattern(ptr, transfer_size, thread_id);
        }
    }

    // Changed vector initialization method
    std::vector<nixlXferReqH*> write_requests(num_threads, nullptr);
    std::vector<nixlXferReqH*> read_requests(num_threads, nullptr);

    // Temporary lists for request creation
    std::vector<nixl_xfer_dlist_t> write_dram_lists;
    std::vector<nixl_xfer_dlist_t> write_file_lists;
    std::vector<nixl_xfer_dlist_t> read_dram_lists;
    std::vector<nixl_xfer_dlist_t> read_file_lists;

    // Create all write requests
    std::cout << "\n=== PREPARING WRITE REQUESTS ===" << std::endl;
    for (int i = 0; i < num_threads; i++) {
        size_t start_idx = i * transfers_per_thread;

        // Create subset lists for this thread's range
        nixl_reg_dlist_t thread_dram_list(DRAM_SEG);
        nixl_reg_dlist_t thread_file_list(FILE_SEG);

        for (int j = 0; j < transfers_per_thread; j++) {
            thread_dram_list.addDesc(dram_list[start_idx + j]);
            thread_file_list.addDesc(file_list[start_idx + j]);
        }

        // Perform write operations
        auto dram_list_copy = thread_dram_list.trim();
        auto file_list_copy = thread_file_list.trim();

        // Store the trimmed lists
        write_dram_lists.push_back(std::move(dram_list_copy));
        write_file_lists.push_back(std::move(file_list_copy));

        // Create the transfer request
        nixlXferReqH* req = nullptr;
        if (agent.createXferReq(NIXL_WRITE, write_dram_lists.back(), write_file_lists.back(),
                              "HF3FSMultiThreadTester", req) != NIXL_SUCCESS) {
            std::cerr << absl::StrFormat("Failed to create write request for thread %d", i)
                      << std::endl;
            return 1;
        }

        write_requests[i] = req;
    }

    // Create all read requests
    std::cout << "\n=== PREPARING READ REQUESTS ===" << std::endl;
    for (int i = 0; i < num_threads; i++) {
        size_t start_idx = i * transfers_per_thread;

        // Create subset lists for this thread's range
        nixl_reg_dlist_t thread_dram_list(DRAM_SEG);
        nixl_reg_dlist_t thread_file_list(FILE_SEG);

        for (int j = 0; j < transfers_per_thread; j++) {
            thread_dram_list.addDesc(dram_list[start_idx + j]);
            thread_file_list.addDesc(file_list[start_idx + j]);
        }

        // Prepare read operations
        auto dram_list_copy = thread_dram_list.trim();
        auto file_list_copy = thread_file_list.trim();

        // Store the trimmed lists
        read_dram_lists.push_back(std::move(dram_list_copy));
        read_file_lists.push_back(std::move(file_list_copy));

        // Create the transfer request
        nixlXferReqH* req = nullptr;
        if (agent.createXferReq(NIXL_READ, read_dram_lists.back(), read_file_lists.back(),
                              "HF3FSMultiThreadTester", req) != NIXL_SUCCESS) {
            std::cerr << absl::StrFormat("Failed to create read request for thread %d", i)
                      << std::endl;
            return 1;
        }

        read_requests[i] = req;
    }

    // Stage 1: Write Operation
    std::cout << "\n=== STAGE 1: WRITE OPERATIONS ===" << std::endl;
    std::cout << absl::StrFormat("Running %d write iterations per thread", write_iterations)
              << std::endl;
    std::vector<std::thread> write_threads;
    std::vector<ThreadStats> thread_write_stats(num_threads);
    size_t total_write_bytes =
        num_threads * transfers_per_thread * transfer_size * write_iterations;

    auto write_start_time = nixlTime::getUs();

    for (int i = 0; i < num_threads; i++) {
        size_t start_idx = i * transfers_per_thread;
        write_threads.emplace_back(worker_write_thread,
                                   i,
                                   std::ref(agent),
                                   test_dir,
                                   transfer_size,
                                   transfers_per_thread,
                                   write_iterations,
                                   std::ref(thread_write_stats[i]),
                                   std::ref(dram_list),
                                   std::ref(file_list),
                                   start_idx,
                                   write_requests[i]);
    }

    // Wait for all write threads to complete
    for (auto& thread : write_threads) {
        thread.join();
    }

    auto write_end_time = nixlTime::getUs();
    auto write_duration = write_end_time - write_start_time;
    double write_duration_seconds = write_duration / 1000000.0;
    double write_total_gbps = ((double)total_write_bytes / gb_size) / write_duration_seconds;

    // Release write requests
    for (auto& req : write_requests) {
        agent.releaseXferReq(req);
    }

    // Print write results
    std::cout << "\nWrite Test Results:" << std::endl;
    std::cout << absl::StrFormat("Total transfers completed: %d",
                                 total_completed_write_transfers.load())
              << std::endl;
    std::cout << absl::StrFormat("Total transfers failed: %d", total_failed_write_transfers.load())
              << std::endl;
    std::cout << absl::StrFormat("Write duration: %.6f seconds", write_duration_seconds)
              << std::endl;
    std::cout << absl::StrFormat("Total write data transferred: %s",
                                 format_data_size(total_write_bytes))
              << std::endl;
    std::cout << absl::StrFormat("Write throughput: %.2f GB/s", write_total_gbps) << std::endl;

    // Stage 2: Read Operation
    std::cout << "\n=== STAGE 2: READ OPERATIONS ===" << std::endl;
    std::cout << absl::StrFormat("Running %d read iterations per thread", read_iterations)
              << std::endl;
    std::vector<std::thread> read_threads;
    std::vector<ThreadStats> thread_read_stats(num_threads);
    size_t total_read_bytes = num_threads * transfers_per_thread * transfer_size * read_iterations;

    auto read_start_time = nixlTime::getUs();

    for (int i = 0; i < num_threads; i++) {
        size_t start_idx = i * transfers_per_thread;
        read_threads.emplace_back(worker_read_thread,
                                  i,
                                  std::ref(agent),
                                  test_dir,
                                  transfer_size,
                                  transfers_per_thread,
                                  read_iterations,
                                  std::ref(thread_read_stats[i]),
                                  std::ref(dram_list),
                                  std::ref(file_list),
                                  start_idx,
                                  read_requests[i]);
    }

    // Wait for all read threads to complete
    for (auto& thread : read_threads) {
        thread.join();
    }

    auto read_end_time = nixlTime::getUs();
    auto read_duration = read_end_time - read_start_time;
    double read_duration_seconds = read_duration / 1000000.0;
    double read_total_gbps = ((double)total_read_bytes / gb_size) / read_duration_seconds;

    // Release read requests
    for (auto& req : read_requests) {
        agent.releaseXferReq(req);
    }

    // Print read results
    std::cout << "\nRead Test Results:" << std::endl;
    std::cout << absl::StrFormat("Total transfers completed: %d",
                                 total_completed_read_transfers.load())
              << std::endl;
    std::cout << absl::StrFormat("Total transfers failed: %d", total_failed_read_transfers.load())
              << std::endl;
    std::cout << absl::StrFormat("Read duration: %.6f seconds", read_duration_seconds) << std::endl;
    std::cout << absl::StrFormat("Total read data transferred: %s",
                                 format_data_size(total_read_bytes))
              << std::endl;
    std::cout << absl::StrFormat("Read throughput: %.2f GB/s", read_total_gbps) << std::endl;

    // Deregister memory
    agent.deregisterMem(file_list);
    agent.deregisterMem(dram_list);


#if NIXL_3FS_VALIDATION_MODE
    // Print per-thread statistics
    std::cout << "\nPer-thread throughput:" << std::endl;
    double total_write_thoughput = 0.0;
    double total_read_thoughput = 0.0;

    for (size_t i = 0; i < thread_write_stats.size(); i++) {
        std::cout << absl::StrFormat("  Thread %zu: read %.2f MB/s write %.2f MB/s",
                                     i,
                                     thread_read_stats[i].get_throughput_mbps(),
                                     thread_write_stats[i].get_throughput_mbps())
                  << std::endl;
        total_write_thoughput += thread_write_stats[i].get_throughput_mbps();
        total_read_thoughput += thread_read_stats[i].get_throughput_mbps();
    }

    std::cout << absl::StrFormat(
                     "\nAggregate Per-thread throughput: read %.2f MB/s write %.2f MB/s",
                     total_read_thoughput,
                     total_write_thoughput)
              << std::endl;
#endif

    int total_failed_transfers = total_failed_write_transfers + total_failed_read_transfers;
    if (total_failed_transfers > 0) {
        std::cout << "Test failed" << std::endl;
    } else {
        std::cout << "Test passed" << std::endl;
    }

    return (total_failed_transfers > 0) ? 1 : 0;
}
