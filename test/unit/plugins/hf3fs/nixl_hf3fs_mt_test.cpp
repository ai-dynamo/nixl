/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
#include <nixl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <getopt.h>
#include "common/nixl_time.h"
#include "temp_file.h"

namespace {
    constexpr int default_num_threads = 4;
    constexpr int default_transfers_per_thread = 10;
    constexpr size_t default_transfer_size = 1024 * 1024;  // 1MB
    constexpr char test_phrase[] = "NIXL HF3FS Multi-Thread Test Pattern 2025";
    constexpr size_t test_phrase_len = sizeof(test_phrase) - 1;
    constexpr char test_file_name[] = "mt_testfile";
    constexpr mode_t std_file_permissions = 0744;
    constexpr char default_test_files_dir_path[] = "/mnt/3fs/";

    constexpr size_t kb_size = 1024;
    constexpr size_t mb_size = 1024 * 1024;
    constexpr size_t gb_size = 1024 * 1024 * 1024;

    std::mutex cout_mutex;
    std::atomic<int> total_completed_transfers{0};
    std::atomic<int> total_failed_transfers{0};

    void print_protected(const std::string& message) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << message << std::endl;
    }

    void fill_test_pattern(void* buffer, size_t size, int thread_id) {
        char* buf = (char*)buffer;
        size_t offset = 0;
        while (offset < size) {
            size_t remaining = size - offset;
            size_t copy_len = std::min(remaining, test_phrase_len);
            memcpy(buf + offset, test_phrase, copy_len);
            offset += copy_len;
        }
        // Add thread ID at the end for validation
        if (size >= sizeof(int)) {
            memcpy(buf + size - sizeof(int), &thread_id, sizeof(int));
        }
    }

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

        double get_throughput_gbps() const {
            if (total_time == 0) {
                return 0.0;
            }
            double seconds = total_time / 1000000.0;
            return ((double)total_data / gb_size) / seconds;
        }
    };

    void worker_thread(int thread_id, 
                      nixlAgent& agent,
                      const std::string& test_dir,
                      size_t transfer_size,
                      int num_transfers,
                      ThreadStats& stats,
                      const nixl_reg_dlist_t& dram_list,
                      const nixl_reg_dlist_t& file_list,
                      size_t start_idx) {
        try {
            // Create subset lists for this thread's range
            nixl_reg_dlist_t thread_dram_list(DRAM_SEG);
            nixl_reg_dlist_t thread_file_list(FILE_SEG);

            for (int i = 0; i < num_transfers; i++) {
                thread_dram_list.addDesc(dram_list[start_idx + i]);
                thread_file_list.addDesc(file_list[start_idx + i]);
            }

            // Perform write operations
            nixl_xfer_dlist_t write_dram_list = thread_dram_list.trim();
            nixl_xfer_dlist_t write_file_list = thread_file_list.trim();
            nixlXferReqH* write_req = nullptr;

            if (agent.createXferReq(NIXL_WRITE, write_dram_list, write_file_list,
                                  "HF3FSMultiThreadTester", write_req) != NIXL_SUCCESS) {
                throw std::runtime_error("Failed to create write request");
            }

            auto write_start = nixlTime::getUs();
            auto status = agent.postXferReq(write_req);
            if (status < 0) {
                agent.releaseXferReq(write_req);
                throw std::runtime_error("Failed to post write request, err: " + std::to_string(status));
            }

            // Wait for write completion
            status = agent.getXferStatus(write_req);
            while (status == NIXL_IN_PROG) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                status = agent.getXferStatus(write_req);
            }

            if (status < 0) {
                agent.releaseXferReq(write_req);
                throw std::runtime_error("Failed to wait for write completion, err: " + std::to_string(status));
            }

            auto write_end = nixlTime::getUs();
            stats.add_transfer(transfer_size * num_transfers, write_end - write_start);
            print_protected("thread_id: " + std::to_string(thread_id) +
                            " write data_size: " + std::to_string(transfer_size * num_transfers) +
                            " duration: " + std::to_string(write_end - write_start));

            // Clear DRAM buffers for this thread's range
            for (int i = 0; i < num_transfers; i++) {
                void* ptr = (void*)dram_list[start_idx + i].addr;
                memset(ptr, 0, transfer_size);
            }

            agent.releaseXferReq(write_req);

            // Perform read operations
            nixl_xfer_dlist_t read_dram_list = thread_dram_list.trim();
            nixl_xfer_dlist_t read_file_list = thread_file_list.trim();
            nixlXferReqH* read_req = nullptr;

            if (agent.createXferReq(NIXL_READ, read_dram_list, read_file_list,
                                  "HF3FSMultiThreadTester", read_req) != NIXL_SUCCESS) {
                throw std::runtime_error("Failed to create read request");
            }

            auto read_start = nixlTime::getUs();
            status = agent.postXferReq(read_req);
            if (status < 0) {
                agent.releaseXferReq(read_req);
                throw std::runtime_error("Failed to post read request, err: " + std::to_string(status));
            }

            // Wait for read completion
            status = agent.getXferStatus(read_req);
            while (status == NIXL_IN_PROG) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                status = agent.getXferStatus(read_req);
            }

            if (status < 0) {
                agent.releaseXferReq(read_req);
                throw std::runtime_error("Failed to wait for read completion, err: " + std::to_string(status));
            }
            auto read_end = nixlTime::getUs();
            stats.add_transfer(transfer_size * num_transfers, read_end - read_start);
            print_protected("thread_id: " + std::to_string(thread_id) +
                            " read data_size: " + std::to_string(transfer_size * num_transfers) +
                            " duration: " + std::to_string(read_end - read_start));

            // Validate read data
            for (int i = 0; i < num_transfers; i++) {
                void* ptr = (void*)dram_list[start_idx + i].addr;
                if (!validate_buffer(ptr, transfer_size, thread_id)) {
                    stats.add_failure();
                    total_failed_transfers++;
                }
            }

            // Cleanup
            agent.releaseXferReq(read_req);

            total_completed_transfers += num_transfers;

        } catch (const std::exception& e) {
            print_protected("Thread " + std::to_string(thread_id) + " failed: " + e.what());
            total_failed_transfers += num_transfers;
            stats.failed_transfers += num_transfers;
        }
    }
}

int main(int argc, char *argv[]) {
    // Default parameters
    int num_threads = default_num_threads;
    int transfers_per_thread = default_transfers_per_thread;
    size_t transfer_size = default_transfer_size;
    std::string test_dir = default_test_files_dir_path;

    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "t:n:s:d:h")) != -1) {
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
            case 'd':
                test_dir = optarg;
                break;
            case 'h':
            default:
                std::cout << "Usage: " << argv[0] << " [-t num_threads] [-n transfers_per_thread] "
                          << "[-s transfer_size] [-d test_dir]" << std::endl;
                return (opt == 'h') ? 0 : 1;
        }
    }

    // Create test directory
    try {
        std::filesystem::create_directories(test_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create test directory: " << e.what() << std::endl;
        return 1;
    }

    // Initialize NIXL
    nixlAgentConfig cfg(true, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
    nixl_b_params_t params;
    nixlAgent agent("HF3FSMultiThreadTester", cfg);

    // Create HF3FS backend
    nixlBackendH* hf3fs = nullptr;
    nixl_status_t ret = agent.createBackend("HF3FS", params, hf3fs);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Error creating HF3FS backend: " << ret << std::endl;
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

    // Create threads
    std::vector<std::thread> threads;
    std::vector<ThreadStats> thread_stats(num_threads);
    auto start_time = nixlTime::getUs();

    for (int i = 0; i < num_threads; i++) {
        size_t start_idx = i * transfers_per_thread;
        threads.emplace_back(worker_thread, i, std::ref(agent), test_dir,
                           transfer_size, transfers_per_thread, std::ref(thread_stats[i]),
                           std::ref(dram_list), std::ref(file_list), start_idx);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Deregister memory
    agent.deregisterMem(file_list);
    agent.deregisterMem(dram_list);

    auto end_time = nixlTime::getUs();
    auto total_duration = end_time - start_time;

    // Print results
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "Total threads: " << num_threads << std::endl;
    std::cout << "Transfers per thread: " << transfers_per_thread << std::endl;
    std::cout << "Transfer size: " << transfer_size << " bytes" << std::endl;
    std::cout << "Total transfers completed: " << total_completed_transfers << std::endl;
    std::cout << "Total transfers failed: " << total_failed_transfers << std::endl;
    std::cout << "Total duration: " << (total_duration / 1000000.0) << " seconds" << std::endl;

    double total_thoughput = 0.0;

    std::cout << "Per-thread throughput:" << std::endl;
    for (size_t i = 0; i < thread_stats.size(); i++) {
        std::cout << "  Thread " << i << ": " << std::setprecision(2) <<
            thread_stats[i].get_throughput_gbps() << " GB/s" << std::endl;
        total_thoughput += thread_stats[i].get_throughput_gbps();
    }

    std::cout << "Total throughput: " << std::setprecision(2) << total_thoughput << " GB/s" << std::endl;

    if (total_failed_transfers > 0) {
        std::cout << "Test failed" << std::endl;
    } else {
        std::cout << "Test passed" << std::endl;
    }

    return (total_failed_transfers > 0) ? 1 : 0;
}