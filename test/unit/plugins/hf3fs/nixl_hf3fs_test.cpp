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
#include <algorithm>
#include <memory>
#include <vector>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <getopt.h>
#include "common/nixl_time.h"
#include "path_mode_common.h"
#include "temp_file.h"

namespace {
    const size_t page_size = sysconf(_SC_PAGESIZE);

    constexpr int default_num_transfers = 10;  // Same as original HF3FS test
    constexpr size_t default_transfer_size = 1024;
    constexpr char test_phrase[] = "NIXL HF3FS Test Pattern 2025";
    constexpr char repost_test_phrase_1[] = "NIXL HF3FS Repost Test Pattern 1111";
    constexpr char repost_test_phrase_2[] = "NIXL HF3FS Repost Test Pattern 2222";
    static_assert(sizeof(repost_test_phrase_1) == sizeof(repost_test_phrase_2),
                  "Test phrases must be the same length");
    constexpr char test_file_name[] = "testfile";
    constexpr mode_t std_file_permissions = 0744;

    constexpr size_t kb_size = 1024;
    constexpr size_t mb_size = 1024 * 1024;
    constexpr size_t gb_size = 1024 * 1024 * 1024;
    constexpr double us_to_s(double us) { return us / 1000000.0; }

    constexpr int line_width = 60;
    constexpr int progress_bar_width = line_width - 2;  // -2 for the brackets
    const std::string line_str(line_width, '=');
    std::string center_str(const std::string& str) {
        return std::string((line_width - str.length()) / 2, ' ') + str;
    }

    constexpr int default_max_waits = 20000;
    constexpr nixlTime::us_t default_wait_time = 10000;
    constexpr char default_test_files_dir_path[] = "/mnt/3fs/";

    // Helper function to generate timestamped filename
    std::string generate_timestamped_filename(const std::string& base_name) {
        std::time_t t = std::time(nullptr);
        char timestamp[100];
        std::strftime(timestamp, sizeof(timestamp),
                      "%Y%m%d%H%M%S", std::localtime(&t));
        return base_name + std::string(timestamp);
    }

    // Helper function to fill buffer with repeating pattern
    void
    fill_test_pattern(void *buffer, size_t size, const char *phrase = test_phrase) {
        char* buf = (char*)buffer;
        size_t phrase_len = strlen(phrase);
        size_t offset = 0;

        while (offset < size) {
            size_t remaining = size - offset;
            size_t copy_len = (remaining < phrase_len) ? remaining : phrase_len;
            memcpy(buf + offset, phrase, copy_len);
            offset += copy_len;
        }
    }

    void clear_buffer(void* buffer, size_t size) {
        memset(buffer, 0, size);
    }

    // Helper function to format duration
    std::string format_duration(nixlTime::us_t us) {
        nixlTime::ms_t ms = us/1000.0;
        if (ms < 1000) {
            return std::to_string((int)ms) + " ms";
        }
        double seconds = ms / 1000.0;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << seconds << " sec";
        return ss.str();
    }

    void printProgress(float progress) {
        std::cout << "[";
        int pos = progress_bar_width * progress;
        for (int i = 0; i < progress_bar_width; ++i) {
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

    std::string phase_title(const std::string& title) {
        static int phase_num = 1;
        return "PHASE " + std::to_string(phase_num++) + ": " + title;
    }

    void print_segment_title(const std::string& title) {
        std::cout << std::endl << line_str << std::endl;
        std::cout << center_str(title) << std::endl;
        std::cout << line_str << std::endl;
    }

    // Custom deleter for malloc allocated memory
    struct MallocDeleter {
        void operator()(void* ptr) const {
            if (ptr) free(ptr);
        }
    };

    int
    runPathModeSmoke() {
        const char *probe_path = std::getenv("NIXL_HF3FS_PROBE_PATH");
        if (!probe_path) {
            probe_path = "/mnt/3fs/nixl_hf3fs_path_mode_smoke.bin";
        }
        return nixl_test::runPathModeSmoke("HF3FSPathModeSmoke", "HF3FS", probe_path, 4096);
    }

    // Posts one WRITE request twice with different buffer contents, then reads the files back.
    // A request may be posted again once it is done, and each post must write the buffer content
    // at the time of that post.
    int
    runRepostTest(const std::string &test_files_dir_path, nixlTime::us_t wait_time, int max_waits) {
        constexpr int num_transfers = 16;
        constexpr size_t transfer_size = 128 * 1024;
        const std::string agent_name = "HF3FSRepostTester";

        print_segment_title("NIXL STORAGE REPOST TEST STARTING (HF3FS PLUGIN)");

        nixlAgentConfig cfg;
        cfg.useProgThread = true;
        nixlAgent agent(agent_name, cfg);

        // The DRAM path copies WRITE data into an IOV; DRAM_ZC writes from the buffer itself.
        nixl_b_params_t params = {{"mem_config", "dram"}};
        nixlBackendH *hf3fs = nullptr;
        if (agent.createBackend("HF3FS", params, hf3fs) != NIXL_SUCCESS) {
            std::cerr << "Failed to create HF3FS backend" << std::endl;
            return 1;
        }

        std::vector<std::unique_ptr<void, MallocDeleter>> dram_addr;
        dram_addr.reserve(num_transfers);
        std::vector<tempFile> fd;
        fd.reserve(num_transfers);
        nixl_reg_dlist_t dram_reg(DRAM_SEG);
        nixl_reg_dlist_t file_reg(FILE_SEG);
        nixl_xfer_dlist_t dram_xfer(DRAM_SEG);
        nixl_xfer_dlist_t file_xfer(FILE_SEG);

        for (int i = 0; i < num_transfers; i++) {
            void *ptr = aligned_alloc(page_size, transfer_size);
            if (!ptr) {
                std::cerr << "DRAM allocation failed" << std::endl;
                return 1;
            }
            dram_addr.emplace_back(ptr);

            std::string name = test_files_dir_path + "/" +
                generate_timestamped_filename("repost_" + std::string(test_file_name)) + "_" +
                std::to_string(i);
            try {
                fd.emplace_back(name, O_RDWR | O_CREAT | O_TRUNC, std_file_permissions);
            }
            catch (const std::exception &e) {
                std::cerr << "Failed to create file: " << name << " - " << e.what() << std::endl;
                return 1;
            }

            nixlBlobDesc dram_desc;
            dram_desc.addr = (uintptr_t)ptr;
            dram_desc.len = transfer_size;
            dram_desc.devId = 0;
            dram_reg.addDesc(dram_desc);
            dram_xfer.addDesc(dram_desc);

            nixlBlobDesc file_desc;
            file_desc.addr = 0;
            file_desc.len = transfer_size;
            file_desc.devId = fd.back();
            file_reg.addDesc(file_desc);
            file_xfer.addDesc(file_desc);
        }

        if (agent.registerMem(dram_reg) != NIXL_SUCCESS ||
            agent.registerMem(file_reg) != NIXL_SUCCESS) {
            std::cerr << "Failed to register memory with NIXL" << std::endl;
            return 1;
        }

        auto post_and_wait = [&agent, wait_time, max_waits](nixlXferReqH *req) {
            nixl_status_t status = agent.postXferReq(req);
            for (int num_waits = 0; status == NIXL_IN_PROG; num_waits++) {
                if (num_waits >= max_waits) {
                    std::cerr << "Transfer timed out after " << max_waits << " retries"
                              << std::endl;
                    break;
                }
                usleep(wait_time);
                status = agent.getXferStatus(req);
            }
            return status;
        };

        nixlXferReqH *write_req = nullptr;
        if (agent.createXferReq(NIXL_WRITE, dram_xfer, file_xfer, agent_name, write_req) !=
            NIXL_SUCCESS) {
            std::cerr << "Failed to create write transfer request" << std::endl;
            return 1;
        }

        const char *phrases[] = {repost_test_phrase_1, repost_test_phrase_2};
        for (int post = 0; post < 2; post++) {
            print_segment_title(phase_title(post == 0 ? "First post of the write request" :
                                                        "Second post of the same request"));
            for (auto &buf : dram_addr) {
                fill_test_pattern(buf.get(), transfer_size, phrases[post]);
            }
            nixl_status_t status = post_and_wait(write_req);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Write transfer failed - status: "
                          << nixlEnumStrings::statusStr(status) << std::endl;
                agent.releaseXferReq(write_req);
                return 1;
            }
        }
        agent.releaseXferReq(write_req);

        print_segment_title(phase_title("Reading the files back"));
        for (auto &buf : dram_addr) {
            clear_buffer(buf.get(), transfer_size);
        }
        nixlXferReqH *read_req = nullptr;
        if (agent.createXferReq(NIXL_READ, dram_xfer, file_xfer, agent_name, read_req) !=
            NIXL_SUCCESS) {
            std::cerr << "Failed to create read transfer request" << std::endl;
            return 1;
        }
        nixl_status_t status = post_and_wait(read_req);
        agent.releaseXferReq(read_req);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Read transfer failed - status: " << nixlEnumStrings::statusStr(status)
                      << std::endl;
            return 1;
        }

        auto expected = std::make_unique<char[]>(transfer_size);
        fill_test_pattern(expected.get(), transfer_size, repost_test_phrase_2);
        int mismatched = 0;
        for (auto &buf : dram_addr) {
            mismatched += memcmp(buf.get(), expected.get(), transfer_size) != 0;
        }

        agent.deregisterMem(file_reg);
        agent.deregisterMem(dram_reg);

        if (mismatched != 0) {
            std::cerr << mismatched << " of " << num_transfers
                      << " files do not hold the data of the second post" << std::endl;
            return 1;
        }
        std::cout << "Repost test passed" << std::endl;
        return 0;
    }
}

int main(int argc, char *argv[])
{
    // Defaults
    int num_transfers = default_num_transfers;
    size_t transfer_size = default_transfer_size;
    std::string test_files_dir_path = default_test_files_dir_path;
    nixlTime::us_t wait_time = default_wait_time;
    int max_waits = default_max_waits;
    bool filled_after_mem_register = false;
    bool run_path_mode_smoke = true;

    // getopt argument parsing
    int opt;
    while ((opt = getopt(argc, argv, "hn:s:d:w:m:l:P")) != -1) {
        switch (opt) {
        case 'l':
            if (strcmp(optarg, "before") == 0) {
                filled_after_mem_register = false;
            } else if (strcmp(optarg, "after") == 0) {
                filled_after_mem_register = true;
            } else {
                std::cerr << "Invalid value for -l (filled_timing): " << optarg << std::endl;
            }
            break;
        case 'n':
            try {
                num_transfers = std::stoi(optarg);
            }
            catch (...) {
                std::cerr << "Invalid value for -n (num_transfers): " << optarg << std::endl;
                return 1;
            }
            break;
        case 's':
            try {
                transfer_size = std::stoul(optarg);
            }
            catch (...) {
                std::cerr << "Invalid value for -s (transfer_size): " << optarg << std::endl;
                return 1;
            }
            break;
        case 'd':
            test_files_dir_path = optarg;
            break;
        case 'w':
            try {
                wait_time = static_cast<nixlTime::us_t>(std::stoul(optarg));
            }
            catch (...) {
                std::cerr << "Invalid value for -w (wait_time): " << optarg << std::endl;
                return 1;
            }
            break;
        case 'm':
            try {
                max_waits = std::stoi(optarg);
            }
            catch (...) {
                std::cerr << "Invalid value for -m (max_waits): " << optarg << std::endl;
                return 1;
            }
            break;
        case 'P':
            run_path_mode_smoke = false;
            break;
        case 'h':
        default:
            std::cout << "Usage: " << argv[0]
                      << " [-n num_transfers] [-s transfer_size] [-d test_files_dir_path] [-w "
                         "wait_time] [-m max_waits] [-P]"
                      << std::endl;
            std::cout << "  -n num_transfers      Number of transfers (default: "
                      << default_num_transfers << ")" << std::endl;
            std::cout << "  -s transfer_size      Size of each transfer in bytes (default: "
                      << default_transfer_size << ")" << std::endl;
            std::cout << "  -d test_files_dir_path Directory for test files (default: "
                      << default_test_files_dir_path << ")" << std::endl;
            std::cout << "                        Note: This directory must exist and be writable "
                         "by the current user."
                      << std::endl;
            std::cout << "  -w wait_time          Wait time in microseconds (default: "
                      << default_wait_time << ")" << std::endl;
            std::cout << "  -m max_waits          Maximum number of waits (default: "
                      << default_max_waits << ")" << std::endl;
            std::cout << "  -l                    Filled memory before or after memory register "
                         "with NIXL: [before, after]"
                      << std::endl;
            std::cout << "  -P                    Skip path-mode smoke (enabled by default)"
                      << std::endl;
            std::cout << "  -h                    Show this help message" << std::endl;
            return (opt == 'h') ? 0 : 1;
        }
    }

    if (page_size == 0) {
        std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
        return 1;
    }

    if (run_path_mode_smoke) {
        if (int rc = runPathModeSmoke(); rc != 0) {
            return rc;
        }
    }

    // Convert directory path to absolute path using std::filesystem
    std::filesystem::path path_obj(test_files_dir_path);
    try {
        std::cout << "Creating directory: " << path_obj.string() << std::endl;
        std::filesystem::create_directories(path_obj);
        std::string abs_path = std::filesystem::absolute(path_obj).string();
        std::cout << "Using absolute path: " << abs_path << std::endl;

        // Verify the directory is accessible
        if (!std::filesystem::exists(path_obj)) {
            std::cerr << "ERROR: Directory doesn't exist after creation: " << path_obj.string() << std::endl;
            return 1;
        }

        // Verify the directory is writable
        std::string test_file = (path_obj / "write_test").string();
        int test_fd = open(test_file.c_str(), O_RDWR|O_CREAT, 0644);
        if (test_fd == -1) {
            std::cerr << "ERROR: Directory is not writable: " << path_obj.string()
                      << " - " << strerror(errno) << " (errno: " << errno << ")" << std::endl;
            return 1;
        }
        close(test_fd);
        unlink(test_file.c_str());
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to create or access directory: " << path_obj.string()
                  << " - " << e.what() << std::endl;
        return 1;
    }

    // Initialize NIXL components
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    nixl_b_params_t   params;
    nixlBackendH      *hf3fs;
    nixl_reg_dlist_t  file_for_hf3fs(FILE_SEG);
    nixl_reg_dlist_t  dram_for_hf3fs(DRAM_SEG);
    nixlXferReqH      *treq;
    nixl_xfer_dlist_t file_for_hf3fs_list(FILE_SEG);
    nixl_xfer_dlist_t dram_for_hf3fs_list(DRAM_SEG);

    // Print test configuration information
    print_segment_title("NIXL STORAGE TEST STARTING (HF3FS PLUGIN)");
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Number of transfers: " << num_transfers << std::endl;
    std::cout << "- Transfer size: " << transfer_size << " bytes" << std::endl;
    std::cout << "- Total data: " << std::fixed << std::setprecision(2) << (float(transfer_size) * num_transfers) / gb_size << " GB" << std::endl;
    std::cout << "- Directory: " << test_files_dir_path << std::endl;
    std::cout << std::endl;
    std::cout << line_str << std::endl;

    // Create NIXL agent
    nixlAgent agent("HF3FSTester", cfg);

    print_segment_title(phase_title("Creating HF3FS backend"));

    // Create HF3FS backend
    nixl_status_t ret = agent.createBackend("HF3FS", params, hf3fs);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Error creating HF3FS backend: " << ret << std::endl;
        return 1;
    }

    if (hf3fs == nullptr) {
        std::cerr << "Error creating a new backend" << std::endl;
        return 1;
    }

    // Control variables for performance measurement
    nixl_status_t status = NIXL_SUCCESS;
    nixlTime::us_t time_start;
    nixlTime::us_t time_end;
    nixlTime::us_t time_duration;
    nixlTime::us_t total_time(0);
    double total_data_gb(0);
    double gbps;
    double seconds;
    double data_gb;
    int num_waits = 0;

    // Memory and file structures with RAII
    std::vector<std::unique_ptr<void, MallocDeleter>> dram_addr;
    dram_addr.reserve(num_transfers);
    std::vector<tempFile> fd;
    fd.reserve(num_transfers);
    std::unique_ptr<nixlBlobDesc[]> dram_buf(new nixlBlobDesc[num_transfers]);
    std::unique_ptr<nixlBlobDesc[]> ftrans(new nixlBlobDesc[num_transfers]);

    print_segment_title(phase_title("Allocating and initializing buffers"));

    for (int i = 0; i < num_transfers; i++) {
        void *ptr = aligned_alloc(page_size, transfer_size);
        if (!ptr) {
            std::cerr << "DRAM allocation failed" << std::endl;
            return 1;
        }
        dram_addr.emplace_back(ptr);
        if (!filled_after_mem_register) {
            fill_test_pattern(dram_addr.back().get(), transfer_size);
        }

        // Create and open test file
        std::string name = generate_timestamped_filename(test_file_name);
        name = test_files_dir_path + "/" + name + "_" + std::to_string(i);

        try {
            // Verify parent directory exists before creating file
            std::filesystem::path file_path(name);
            std::filesystem::path parent_dir = file_path.parent_path();
            if (!std::filesystem::exists(parent_dir)) {
                std::cerr << "Parent directory doesn't exist: " << parent_dir.string() << std::endl;
                std::cout << "Attempting to create parent directory..." << std::endl;
                std::filesystem::create_directories(parent_dir);
            }

            // Create file with more explicit flags and mode
            fd.emplace_back(name, O_RDWR|O_CREAT|O_TRUNC, std_file_permissions);
            int fd_val = fd.back();
            if (fd_val < 0) {
                // This should never happen since constructor throws, but checking anyway
                std::cerr << "ERROR: File descriptor is invalid after successful open: " << fd_val << std::endl;
                return 1;
            }

            // Verify file was created
            if (!std::filesystem::exists(file_path)) {
                std::cerr << "ERROR: File doesn't exist after creation: " << name << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to create file: " << name << " - " << e.what() << std::endl;
            return 1;
        }

        // Setup DRAM descriptors
        dram_buf[i].addr   = (uintptr_t)(dram_addr.back().get());
        dram_buf[i].len    = transfer_size;
        dram_buf[i].devId  = 0;
        dram_for_hf3fs.addDesc(dram_buf[i]);

        // Setup file descriptors
        ftrans[i].addr  = 0; // offset
        ftrans[i].len   = transfer_size;
        ftrans[i].devId = fd[i];
        file_for_hf3fs.addDesc(ftrans[i]);

        printProgress(float(i + 1) / num_transfers);
    }

    print_segment_title(phase_title("Registering memory with NIXL"));

    // Register memory with NIXL
    int i = 0;
    status = agent.registerMem(dram_for_hf3fs);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Failed to register DRAM memory with NIXL" << std::endl;
        return 1;
    }
    printProgress(float(++i) / 2);

    status = agent.registerMem(file_for_hf3fs);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Failed to register file memory with NIXL" << std::endl;
        return 1;
    }
    printProgress(float(++i) / 2);

    if (filled_after_mem_register) {
        for (i = 0; i < num_transfers; ++i) {
            fill_test_pattern(dram_addr[i].get(), transfer_size);
        }
    }

    print_segment_title(phase_title("Memory to File Transfer (Write Test)"));

    // Prepare for transfer
    file_for_hf3fs_list = file_for_hf3fs.trim();
    dram_for_hf3fs_list = dram_for_hf3fs.trim();

    // Connect the agent to itself for local transfer
       // Create transfer request
    std::cout << "Creating transfer request:" << std::endl;
    std::cout << "- Operation: NIXL_WRITE" << std::endl;
    std::cout << "- Source list count: " << dram_for_hf3fs_list.descCount() << std::endl;
    std::cout << "- Destination list count: " << file_for_hf3fs_list.descCount() << std::endl;
    std::cout << "- Remote agent: " << "HF3FSTester" << std::endl;

    ret = agent.createXferReq(NIXL_WRITE, dram_for_hf3fs_list, file_for_hf3fs_list,
                              "HF3FSTester", treq);

    if (ret != NIXL_SUCCESS) {
        std::cerr << "Error creating transfer request: " << ret << std::endl;
        return 1;
    }

    // Execute write transfer and measure performance
    time_start = nixlTime::getUs();
    status = agent.postXferReq(treq);
    if (status < 0) {
        std::cerr << "Failed to post write transfer request - status: " << status << std::endl;
        agent.releaseXferReq(treq);
        return 1;
    }

    // Wait for transfer to complete
    num_waits = 0;
    do {
        status = agent.getXferStatus(treq);
        if (status < 0) {
            std::cerr << "Error during write transfer - status: " << status << std::endl;
            agent.releaseXferReq(treq);
            return 1;
        }
        if (num_waits++ >= max_waits) {
            std::cerr << "Write operation timed out after " << max_waits << " retries" << std::endl;
            agent.releaseXferReq(treq);
            return 1;
        }
        usleep(wait_time);

        // Update progress based on waits
        float progress = static_cast<float>(num_waits) / max_waits;
        progress = progress > 1.0f ? 1.0f : progress;
        printProgress(progress);

    } while (status == NIXL_IN_PROG);

    time_end = nixlTime::getUs();
    time_duration = time_end - time_start;
    total_time += time_duration;

    data_gb = (float(transfer_size) * num_transfers) / gb_size;
    total_data_gb += data_gb;
    seconds = us_to_s(time_duration);
    gbps = data_gb / seconds;

    std::cout << "Write completed with status: " << status << std::endl;
    std::cout << "- Time: " << format_duration(time_duration) << std::endl;
    std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB" << std::endl;
    std::cout << "- Speed: " << gbps << " GB/s" << std::endl;

    print_segment_title(phase_title("Syncing files"));
    std::cout << "Syncing files to ensure data is written to disk" << std::endl;
    // Sync all files to ensure data is written to disk
    for (i = 0; i < num_transfers; ++i) {
        if (fsync(fd[i]) < 0) {
            std::cerr << "Failed to sync file " << i << " - " << strerror(errno) << std::endl;
            return 1;
        }
        printProgress(float(i + 1) / num_transfers);
    }

    print_segment_title(phase_title("Clearing DRAM buffers"));
    std::cout << "Clearing DRAM buffers" << std::endl;
    for (i = 0; i < num_transfers; ++i) {
        clear_buffer(dram_addr[i].get(), transfer_size);
        printProgress(float(i + 1) / num_transfers);
    }

    print_segment_title(phase_title("File to Memory Transfer (Read Test)"));

    // Create read transfer request
    ret = agent.createXferReq(NIXL_READ, dram_for_hf3fs_list, file_for_hf3fs_list,
                              "HF3FSTester", treq);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Error creating read transfer request: " << ret << std::endl;
        return 1;
    }

    // Execute read transfer and measure performance
    time_start = nixlTime::getUs();
    status = agent.postXferReq(treq);
    if (status < 0) {
        std::cerr << "Failed to post read transfer request - status: " << status << std::endl;
        agent.releaseXferReq(treq);
        return 1;
    }

    // Wait for transfer to complete
    num_waits = 0;
    do {
        status = agent.getXferStatus(treq);
        if (status < 0) {
            std::cerr << "Error during read transfer - status: " << status << std::endl;
            agent.releaseXferReq(treq);
            return 1;
        }
        if (num_waits++ >= max_waits) {
            std::cerr << "Read operation timed out after " << max_waits << " retries" << std::endl;
            agent.releaseXferReq(treq);
            return 1;
        }
        usleep(wait_time);

        // Update progress based on waits
        float progress = static_cast<float>(num_waits) / max_waits;
        progress = progress > 1.0f ? 1.0f : progress;
        printProgress(progress);

    } while (status == NIXL_IN_PROG);

    time_end = nixlTime::getUs();
    time_duration = time_end - time_start;
    total_time += time_duration;

    data_gb = (float(transfer_size) * num_transfers) / gb_size;
    total_data_gb += data_gb;
    seconds = us_to_s(time_duration);
    gbps = data_gb / seconds;

    std::cout << "Read completed with status: " << status << std::endl;
    std::cout << "- Time: " << format_duration(time_duration) << std::endl;
    std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB" << std::endl;
    std::cout << "- Speed: " << gbps << " GB/s" << std::endl;

    print_segment_title(phase_title("Validating read data"));

    std::unique_ptr<char[]> expected_buffer = std::make_unique<char[]>(transfer_size);
    fill_test_pattern(expected_buffer.get(), transfer_size);

    bool validation_passed = true;
    for (i = 0; i < num_transfers; ++i) {
        int ret = memcmp(dram_addr[i].get(), expected_buffer.get(), transfer_size);
        if (ret != 0) {
            std::cerr << "DRAM buffer " << i << " validation failed with error: " << ret << std::endl;

            // Find the first difference byte
            char* expected = expected_buffer.get();
            char* actual = static_cast<char*>(dram_addr[i].get());
            size_t diff_position = 0;

            for (size_t pos = 0; pos < transfer_size; pos++) {
                if (expected[pos] != actual[pos]) {
                    diff_position = pos;
                    break;
                }
            }

            // Display difference information
            std::cerr << "First difference at position " << diff_position << std::endl;

            // Show a few bytes before and after the difference position (up to 10 on each side)
            size_t start = (diff_position > 10) ? diff_position - 10 : 0;
            size_t end = std::min(diff_position + 10, transfer_size - 1);

            std::cerr << "Expected bytes (hex): ";
            for (size_t pos = start; pos <= end; pos++) {
                std::cerr << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(static_cast<unsigned char>(expected[pos])) << " ";
            }
            std::cerr << std::dec << std::endl;

            std::cerr << "Actual bytes (hex):   ";
            for (size_t pos = start; pos <= end; pos++) {
                std::cerr << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(static_cast<unsigned char>(actual[pos])) << " ";
            }
            std::cerr << std::dec << std::endl;

            validation_passed = false;
            break; // Exit the loop on first validation failure
        }
        printProgress(float(i + 1) / num_transfers);
    }

    if (validation_passed) {
        std::cout << "All data validated successfully!" << std::endl;
    } else {
        std::cerr << "Data validation failed. Test incomplete." << std::endl;
        return 1;
    }

    print_segment_title("Freeing resources");

    if (treq) {
        agent.releaseXferReq(treq);
    }

    agent.deregisterMem(file_for_hf3fs);
    agent.deregisterMem(dram_for_hf3fs);

    print_segment_title("TEST SUMMARY");
    std::cout << "Total time: " << format_duration(total_time) << std::endl;
    std::cout << "Total data: " << std::fixed << std::setprecision(2) << total_data_gb << " GB" << std::endl;
    std::cout << line_str << std::endl;

    return runRepostTest(test_files_dir_path, wait_time, max_waits);
}
