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
#include <sys/stat.h>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <string>
#include <absl/strings/str_format.h>
#include "nixl.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"
#include "common/nixl_time.h"
#include <stdexcept>
#include <cstdio>
#include <getopt.h>

namespace {
    const size_t page_size = sysconf(_SC_PAGESIZE);

    constexpr int default_num_transfers = 1024;
    constexpr size_t default_transfer_size = 1 * 512 * 1024; // 512KB
    constexpr char test_phrase[] = "NIXL Storage Test Pattern 2025 POSIX";
    constexpr size_t test_phrase_len = sizeof(test_phrase) - 1; // -1 to exclude null terminator
    constexpr char test_file_name[] = "testfile";
    constexpr mode_t std_file_permissions = 0744;

    constexpr size_t kb_size = 1024;
    constexpr size_t mb_size = 1024 * 1024;
    constexpr size_t gb_size = 1024 * 1024 * 1024;
    constexpr double us_to_s(double us) { return us / 1000000.0; }

    constexpr int line_width = 60;
    constexpr int progress_bar_width = line_width - 2; // -2 for the brackets
    const std::string line_str(line_width, '=');
    std::string center_str(const std::string& str) {
        return std::string((line_width - str.length()) / 2, ' ') + str;
    }

    constexpr char default_test_files_dir_path[] = "tmp/testfiles";

    // Custom deleter for posix_memalign allocated memory
    struct PosixMemalignDeleter {
        void operator()(void* ptr) const {
            if (ptr) free(ptr);
        }
    };

    // Helper function to fill buffer with repeating pattern
    void fill_test_pattern(void* buffer, size_t size) {
        char* buf = (char*)buffer;
        size_t phrase_len = test_phrase_len;
        size_t offset = 0;

        while (offset < size) {
            size_t remaining = size - offset;
            size_t copy_len = (remaining < phrase_len) ? remaining : phrase_len;
            memcpy(buf + offset, test_phrase, copy_len);
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
            return absl::StrFormat("%.0f ms", ms);
        }
        double seconds = ms / 1000.0;
        return absl::StrFormat("%.3f sec", seconds);
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
        std::cout << "[";
        int pos = progress_bar_width * progress;
        for (int i = 0; i < progress_bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << absl::StrFormat("] %.1f%% ", progress * 100.0);

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
        return absl::StrFormat("PHASE %d: %s", phase_num++, title);
    }

    void print_segment_title(const std::string& title) {
        std::cout << std::endl << line_str << std::endl;
        std::cout << center_str(title) << std::endl;
        std::cout << line_str << std::endl;
    }

    class tempFile {
    public:
        int fd;
        std::string path;

        // Constructor: opens the file and stores the fd and path
        tempFile(const std::string& filename, int flags, mode_t mode = 0600)
            : path(filename)
        {
            fd = open(filename.c_str(), flags, mode);
            if (fd == -1) {
                throw std::runtime_error("Failed to open file: " + filename);
            }
        }

        // Deleted copy constructor and assignment to avoid double-close/unlink
        tempFile(const tempFile&) = delete;
        tempFile& operator=(const tempFile&) = delete;

        // Move constructor and assignment
        tempFile(tempFile&& other) noexcept
            : fd(other.fd), path(std::move(other.path))
        {
            other.fd = -1;
        }
        tempFile& operator=(tempFile&& other) noexcept {
            if (this != &other) {
                close_fd();
                path = std::move(other.path);
                fd = other.fd;
                other.fd = -1;
            }
            return *this;
        }

        // Conversion operator to int (file descriptor)
        operator int() const { return fd; }

        // Destructor: closes the fd and deletes the file
        ~tempFile() {
            close_fd();
            if (!path.empty()) {
                unlink(path.c_str());
            }
        }

    private:
        void close_fd() {
            if (fd != -1) {
                close(fd);
                fd = -1;
            }
        }
    };
}

int main(int argc, char *argv[])
{
    std::cout << "NIXL POSIX Plugin Test" << std::endl;

    int                opt;
    int                num_transfers = default_num_transfers;
    size_t             transfer_size = default_transfer_size;
    std::string        test_files_dir_path = default_test_files_dir_path;
    bool               use_direct_io = false;  // New option for O_DIRECT
    bool               use_uring = false;      // New option for io_uring
    long               page_size = sysconf(_SC_PAGESIZE);

    while ((opt = getopt(argc, argv, "n:s:d:DUh")) != -1) {
        switch (opt) {
            case 'n':
                num_transfers = std::stoi(optarg);
                break;
            case 's':
                transfer_size = std::stoull(optarg);
                break;
            case 'd':
                test_files_dir_path = optarg;
                break;
            case 'D':
                use_direct_io = true;
                break;
            case 'U':
                use_uring = true;
                break;
            case 'h':
            default:
                std::cout << absl::StrFormat("Usage: %s [-n num_transfers] [-s transfer_size] [-d test_files_dir_path] [-D] [-U]", argv[0]) << std::endl;
                std::cout << absl::StrFormat("  -n num_transfers      Number of transfers (default: %d)", default_num_transfers) << std::endl;
                std::cout << absl::StrFormat("  -s transfer_size      Size of each transfer in bytes (default: %zu)", default_transfer_size) << std::endl;
                std::cout << absl::StrFormat("  -d test_files_dir_path Directory for test files, strongly recommended to use nvme device (default: %s)", default_test_files_dir_path) << std::endl;
                std::cout << absl::StrFormat("  -D                    Use O_DIRECT for file I/O") << std::endl;
                std::cout << absl::StrFormat("  -U                    Use io_uring backend instead of AIO") << std::endl;
                std::cout << absl::StrFormat("  -h                    Show this help message") << std::endl;
                return (opt == 'h') ? 0 : 1;
        }
    }

    if (page_size == 0) {
        std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
        return 1;
    }

    // If using O_DIRECT, align transfer size to page size
    if (use_direct_io) {
        if (transfer_size % page_size != 0) {
            transfer_size = ((transfer_size + page_size - 1) / page_size) * page_size;
            std::cout << "Adjusted transfer size to " << transfer_size << " bytes for O_DIRECT alignment" << std::endl;
        }
    }

    // Convert directory path to absolute path using std::filesystem
    std::filesystem::path path_obj(test_files_dir_path);
    std::filesystem::create_directories(path_obj);
    std::string abs_path = std::filesystem::absolute(path_obj).string();

    // Initialize NIXL components first
    nixlAgent agent("POSIXTester", nixlAgentConfig(true));

    // Set up backend parameters
    nixl_b_params_t params;
    if (use_uring) {
        // Explicitly request io_uring
        params["use_uring"] = "true";
        params["use_aio"] = "false";
    } else {
        // Explicitly request AIO
        params["use_aio"] = "true";
        params["use_uring"] = "false";
    }

    if (use_direct_io) {
        params["use_direct_io"] = "true";
    }

    // Print test configuration information
    print_segment_title("NIXL STORAGE TEST STARTING (POSIX PLUGIN)");
    std::cout << absl::StrFormat("Configuration:\n");
    std::cout << absl::StrFormat("- Number of transfers: %d\n", num_transfers);
    std::cout << absl::StrFormat("- Transfer size: %zu bytes\n", transfer_size);
    std::cout << absl::StrFormat("- Total data: %.2f GB\n", (float(transfer_size) * num_transfers) / gb_size);
    std::cout << absl::StrFormat("- Directory: %s\n", abs_path);
    std::cout << absl::StrFormat("- Backend: %s\n", use_uring ? "io_uring" : "AIO");
    std::cout << absl::StrFormat("- Direct I/O: %s\n", use_direct_io ? "enabled" : "disabled");
    std::cout << std::endl;
    std::cout << line_str << std::endl;

    // Create POSIX backend first - before allocating any resources
    nixlBackendH* posix = nullptr;
    nixl_status_t status = agent.createBackend("POSIX", params, posix);
    if (status != NIXL_SUCCESS) {
        std::cerr << std::endl << line_str << std::endl;
        std::cerr << center_str("ERROR: Backend Creation Failed") << std::endl;
        std::cerr << line_str << std::endl;
        std::cerr << "Error creating POSIX backend: " << nixlEnumStrings::statusStr(status) << std::endl;
        if (use_uring) {
            std::cerr << "io_uring was requested but may not be available. Try running without -U flag to use AIO instead." << std::endl;
        }
        std::cerr << std::endl << line_str << std::endl;
        return 1;
    }

    // Only proceed with resource allocation if backend creation succeeded
    try {
        print_segment_title("PHASE 1: Allocating and initializing buffers");

        // Allocate resources
        std::vector<std::unique_ptr<void, PosixMemalignDeleter>> dram_addr;
        dram_addr.reserve(num_transfers);

        std::vector<tempFile> fd;
        fd.reserve(num_transfers);

        // File open flags
        int file_open_flags = O_RDWR|O_CREAT;
        if (use_direct_io) {
            file_open_flags |= O_DIRECT;
        }
        mode_t file_mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;  // rw-r--r--

        // Create descriptor lists
        nixl_reg_dlist_t dram_for_posix(DRAM_SEG);
        nixl_reg_dlist_t file_for_posix(FILE_SEG);
        nixl_xfer_dlist_t dram_for_posix_xfer(DRAM_SEG);
        nixl_xfer_dlist_t file_for_posix_xfer(FILE_SEG);
        std::unique_ptr<nixlBlobDesc[]> dram_buf(new nixlBlobDesc[num_transfers]);
        std::unique_ptr<nixlBlobDesc[]> ftrans(new nixlBlobDesc[num_transfers]);
        nixlXferReqH* treq = nullptr;
        std::string name;

        // Control variables
        int ret = 0;
        int i = 0;
        nixlTime::us_t time_start;
        nixlTime::us_t time_end;
        nixlTime::us_t time_duration;
        nixlTime::us_t total_time(0);
        double total_data_gb(0);
        double gbps;
        double seconds;
        double data_gb;

        // Allocate and initialize DRAM buffer
        for (i = 0; i < num_transfers; ++i) {
            void* ptr;
            if (posix_memalign(&ptr, page_size, transfer_size) != 0) {
                std::cerr << "DRAM allocation failed" << std::endl;
                return 1;
            }
            dram_addr.emplace_back(ptr);
            fill_test_pattern(dram_addr.back().get(), transfer_size);

            // Create test file
            name = generate_timestamped_filename(test_file_name);
            name = test_files_dir_path + "/" + name + "_" + std::to_string(i);

            try {
                fd.emplace_back(name, file_open_flags, file_mode);
            } catch (const std::exception& e) {
                std::cerr << "Failed to open file: " << name << " - " << e.what() << std::endl;
                return 1;
            }

            dram_buf[i].addr   = (uintptr_t)(dram_addr.back().get());
            dram_buf[i].len    = transfer_size;
            dram_buf[i].devId  = 0;
            dram_for_posix.addDesc(dram_buf[i]);
            dram_for_posix_xfer.addDesc(dram_buf[i]);

            ftrans[i].addr  = 0;
            ftrans[i].len   = transfer_size;
            ftrans[i].devId = fd[i];
            file_for_posix.addDesc(ftrans[i]);
            file_for_posix_xfer.addDesc(ftrans[i]);

            printProgress(float(i + 1) / num_transfers);
        }

        print_segment_title(phase_title("Registering memory with NIXL"));

        i = 0;
        ret = agent.registerMem(dram_for_posix);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM memory with NIXL" << std::endl;
            return 1;
        }
        printProgress(float(++i) / 2);

        ret = agent.registerMem(file_for_posix);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register file memory with NIXL" << std::endl;
            return 1;
        }
        printProgress(float(i + 1) / 2);

        print_segment_title(phase_title("Memory to File Transfer (Write Test)"));

        status = agent.createXferReq(NIXL_WRITE, dram_for_posix_xfer, file_for_posix_xfer,
                                     "POSIXTester", treq);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to create write transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            return 1;
        }

        time_start = nixlTime::getUs();
        status = agent.postXferReq(treq);
        if (status < 0) {
            std::cerr << "Failed to post write transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            agent.releaseXferReq(treq);
            return 1;
        }

        // Wait for transfer to complete
        do {
            status = agent.getXferStatus(treq);
            if (status < 0) {
                std::cerr << "Error during write transfer - status: " << nixlEnumStrings::statusStr(status) << std::endl;
                agent.releaseXferReq(treq);
                return 1;
            }
        } while (status == NIXL_IN_PROG);

        time_end = nixlTime::getUs();
        time_duration = time_end - time_start;
        total_time += time_duration;

        data_gb = (float(transfer_size) * num_transfers) / (gb_size);
        total_data_gb += data_gb;
        seconds = us_to_s(time_duration);
        gbps = data_gb / seconds;

        std::cout << "Write completed with status: " << nixlEnumStrings::statusStr(status) << std::endl;
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

        status = agent.createXferReq(NIXL_READ, dram_for_posix_xfer, file_for_posix_xfer,
                                     "POSIXTester", treq);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to create read transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            return 1;
        }

        // Execute read transfer and measure performance
        time_start = nixlTime::getUs();
        status = agent.postXferReq(treq);
        if (status < 0) {
            std::cerr << "Failed to post read transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            agent.releaseXferReq(treq);
            return 1;
        }

        // Wait for transfer to complete
        do {
            status = agent.getXferStatus(treq);
            if (status < 0) {
                std::cerr << "Error during read transfer - status: " << nixlEnumStrings::statusStr(status) << std::endl;
                agent.releaseXferReq(treq);
                return 1;
            }
        } while (status == NIXL_IN_PROG);

        time_end = nixlTime::getUs();
        time_duration = time_end - time_start;
        total_time += time_duration;

        data_gb = (float(transfer_size) * num_transfers) / (gb_size);
        total_data_gb += data_gb;
        seconds = us_to_s(time_duration);
        gbps = data_gb / seconds;

        std::cout << "Read completed with status: " << nixlEnumStrings::statusStr(status) << std::endl;
        std::cout << "- Time: " << format_duration(time_duration) << std::endl;
        std::cout << "- Data: " << std::fixed << std::setprecision(2) << data_gb << " GB" << std::endl;
        std::cout << "- Speed: " << gbps << " GB/s" << std::endl;

        print_segment_title(phase_title("Validating read data"));

        std::unique_ptr<char[]> expected_buffer = std::make_unique<char[]>(transfer_size);
        fill_test_pattern(expected_buffer.get(), transfer_size);

        for (i = 0; i < num_transfers; ++i) {
            int ret = memcmp(dram_addr[i].get(), expected_buffer.get(), transfer_size);
            if (ret != 0) {
                std::cerr << "DRAM buffer " << i << " validation failed with error: " << ret << std::endl;
                return 1;
            }
            printProgress(float(i + 1) / num_transfers);
        }

        print_segment_title("Freeing resources");

        if (treq) {
            agent.releaseXferReq(treq);
        }

        agent.deregisterMem(file_for_posix);
        agent.deregisterMem(dram_for_posix);

        print_segment_title("TEST SUMMARY");
        std::cout << "Total time: " << format_duration(total_time) << std::endl;
        std::cout << "Total data: " << std::fixed << std::setprecision(2) << total_data_gb << " GB" << std::endl;
        std::cout << line_str << std::endl;

        return ret;
    } catch (const std::exception& e) {
        std::cerr << "Exception during test execution: " << e.what() << std::endl;
        return 1;
    }
}
