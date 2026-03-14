/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Dell Technologies Inc. All rights reserved.
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
#include <iomanip>
#include <unistd.h>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <absl/strings/str_format.h>
#include "nixl.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"
#include "common/nixl_time.h"
#include <stdexcept>
#include <cstdio>
#include <getopt.h>
#include <filesystem>
#include <spawn.h>
#include <sys/wait.h>

namespace {
    const size_t page_size = sysconf(_SC_PAGESIZE);

    constexpr int default_num_transfers = 1024;
    constexpr size_t default_transfer_size = 1 * 512 * 1024; // 512KB
    constexpr char repost_test_phrase_1[] = "NIXL Storage Test Pattern 2026 LIBBLKIO 1111";
    constexpr char repost_test_phrase_2[] = "NIXL Storage Test Pattern 2026 LIBBLKIO 2222";
    static_assert (sizeof (repost_test_phrase_1) == sizeof (repost_test_phrase_2),
                   "Test phrases must be the same length");
    constexpr char read_write_test_phrase[] = "NIXL Storage Test Pattern 2026 LIBBLKIO";
    constexpr char test_file_name[] = "testfile";
    constexpr mode_t std_file_permissions = 0744;

    constexpr size_t kb_size = 1024;
    constexpr size_t mb_size = 1024 * 1024;
    constexpr size_t gb_size = 1024 * 1024 * 1024;

    constexpr int line_width = 60;
    constexpr int progress_bar_width = line_width - 2; // -2 for the brackets
    const std::string line_str(line_width, '=');
    int phase_num = 1;

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
    void fill_test_pattern(void *buffer, const char *test_phrase, size_t size) {
        char* buf = (char*)buffer;
        size_t phrase_len = strlen(test_phrase);
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

    // Helper function to print phase header
    void print_phase_header(const std::string& phase_name) {
        std::cout << "\n" << line_str << std::endl;
        std::cout << center_str(absl::StrFormat("Phase %d: %s", phase_num++, phase_name)) << std::endl;
        std::cout << line_str << std::endl;
    }

    // Helper function to print phase footer
    void print_phase_footer() {
        std::cout << line_str << std::endl;
        std::cout << center_str("Phase Complete") << std::endl;
        std::cout << line_str << std::endl;
    }

    // Test configuration structure
    struct TestConfig {
        int num_transfers = default_num_transfers;
        size_t transfer_size = default_transfer_size;
        bool enable_direct_io = false;
        bool enable_io_polling = false;
        std::string device_path;
        std::string test_files_dir = default_test_files_dir_path;
    };

    // Test statistics structure
    struct TestStats {
        nixlTime::us_t total_time = 0;
        double throughput_mbps = 0.0;
        size_t total_bytes = 0;
        int successful_transfers = 0;
        int failed_transfers = 0;
    };

    // Safe helper functions to avoid shell injection
    bool safe_create_file(const std::string& path, size_t size_mb) {
        int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd == -1) {
            std::cerr << "Failed to create file: " << path << std::endl;
            return false;
        }
        
        // Write zeros using fallocate if available, otherwise write manually
        if (posix_fallocate(fd, 0, size_mb * 1024 * 1024) != 0) {
            // Fallback to manual write
            std::vector<char> zeros(1024 * 1024, 0); // 1MB buffer
            size_t remaining = size_mb * 1024 * 1024;
            while (remaining > 0) {
                size_t to_write = std::min(remaining, zeros.size());
                ssize_t written = write(fd, zeros.data(), to_write);
                if (written <= 0) {
                    close(fd);
                    return false;
                }
                remaining -= written;
            }
        }
        close(fd);
        return true;
    }

    bool safe_execute_command(const std::string& executable, const std::vector<std::string>& args) {
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(executable.c_str()));
        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        pid_t pid;
        int status = posix_spawnp(&pid, executable.c_str(), nullptr, nullptr, argv.data(), nullptr);
        if (status != 0) {
            std::cerr << "Failed to spawn " << executable << ": " << strerror(errno) << std::endl;
            return false;
        }

        if (waitpid(pid, &status, 0) == -1) {
            std::cerr << "Failed to wait for " << executable << std::endl;
            return false;
        }

        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }

    std::string safe_execute_command_with_output(const std::string& executable, const std::vector<std::string>& args) {
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(executable.c_str()));
        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        int pipefd[2];
        if (pipe(pipefd) == -1) {
            std::cerr << "Failed to create pipe" << std::endl;
            return "";
        }

        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, pipefd[0]);
        posix_spawn_file_actions_addclose(&actions, pipefd[1]);

        pid_t pid;
        int status = posix_spawnp(&pid, executable.c_str(), &actions, nullptr, argv.data(), nullptr);
        posix_spawn_file_actions_destroy(&actions);

        close(pipefd[1]);

        if (status != 0) {
            close(pipefd[0]);
            std::cerr << "Failed to spawn " << executable << ": " << strerror(errno) << std::endl;
            return "";
        }

        std::string result;
        char buffer[256];
        ssize_t bytes_read;
        while ((bytes_read = read(pipefd[0], buffer, sizeof(buffer))) > 0) {
            result.append(buffer, bytes_read);
        }
        close(pipefd[0]);

        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            return "";
        }

        return result;
    }

    // Create test directory and setup loop device
    bool setup_test_environment(TestConfig& config) {
        // Validate test_files_dir to prevent path traversal
        if (config.test_files_dir.empty() || 
            config.test_files_dir.find("..") != std::string::npos ||
            config.test_files_dir.find(";") != std::string::npos ||
            config.test_files_dir.find("&") != std::string::npos ||
            config.test_files_dir.find("|") != std::string::npos ||
            config.test_files_dir.find("`") != std::string::npos ||
            config.test_files_dir.find("$") != std::string::npos) {
            std::cerr << "Invalid test directory path: " << config.test_files_dir << std::endl;
            return false;
        }

        // Create test directory safely
        std::error_code ec;
        if (!std::filesystem::create_directories(config.test_files_dir, ec)) {
            if (ec) {
                std::cerr << "Failed to create test directory: " << config.test_files_dir 
                          << " - " << ec.message() << std::endl;
                return false;
            }
        }

        // Create a test file for loop device safely
        const std::string test_file = config.test_files_dir + "/" + test_file_name;
        if (!safe_create_file(test_file, 100)) { // 100MB file
            std::cerr << "Failed to create test file: " << test_file << std::endl;
            return false;
        }

        // Setup loop device (this requires root privileges)
        if (!safe_execute_command("losetup", {"-f", test_file})) {
            std::cerr << "Failed to setup loop device (requires root privileges)" << std::endl;
            std::cerr << "Please run: sudo losetup -f " << test_file << std::endl;
            return false;
        }

        // Find the loop device that was created safely
        std::string output = safe_execute_command_with_output("losetup", {"-j", test_file});
        if (output.empty()) {
            std::cerr << "Failed to find loop device" << std::endl;
            return false;
        }

        // Parse output: format is "/dev/loopX: /path/to/file"
        size_t colon_pos = output.find(':');
        if (colon_pos == std::string::npos) {
            std::cerr << "Invalid losetup output format" << std::endl;
            return false;
        }

        config.device_path = output.substr(0, colon_pos);
        // Remove any trailing whitespace
        config.device_path.erase(config.device_path.find_last_not_of(" \t\n\r") + 1);
        std::cout << "Using loop device: " << config.device_path << std::endl;

        return true;
    }

    // Cleanup test environment
    void cleanup_test_environment(const TestConfig& config) {
        std::string test_file = config.test_files_dir + "/" + test_file_name;
        
        // Detach loop device safely
        std::string output = safe_execute_command_with_output("losetup", {"-j", test_file});
        if (!output.empty()) {
            // Parse output to get device path
            size_t colon_pos = output.find(':');
            if (colon_pos != std::string::npos) {
                std::string device_path = output.substr(0, colon_pos);
                device_path.erase(device_path.find_last_not_of(" \t\n\r") + 1);
                
                // Detach the specific loop device
                safe_execute_command("losetup", {"-d", device_path});
            }
        }

        // Remove test directory safely
        std::error_code ec;
        std::filesystem::remove_all(config.test_files_dir, ec);
        if (ec) {
            std::cerr << "Warning: Failed to remove test directory: " << config.test_files_dir 
                      << " - " << ec.message() << std::endl;
        }
    }

    // Test basic backend creation and destruction
    bool test_backend_creation(const TestConfig& config) {
        print_phase_header("Backend Creation Test");

        try {
            nixlAgent agent("libblkio_test_agent", nixlAgentConfig(true));

            nixl_b_params_t params;
            params["api_type"] = "IO_URING";
            params["device_list"] = "1:B:" + config.device_path;
            params["direct_io"] = config.enable_direct_io ? "1" : "0";
            params["io_polling"] = config.enable_io_polling ? "1" : "0";

            nixlBackendH* backend = nullptr;
            nixl_status_t status = agent.createBackend("LIBBLKIO", params, backend);

            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to create libblkio backend: " << status << std::endl;
                return false;
            }

            if (!backend) {
                std::cerr << "Backend handle is null" << std::endl;
                return false;
            }

            std::cout << "✓ Backend created successfully" << std::endl;
            std::cout << "  - API Type: IO_URING" << std::endl;
            std::cout << "  - Device: " << config.device_path << std::endl;
            std::cout << "  - Direct I/O: " << (config.enable_direct_io ? "enabled" : "disabled") << std::endl;
            std::cout << "  - I/O Polling: " << (config.enable_io_polling ? "enabled" : "disabled") << std::endl;

            // Note: nixlAgent destructor handles backend cleanup

            std::cout << "✓ Backend destroyed successfully" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Exception in backend creation test: " << e.what() << std::endl;
            return false;
        }

        print_phase_footer();
        return true;
    }

    // Test invalid API type handling
    bool test_invalid_api_type(const TestConfig& config) {
        print_phase_header("Invalid API Type Test");

        try {
            nixlAgent agent("libblkio_test_agent", nixlAgentConfig(true));

            nixl_b_params_t params;
            params["api_type"] = "INVALID_API";
            params["device_list"] = "1:B:" + config.device_path;
            params["direct_io"] = "0";

            nixlBackendH* backend = nullptr;
            nixl_status_t status = agent.createBackend("LIBBLKIO", params, backend);

            // Backend should succeed but default to io_uring (with warning logged)
            if (status != NIXL_SUCCESS) {
                std::cerr << "✗ Backend creation failed unexpectedly: " << status << std::endl;
                return false;
            }

            std::cout << "✓ Invalid API type handled gracefully (defaulted to io_uring)" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Exception in invalid API type test: " << e.what() << std::endl;
            return false;
        }

        print_phase_footer();
        return true;
    }

    // Test memory registration
    bool test_memory_registration(const TestConfig& config) {
        print_phase_header("Memory Registration Test");

        try {
            nixlAgent agent("libblkio_test_agent", nixlAgentConfig(true));

            nixl_b_params_t params;
            params["api_type"] = "IO_URING";
            params["device_list"] = "1:B:" + config.device_path;
            params["direct_io"] = config.enable_direct_io ? "1" : "0";

            nixlBackendH* backend = nullptr;
            nixl_status_t status = agent.createBackend("LIBBLKIO", params, backend);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to create backend: " << status << std::endl;
                return false;
            }

            // Allocate test buffer
            std::unique_ptr<void, PosixMemalignDeleter> buffer;
            void* buf_ptr = nullptr;
            int ret = posix_memalign(&buf_ptr, page_size, config.transfer_size);
            if (ret != 0) {
                std::cerr << "Failed to allocate aligned memory" << std::endl;
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }
            buffer.reset(buf_ptr);

            // Fill buffer with test pattern
            fill_test_pattern(buffer.get(), read_write_test_phrase, config.transfer_size);

            // Register memory
            nixl_reg_dlist_t desc_list(DRAM_SEG);
            nixlBlobDesc desc;
            desc.addr = reinterpret_cast<uintptr_t>(buffer.get());
            desc.len = config.transfer_size;
            desc.devId = 1; // Device ID from device_list
            desc.metaInfo = ""; // Empty metadata for DRAM

            desc_list.addDesc(desc);
            status = agent.registerMem(desc_list);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to register DRAM memory: " << status << std::endl;
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            std::cout << "✓ DRAM memory registered successfully" << std::endl;

            // Register block device memory
            nixl_reg_dlist_t blk_desc_list(BLK_SEG);
            nixlBlobDesc blk_desc;
            blk_desc.addr = 0; // Offset on device
            blk_desc.len = config.transfer_size;
            blk_desc.devId = 1; // Same device ID
            blk_desc.metaInfo = config.device_path; // Device path as metadata

            blk_desc_list.addDesc(blk_desc);
            status = agent.registerMem(blk_desc_list);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to register BLK memory: " << status << std::endl;
                // Note: Already handled by desc_list deregistration
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            std::cout << "✓ BLK memory registered successfully" << std::endl;

            // Deregister memory
            status = agent.deregisterMem(desc_list);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to deregister DRAM memory: " << status << std::endl;
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            status = agent.deregisterMem(blk_desc_list);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to deregister BLK memory: " << status << std::endl;
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            std::cout << "✓ Memory deregistered successfully" << std::endl;

            // Destroy backend
            // Note: nixlAgent destructor handles backend cleanup

        } catch (const std::exception& e) {
            std::cerr << "Exception in memory registration test: " << e.what() << std::endl;
            return false;
        }

        print_phase_footer();
        return true;
    }

    // Test basic I/O operations
    bool test_basic_io(const TestConfig& config) {
        print_phase_header("Basic I/O Test");

        try {
            nixlAgent agent("libblkio_test_agent", nixlAgentConfig(true));

            nixl_b_params_t params;
            params["api_type"] = "IO_URING";
            params["device_list"] = "1:B:" + config.device_path;
            params["direct_io"] = config.enable_direct_io ? "1" : "0";

            nixlBackendH* backend = nullptr;
            nixl_status_t status = agent.createBackend("LIBBLKIO", params, backend);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to create backend: " << status << std::endl;
                return false;
            }

            // Allocate test buffers
            std::unique_ptr<void, PosixMemalignDeleter> write_buffer;
            std::unique_ptr<void, PosixMemalignDeleter> read_buffer;
            void* write_ptr = nullptr;
            void* read_ptr = nullptr;

            int ret = posix_memalign(&write_ptr, page_size, config.transfer_size);
            if (ret != 0) {
                std::cerr << "Failed to allocate write buffer" << std::endl;
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }
            write_buffer.reset(write_ptr);

            ret = posix_memalign(&read_ptr, page_size, config.transfer_size);
            if (ret != 0) {
                std::cerr << "Failed to allocate read buffer" << std::endl;
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }
            read_buffer.reset(read_ptr);

            // Fill write buffer with test pattern
            fill_test_pattern(write_buffer.get(), read_write_test_phrase, config.transfer_size);
            clear_buffer(read_buffer.get(), config.transfer_size);

            // Register memory
            nixl_reg_dlist_t write_desc_list(DRAM_SEG);
            nixlBlobDesc write_desc;
            write_desc.addr = reinterpret_cast<uintptr_t>(write_buffer.get());
            write_desc.len = config.transfer_size;
            write_desc.devId = 1;
            write_desc.metaInfo = ""; // Empty metadata for DRAM

            nixl_reg_dlist_t read_desc_list(DRAM_SEG);
            nixlBlobDesc read_desc;
            read_desc.addr = reinterpret_cast<uintptr_t>(read_buffer.get());
            read_desc.len = config.transfer_size;
            read_desc.devId = 1;
            read_desc.metaInfo = ""; // Empty metadata for DRAM

            write_desc_list.addDesc(write_desc);
            read_desc_list.addDesc(read_desc);
            
            bool write_registered = false;
            bool read_registered = false;
            bool blk_registered = false;
            
            status = agent.registerMem(write_desc_list);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to register write buffer: " << status << std::endl;
                // Nothing to deregister since write_desc_list registration failed
                // and read_desc_list was never registered
                return false;
            }
            write_registered = true;

            status = agent.registerMem(read_desc_list);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to register read buffer: " << status << std::endl;
                agent.deregisterMem(write_desc_list);  // Only deregister write_desc_list
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }
            read_registered = true;

            std::cout << "✓ Memory registered for I/O test" << std::endl;

            // Register BLK memory for the block device
            nixl_reg_dlist_t blk_reg_list(BLK_SEG);
            nixlBlobDesc blk_reg_desc;
            blk_reg_desc.addr = 0; // Start offset on device
            blk_reg_desc.len = config.transfer_size; // Size to register
            blk_reg_desc.devId = 1;
            blk_reg_desc.metaInfo = config.device_path; // Device path
            blk_reg_list.addDesc(blk_reg_desc);
            
            status = agent.registerMem(blk_reg_list);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to register BLK memory: " << status << std::endl;
                if (write_registered) agent.deregisterMem(write_desc_list);
                if (read_registered) agent.deregisterMem(read_desc_list);
                return false;
            }
            blk_registered = true;

            // Perform write operation using createXferReq with backend hint
            nixl_xfer_dlist_t write_list(DRAM_SEG);
            write_list.addDesc(write_desc);

            nixl_xfer_dlist_t blk_list(BLK_SEG);
            nixlBasicDesc blk_desc;
            blk_desc.addr = 0; // Offset on device
            blk_desc.len = config.transfer_size;
            blk_desc.devId = 1;
            blk_list.addDesc(blk_desc);

            nixlXferReqH* req = nullptr;
            nixl_opt_args_t opt_args;
            opt_args.backends.push_back(backend);

            uint64_t start_time = nixlTime::getUs();
            // For libblkio local transfers, use empty string for remote agent
            status = agent.createXferReq(NIXL_WRITE, write_list, blk_list, "", req, &opt_args);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to create write request: " << status << std::endl;
                if (blk_registered) agent.deregisterMem(blk_reg_list);
                if (write_registered) agent.deregisterMem(write_desc_list);
                if (read_registered) agent.deregisterMem(read_desc_list);
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            status = agent.postXferReq(req);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to post write request: " << status << std::endl;
                agent.releaseXferReq(req);
                if (blk_registered) agent.deregisterMem(blk_reg_list);
                if (write_registered) agent.deregisterMem(write_desc_list);
                if (read_registered) agent.deregisterMem(read_desc_list);
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            status = agent.getXferStatus(req);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Write operation failed: " << status << std::endl;
                agent.releaseXferReq(req);
                if (blk_registered) agent.deregisterMem(blk_reg_list);
                if (write_registered) agent.deregisterMem(write_desc_list);
                if (read_registered) agent.deregisterMem(read_desc_list);
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            agent.releaseXferReq(req);

            uint64_t write_time = nixlTime::getUs() - start_time;
            double write_throughput = (config.transfer_size / (1024.0 * 1024.0)) / (write_time / 1000000.0);

            std::cout << "✓ Write operation completed" << std::endl;
            std::cout << "  - Size: " << config.transfer_size / 1024 << " KB" << std::endl;
            std::cout << "  - Time: " << format_duration(write_time) << std::endl;
            std::cout << "  - Throughput: " << std::fixed << std::setprecision(2) 
                      << write_throughput << " MB/s" << std::endl;

            // Perform read operation
            nixl_xfer_dlist_t read_list(DRAM_SEG);
            read_list.addDesc(read_desc);

            start_time = nixlTime::getUs();
            status = agent.createXferReq(NIXL_READ, read_list, blk_list, "", req, &opt_args);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to create read request: " << status << std::endl;
                if (blk_registered) agent.deregisterMem(blk_reg_list);
                if (write_registered) agent.deregisterMem(write_desc_list);
                if (read_registered) agent.deregisterMem(read_desc_list);
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            status = agent.postXferReq(req);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Failed to post read request: " << status << std::endl;
                agent.releaseXferReq(req);
                if (blk_registered) agent.deregisterMem(blk_reg_list);
                if (write_registered) agent.deregisterMem(write_desc_list);
                if (read_registered) agent.deregisterMem(read_desc_list);
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            status = agent.getXferStatus(req);
            if (status != NIXL_SUCCESS) {
                std::cerr << "Read operation failed: " << status << std::endl;
                agent.releaseXferReq(req);
                if (blk_registered) agent.deregisterMem(blk_reg_list);
                if (write_registered) agent.deregisterMem(write_desc_list);
                if (read_registered) agent.deregisterMem(read_desc_list);
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            agent.releaseXferReq(req);

            uint64_t read_time = nixlTime::getUs() - start_time;
            double read_throughput = (config.transfer_size / (1024.0 * 1024.0)) / (read_time / 1000000.0);

            std::cout << "✓ Read operation completed" << std::endl;
            std::cout << "  - Size: " << config.transfer_size / 1024 << " KB" << std::endl;
            std::cout << "  - Time: " << format_duration(read_time) << std::endl;
            std::cout << "  - Throughput: " << std::fixed << std::setprecision(2) 
                      << read_throughput << " MB/s" << std::endl;

            // Verify data integrity
            if (memcmp(write_buffer.get(), read_buffer.get(), config.transfer_size) != 0) {
                std::cerr << "✗ Data integrity check failed" << std::endl;
                if (blk_registered) agent.deregisterMem(blk_reg_list);
                if (write_registered) agent.deregisterMem(write_desc_list);
                if (read_registered) agent.deregisterMem(read_desc_list);
                // Note: nixlAgent destructor handles backend cleanup
                return false;
            }

            std::cout << "✓ Data integrity check passed" << std::endl;

            // Cleanup
            agent.deregisterMem(write_desc_list);
            agent.deregisterMem(read_desc_list);
            agent.deregisterMem(blk_reg_list);

        } catch (const std::exception& e) {
            std::cerr << "Exception in basic I/O test: " << e.what() << std::endl;
            return false;
        }

        print_phase_footer();
        return true;
    }

    // Print usage information
    void print_usage(const char* program_name) {
        std::cout << "Usage: " << program_name << " [options]" << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -h, --help              Show this help message" << std::endl;
        std::cout << "  -d, --device <path>     Device path (overridden by NIXL_LIBBLKIO_TEST_DEVICE)" << std::endl;
        std::cout << "  -t, --transfers <num>   Number of transfers (default: " << default_num_transfers << ")" << std::endl;
        std::cout << "  -s, --size <bytes>      Transfer size in bytes (default: " << default_transfer_size << ")" << std::endl;
        std::cout << "  --direct-io             Enable direct I/O" << std::endl;
        std::cout << "  --io-polling            Enable I/O polling" << std::endl;
        std::cout << "  --test-dir <path>       Test files directory (default: " << default_test_files_dir_path << ")" << std::endl;
        std::cout << "  --no-setup              Skip test environment setup" << std::endl;
        std::cout << "  --no-cleanup            Skip test environment cleanup" << std::endl;
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    TestConfig config;
    bool skip_setup = false;
    bool skip_cleanup = false;

    // Parse command line arguments
    static struct option long_options[] = {
        {"help",        no_argument,       0, 'h'},
        {"device",      required_argument, 0, 'd'},
        {"transfers",   required_argument, 0, 't'},
        {"size",        required_argument, 0, 's'},
        {"direct-io",   no_argument,       0, 1001},
        {"io-polling",  no_argument,       0, 1002},
        {"test-dir",    required_argument, 0, 1003},
        {"no-setup",    no_argument,       0, 1004},
        {"no-cleanup",  no_argument,       0, 1005},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "hd:t:s:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'd':
                config.device_path = optarg;
                break;
            case 't':
                try {
                    config.num_transfers = std::stoi(optarg);
                } catch (const std::exception& e) {
                    std::cerr << "Invalid value for --transfers: " << optarg << std::endl;
                    return 1;
                }
                break;
            case 's':
                try {
                    config.transfer_size = std::stoull(optarg);
                } catch (const std::exception& e) {
                    std::cerr << "Invalid value for --size: " << optarg << std::endl;
                    return 1;
                }
                break;
            case 1001:
                config.enable_direct_io = true;
                break;
            case 1002:
                config.enable_io_polling = true;
                break;
            case 1003:
                config.test_files_dir = optarg;
                break;
            case 1004:
                skip_setup = true;
                break;
            case 1005:
                skip_cleanup = true;
                break;
            case '?':
                print_usage(argv[0]);
                return 1;
            default:
                break;
        }
    }

    // Check for NIXL_LIBBLKIO_TEST_DEVICE environment variable
    const char *env_dev = std::getenv("NIXL_LIBBLKIO_TEST_DEVICE");
    if (env_dev && *env_dev != '\0') {
        config.device_path = env_dev;
    }

    // Require explicit device specification
    if (config.device_path.empty()) {
        std::cerr << "Error: Set NIXL_LIBBLKIO_TEST_DEVICE to a disposable block device" << std::endl;
        return 1;
    }

    // Print test configuration
    std::cout << line_str << std::endl;
    std::cout << center_str("NIXL libblkio Plugin Test") << std::endl;
    std::cout << line_str << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Device: " << config.device_path << std::endl;
    std::cout << "  Transfers: " << config.num_transfers << std::endl;
    std::cout << "  Transfer Size: " << config.transfer_size / 1024 << " KB" << std::endl;
    std::cout << "  Direct I/O: " << (config.enable_direct_io ? "enabled" : "disabled") << std::endl;
    std::cout << "  I/O Polling: " << (config.enable_io_polling ? "enabled" : "disabled") << std::endl;
    std::cout << "  Test Files Dir: " << config.test_files_dir << std::endl;
    std::cout << std::endl;

    bool all_tests_passed = true;

    // Setup test environment if not skipped
    if (!skip_setup) {
        std::cout << "Setting up test environment..." << std::endl;
        if (!setup_test_environment(config)) {
            std::cerr << "Failed to setup test environment" << std::endl;
            std::cerr << "Note: This test requires root privileges for loop device setup" << std::endl;
            std::cerr << "You can skip setup with --no-setup if you have already set up a device" << std::endl;
            return 1;
        }
        std::cout << "✓ Test environment setup complete" << std::endl;
    }

    // Run tests
    if (!test_backend_creation(config)) {
        all_tests_passed = false;
    }

    if (!test_invalid_api_type(config)) {
        all_tests_passed = false;
    }

    if (!test_memory_registration(config)) {
        all_tests_passed = false;
    }

    if (!test_basic_io(config)) {
        all_tests_passed = false;
    }

    // Cleanup test environment if not skipped
    if (!skip_cleanup) {
        std::cout << "Cleaning up test environment..." << std::endl;
        cleanup_test_environment(config);
        std::cout << "✓ Test environment cleanup complete" << std::endl;
    }

    // Print final results
    std::cout << "\n" << line_str << std::endl;
    std::cout << center_str("Test Results") << std::endl;
    std::cout << line_str << std::endl;
    
    if (all_tests_passed) {
        std::cout << "✓ All tests PASSED" << std::endl;
        std::cout << line_str << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some tests FAILED" << std::endl;
        std::cout << line_str << std::endl;
        return 1;
    }
}
