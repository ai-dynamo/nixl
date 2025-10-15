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
#include <iostream> // std:cerr
#include <iomanip> // std::setprecision
#include <unistd.h>
#include <stdlib.h>
#include <absl/strings/str_format.h>
#include "nixl.h"
#include "common/nixl_time.h"
#include <getopt.h>
#include "test_utils.h"

#define UUID_LOCAL_FILE_0 11 // Just some numbers
#define UUID_K_DEV_ZERO_1 14
#define UUID_NVME_DISK__0 27

#define DEF_TEST_PHRASE "|NIXL bdev 32[b] GUSLI pattern |"
#define DEF_TEST_PHRASE_LEN (sizeof(DEF_TEST_PHRASE) - 1) // -1 to exclude null terminator

static std::ostream &err_log = std::cerr;
static std::ostream &out_log = std::cout;

class test_pattern_t {
    static constexpr const size_t test_phrase_len = DEF_TEST_PHRASE_LEN;
    char test_phrase[test_phrase_len + 1] __attribute__((aligned(sizeof(long))));
    uint64_t unique_stage = '!';

    void
    inject_unique(size_t i) {
        *((size_t *)&test_phrase[24]) = ((unique_stage & 0xFF) << 56) |
            __builtin_bswap64(i); // Unique last 64 bits for each 32[b] string
    }

    bool
    error_print(const char *p, size_t i, size_t size, const char *expected) const {
        err_log << "DRAM[" << i << "]=" << (void *)&p[i] << ", validation error, size=" << size
                << ": test=" << expected << ",\t\t buf=" << (const char *)&p[i] << "\n";
        return false;
    }

public:
    void
    change_unique(void) {
        unique_stage++;
    }

    void
    fill(void *buffer, size_t size) {
        strcpy(test_phrase, DEF_TEST_PHRASE);
        char *p = (char *)buffer;
        for (size_t i = 0; i < size; i += DEF_TEST_PHRASE_LEN) {
            // inject_unique(i);
            memcpy(&p[i], test_phrase, test_phrase_len);
        }
    }

    void
    clear(void *p, size_t size) const {
        memset(p, 'c', size);
    }

    void
    zero(void *p, size_t size) const {
        memset(p, 0, size);
    }

    void
    print(const void *p, size_t size) const {
        out_log << "BUF: ";
        out_log.write((char *)p, 16);
        out_log << "\n";
    }

    bool
    verify(const void *buffer, size_t size) {
        char *p = (char *)buffer;
        for (size_t i = 0; i < size; i += DEF_TEST_PHRASE_LEN) {
            // inject_unique(i);
            if (0 != memcmp(&p[i], test_phrase, DEF_TEST_PHRASE_LEN))
                return error_print(p, i, size, test_phrase);
        }
        return true;
    }

    bool
    verify_zero(const void *buffer, size_t size) const {
        char *p = (char *)buffer;
        size_t zero = 0UL;
        for (size_t i = 0; i < size; i += 8) {
            if (0 != memcmp(&p[i], &zero, 8)) return error_print(p, i, size, (const char *)&zero);
        }
        return true;
    }
};

class gtest { // Gusli tester class
private:
    static constexpr const size_t gb_size = (1 << 30);
    static constexpr const int line_width = 60; // Unites typical line width
    static constexpr const char *agent_name = "GUSLITester";
    static constexpr const char *line_str =
        "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    static constexpr const bool verbose = false;
    int num_transfers;
    size_t transfer_size, n_total_mapped_bytes;
    const size_t bdev_byte_offset = (1UL << 20); // Write at offset 1[MB] in block device
    long page_size;
    long sg_buf_size; // Array of pages for describing the list of io descriptors in registered
                      // memory
    void *ptr; // Registered mem RAM buffer for ios
    nixlXferReqH *treq = nullptr; // io request
    test_pattern_t test_pattern;

    size_t
    get_total_mem_useage(void) const {
        return n_total_mapped_bytes + sg_buf_size;
    }

    static std::string
    center_str(const std::string &str) {
        return std::string((line_width - str.length()) / 2, ' ') + str;
    }

    static std::string
    format_time(nixlTime::us_t us) { // Helper function to format duration
        const nixlTime::ms_t ms = us / 1000.0;
        return (ms < 1000) ? absl::StrFormat("%.0f[ms]", ms) :
                             absl::StrFormat("%.3f[sec]", ((double)ms / 1000.0));
    }

    static void
    progress_bar(float p /* [0..1]*/) {
        if (verbose) return; // Progress bar may ibsucre user prints
        static constexpr const int progress_bar_width = (line_width - 2); // -2 for the brackets
        out_log << "[";
        int i;
        const int n_chars = (progress_bar_width * p);
        for (i = 0; i < n_chars; ++i)
            out_log << "=";
        out_log << ">";
        for (; i < progress_bar_width; ++i)
            out_log << " ";
        out_log << absl::StrFormat("] %.1f%% ", (p * 100.0));
        if (p >= 1.0) { // Add completion indicator
            out_log << "DONE!\n";
        } else {
            out_log << "\r";
            out_log.flush();
        }
    }

    static std::string
    phase_title(const std::string &title) {
        static int phase_num = 1;
        return absl::StrFormat("PHASE %d: %s", phase_num++, title.c_str());
    }

    static void
    print_segment_title(const std::string &title) {
        out_log << line_str << center_str(title) << line_str;
    }

public:
    gtest(int _num_transfers, size_t _transfer_size)
        : num_transfers(_num_transfers),
          transfer_size(_transfer_size),
          page_size(-1),
          ptr(nullptr) {
        page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) {
            err_log << "Error: Invalid page size returned by sysconf\n";
            return;
        }
        if (num_transfers < 8) num_transfers = 8; // At least 8
        if (num_transfers % 4) // Make num_transfers aligned to 4
            num_transfers = ((num_transfers + 3) / 4) * 4;
        if ((transfer_size % page_size) != 0) // align transfer size to page size
            transfer_size = ((transfer_size + page_size - 1) / page_size) * page_size;
        sg_buf_size = num_transfers *
            32; // SG (scatter gather element is 24[b] (ptr+len+offset). Round to 32)
        sg_buf_size = ((sg_buf_size + page_size - 1) / page_size) * page_size; // Round to page size
        out_log << "sg_buf_size=" << sg_buf_size << std::endl;
        n_total_mapped_bytes = (num_transfers * transfer_size);
        out_log << "n_total_mapped_bytes=" << n_total_mapped_bytes << std::endl;
        out_log << "page_size=" << page_size << std::endl;
        out_log << "transfer_size=" << transfer_size << std::endl;
        out_log << "num_transfers=" << num_transfers << std::endl;
    }

    ~gtest() {
        if (ptr) free(ptr);
    }

    nixl_b_params_t
    gen_gusli_plugin_params(const nixlAgent &agent) const { // Set up backend parameters
        // Get default params / supported mem
        nixl_b_params_t params;
        nixl_mem_list_t mems1;
        nixl_status_t ret1 = agent.getPluginParams("GUSLI", mems1, params);
        nixl_exit_on_failure(ret1, "Failed to get GUSLI plugin params", std::string(agent_name));
        if (verbose) {
            out_log << absl::StrFormat("Default Plugin params:\n");
            for (const auto &q : params) {
                out_log << "key=" << q.first << ", val=" << q.second << "\n";
            }
            out_log << absl::StrFormat("Plugin supported mem:\n");
            for (const auto &q : mems1) {
                out_log << nixlEnumStrings::memTypeStr(q) << ",";
            }
            out_log << "\n";
        }

// Add gusli specific params
#ifndef __stringify
#define __stringify_1(x...) #x
#define __stringify(x...) __stringify_1(x)
#endif

        params["client_name"] = agent_name;
#if 0
            // You can include gusli api and build config file using its methods
            gusli::client_config_file conf(1 /*Version*/);
            using gsc = gusli::bdev_config_params;
            conf.bdev_add(gsc(__stringify (UUID_LOCAL_FILE_0), gsc::bdev_type::DEV_FS_FILE,    "./store0.bin", "sec=0x03", 0, gsc::connect_how::SHARED_RW));
            conf.bdev_add(gsc(__stringify (UUID_K_DEV_ZERO_1), gsc::bdev_type::DEV_BLK_KERNEL, "/dev/zero",    "sec=0x71", 0, gsc::connect_how::EXCLUSIVE_RW));
            conf.bdev_add(gsc(__stringify (UUID_NVME_DISK__0), gsc::bdev_type::DEV_BLK_KERNEL, "/dev/nvme0n1", "sec=0x07", 1, gsc::connect_how::EXCLUSIVE_RW));
            params["config_file"] = conf.get();
#else
        // Unsafe method: Just generate the config string
        params["config_file"] = "# Config file\nversion=1\n" __stringify(
            UUID_LOCAL_FILE_0) " F W N ./store0.bin sec=0x3\n" __stringify(UUID_K_DEV_ZERO_1) " K "
                                                                                              "X N "
                                                                                              "/dev"
                                                                                              "/zer"
                                                                                              "o   "
                                                                                              " sec"
                                                                                              "=0x7"
                                                                                              "1\n";
#endif
        params["max_num_simultaneous_requests"] = std::to_string(num_transfers);
        return params;
    }

    void
    __alloc_sgl(uint64_t devId,
                nixl_xfer_dlist_t &bdev_io_src,
                nixl_xfer_dlist_t
                    &bdev_io_dst) { // First entry is dummy, with enough space for scatter gather
        nixlBlobDesc d;
        d.devId = devId;
        d.len = sg_buf_size;
        d.addr = (uintptr_t)((u_int64_t)ptr + n_total_mapped_bytes);
        out_log << "Adding SGL to bdev_io_src, devId=" << devId << ", len=" << sg_buf_size
                << ", addr=" << (void *)d.addr << std::endl;
        bdev_io_src.addDesc(d);
        d.addr = bdev_byte_offset; // Dummy
        bdev_io_dst.addDesc(d);
        out_log << "Adding SGL to bdev_io_dst, devId=" << devId << ", len=" << sg_buf_size
                << ", addr=" << (void *)d.addr << std::endl;
    }

    void
    single_bdev_request_build(nixl_xfer_dlist_t &bdev_io_src,
                              nixl_xfer_dlist_t &bdev_io_dst,
                              bool with_sgl = false) {
        bdev_io_src.clear();
        bdev_io_dst.clear();
        nixlBlobDesc d;
        d.devId = UUID_LOCAL_FILE_0;
        if (with_sgl) __alloc_sgl(d.devId, bdev_io_src, bdev_io_dst);
        out_log << "Building single bdev request, with_sgl=" << (with_sgl ? 'Y' : 'N') << std::endl;
        for (int i = 0; i < num_transfers; ++i) { // Create all the transfer
            const size_t io_offset = (i * transfer_size);
            d.len = transfer_size;
            d.addr = (uintptr_t)((size_t)ptr + io_offset); // Offset in RAM buffer
            out_log << "Adding SGL to bdev_io_src, devId=" << d.devId << ", len=" << d.len
                    << ", addr=" << (void *)d.addr << std::endl;
            bdev_io_src.addDesc(d);
            d.addr = bdev_byte_offset + io_offset;
            bdev_io_dst.addDesc(d);
            out_log << "Adding SGL to bdev_io_dst, devId=" << d.devId << ", len=" << d.len
                    << ", addr=" << (void *)d.addr << std::endl;
            progress_bar(float(i + 1) / num_transfers);
        }
    }

#define QUIT_ON_ERR(msg, status)                                              \
    do {                                                                      \
        if (status < NIXL_SUCCESS) {                                          \
            err_log << "Error: " << msg << nixlEnumStrings::statusStr(status) \
                    << " Line: " << __LINE__ << std::endl;                    \
            if (treq) agent.releaseXferReq(treq);                             \
            return -__LINE__;                                                 \
        }                                                                     \
    } while (0)

    int
    register_bufs_on_multi_bdev(
        nixlAgent &agent,
        bool do_reg) { // Register the large IO buffer + additional sg on 2 bdevs
        const char *action_str = (do_reg ? "R" : "Unr");
        nixl_reg_dlist_t dram_reg(DRAM_SEG), bdev_reg(BLK_SEG);
        int bdevs[2] = {UUID_LOCAL_FILE_0, UUID_K_DEV_ZERO_1};
        nixlBlobDesc d;
        nixl_status_t status;
        d.devId = 0;
        d.len = get_total_mem_useage();
        d.addr = (uintptr_t)ptr;
        dram_reg.addDesc(d);
        d.addr = bdev_byte_offset;
        bdev_reg.addDesc(d);
        for (int i = 0; i < 2; i++) {
            dram_reg[0].devId = bdev_reg[0].devId = bdevs[i];
            dram_reg[0].metaInfo = bdev_reg[0].metaInfo = absl::StrFormat("DummyMd%d", i);
            status = (do_reg ? agent.registerMem(dram_reg) : agent.deregisterMem(dram_reg));
            QUIT_ON_ERR(absl::StrFormat("Failed bdev=%u %eg=%s, rv=",
                                        bdevs[i],
                                        action_str,
                                        nixlEnumStrings::memTypeStr(dram_reg.getType())),
                        status);
            progress_bar(i * 0.5f + 0.25f);
            status = (do_reg ? agent.registerMem(bdev_reg) : agent.deregisterMem(bdev_reg));
            QUIT_ON_ERR(absl::StrFormat("Failed bdev=%u %eg=%s, rv=",
                                        bdevs[i],
                                        action_str,
                                        nixlEnumStrings::memTypeStr(bdev_reg.getType())),
                        status);
            progress_bar(i * 0.5f + 0.50f);
        }
        return 0;
    }

    void
    multi_bdev_single_request_build(nixl_xfer_dlist_t &bdev_io_src,
                                    nixl_xfer_dlist_t &bdev_io_dst,
                                    bool with_sgl,
                                    bool force_first_bdev = false) {
        bdev_io_src.clear();
        bdev_io_dst.clear();
        int bdevs[2] = {UUID_LOCAL_FILE_0, UUID_K_DEV_ZERO_1}, n_ranges = 7;
        if (with_sgl) __alloc_sgl(UUID_LOCAL_FILE_0, bdev_io_src, bdev_io_dst);
        for (int i = 0; i < n_ranges; ++i) { // Create transfers
            const size_t io_offset = (i * transfer_size);
            const int cur_bdev = force_first_bdev ? 0 : (((i > 4) || (i == 1)) ? 1 : 0);
            nixlBlobDesc d;
            d.devId = bdevs[cur_bdev]; // Interleave requests to 2 block devices
            d.len = transfer_size;
            d.addr = (uintptr_t)((size_t)ptr + io_offset); // Offset in RAM buffer
            const bool is_zero = (d.devId == UUID_K_DEV_ZERO_1);
            test_pattern.fill((void *)d.addr, d.len);
            bdev_io_src.addDesc(d);
            if (verbose)
                out_log << "MULTI-bdev: Range=" << i << ", curbdev=" << d.devId
                        << ", ptr=" << (void *)d.addr << ", len=" << d.len
                        << ", lba=" << (bdev_byte_offset + io_offset) << ", is_zero=" << is_zero
                        << "\n";
            d.addr = bdev_byte_offset + io_offset;
            bdev_io_dst.addDesc(d);
            progress_bar(float(i + 1) / n_ranges);
        }
    }

    bool
    multi_bdev_single_request_verify(const nixl_xfer_dlist_t &bdev_io_src, const bool has_sgl) {
        const int start_range = (has_sgl ? 1 : 0);
        const int n_ranges = bdev_io_src.descCount() - start_range;
        if (verbose) out_log << "MULTI-bdev: verify n_ranges=" << n_ranges << "\n";
        for (int i = start_range; i < n_ranges; ++i) { // Create transfers
            const nixlBasicDesc &d = bdev_io_src[i];
            const bool is_zero = (d.devId == UUID_K_DEV_ZERO_1);
            if (verbose)
                out_log << "Range=" << i << ", curbdev=" << d.devId << ", ptr=" << (void *)d.addr
                        << ", len=" << d.len << ", is_zero=" << is_zero << "\n";
            if (is_zero) {
                if (!test_pattern.verify_zero((void *)d.addr, d.len)) return false;
            } else {
                if (!test_pattern.verify((void *)d.addr, d.len)) return false;
            }
            progress_bar(float(i + 1) / n_ranges);
        }
        return true;
    }

    int
    run_write_read_verify(void) {
        nixlAgent agent(agent_name, nixlAgentConfig(true));
        print_segment_title("NIXL STORAGE TEST STARTING (GUSLI PLUGIN)");
        nixl_b_params_t params = gen_gusli_plugin_params(agent);

        // Print test configuration information
        out_log << absl::StrFormat("Configuration:\n");
        out_log << absl::StrFormat("- Number of transfers=%d\n", num_transfers);
        out_log << absl::StrFormat(
            "- Transfer=%zu[KB], sg=%zu[KB]\n", (transfer_size >> 10), (sg_buf_size >> 10));
        out_log << absl::StrFormat("- Total data: %.2f[GB], 0x%lx[B]\n",
                                   float(n_total_mapped_bytes) / gb_size,
                                   get_total_mem_useage());
        out_log << absl::StrFormat("- Backend: GUSLI, Direct IO enabled\n") << line_str;

        // Create GUSLI backend first - before allocating any resources
        nixlBackendH *n_backend = nullptr; // Backend gusli plugin
        nixl_status_t status;
        status = agent.createBackend("GUSLI", params, n_backend);
        QUIT_ON_ERR("Backend Creation Failed: ", status);

        if (1) {
            print_segment_title(phase_title("Failed Second plugin initialization"));
            nixlBackendH *_2nd_plugin = nullptr;
            nixlAgent agent2("2nd_agent", nixlAgentConfig(true));
            bool init_exception_caught = false;
            try {
                status = agent2.createBackend("GUSLI", params, n_backend);
            }
            catch (const std::runtime_error &e) {
                init_exception_caught = true;
            }

            nixl_exit_on_failure((_2nd_plugin == nullptr), "2nd plugin instance could be created");
            nixl_exit_on_failure((init_exception_caught = true),
                                 "2nd plugin creation exception not caught!");
        }

        print_segment_title(
            phase_title(absl::StrFormat("Allocating buffers, bdev %u", UUID_LOCAL_FILE_0)));
        if (posix_memalign(&ptr, page_size, get_total_mem_useage()) != 0)
            QUIT_ON_ERR("DRAM allocation failed", NIXL_ERR_NOT_SUPPORTED);
        nixl_xfer_dlist_t bdev_io_src(DRAM_SEG), bdev_io_dst(BLK_SEG);
        single_bdev_request_build(bdev_io_src, bdev_io_dst, true);

        print_segment_title(phase_title("Registering memory with NIXL"));
        nixl_reg_dlist_t dram_reg(DRAM_SEG), bdev_reg(BLK_SEG);
        { // Register the large IO buffer + additional sg
            nixlBlobDesc d;
            d.devId = UUID_LOCAL_FILE_0;
            d.len = n_total_mapped_bytes; // Just for debug, register in 2 descriptos, can register
                                          // as 1 buffer as well
            d.addr = (uintptr_t)ptr;
            dram_reg.addDesc(d);
            d.len = sg_buf_size;
            d.addr = (uintptr_t)((size_t)ptr + n_total_mapped_bytes);
            dram_reg.addDesc(d);
            // Just for debug, register in 4 quarters, can register as 1 buffer as well
            d.len =
                n_total_mapped_bytes / 4; // Security reason: Enforces all IO's within registered
                                          // memory. Not needed internally by the plugin
            for (int i = 0; i < 4; i++) {
                d.addr = bdev_byte_offset + i * d.len;
                bdev_reg.addDesc(d);
            }
            status = agent.registerMem(dram_reg);
            QUIT_ON_ERR(absl::StrFormat("Failed reg=%s, rv=",
                                        nixlEnumStrings::memTypeStr(dram_reg.getType())),
                        status);
            progress_bar(0.5f);
            status = agent.registerMem(bdev_reg);
            QUIT_ON_ERR(absl::StrFormat("Failed reg=%s, rv=",
                                        nixlEnumStrings::memTypeStr(bdev_reg.getType())),
                        status);
            progress_bar(1.0f);
        }

        enum nixl_xfer_op_t io_phases[] = {NIXL_WRITE, NIXL_READ}; // First write - then read

        nixl_opt_args_t extra_params;
        extra_params.customParam = "-sgl";
        print_segment_title(
            phase_title(absl::StrFormat("1[xfer] Write-Read-Verify %u[KB]", transfer_size >> 10)));
        if (1) {
            nixl_xfer_dlist_t bdev_io_1src(DRAM_SEG), bdev_io_1dst(BLK_SEG);
            bdev_io_1src.addDesc(bdev_io_src[4]); // Just an arbitrary 4'th io
            bdev_io_1dst.addDesc(bdev_io_dst[4]);
            test_pattern.fill((char *)bdev_io_1src[0].addr, transfer_size);
            const nixlTime::us_t time_start = nixlTime::getUs();
            for (int i = 0; i < 2; i++, treq = nullptr) {
                const std::string io_t_str = nixlEnumStrings::xferOpStr(io_phases[i]);
                status =
                    agent.createXferReq(io_phases[i], bdev_io_1src, bdev_io_1dst, agent_name, treq);
                QUIT_ON_ERR("Failed to create req, rv=", status);
                status = agent.postXferReq(treq);
                QUIT_ON_ERR("Failed to post req, rv=", status);
                do { // Busy loop wait for transfer to complete
                    status = agent.getXferStatus(treq);
                    QUIT_ON_ERR("Failed during transfer req, rv=", status);
                } while (status == NIXL_IN_PROG);
                if (io_phases[i] == NIXL_WRITE) // Clear buffers before read
                    test_pattern.clear(ptr, transfer_size);
                agent.releaseXferReq(treq);
            }
            if (!test_pattern.verify((char *)bdev_io_1src[0].addr, transfer_size)) return -__LINE__;
            const nixlTime::us_t time_end = nixlTime::getUs();
            const nixlTime::us_t micro_secs = (time_end - time_start);
            out_log << "- Time: " << format_time(micro_secs) << std::endl;
        }

        print_segment_title(phase_title(
            absl::StrFormat("Generating unique data %u[MB]", n_total_mapped_bytes >> 20)));
        test_pattern.change_unique();

        out_log << "Filling ptr with unique data, ptr=" << (void *)ptr
                << ", n_total_mapped_bytes=" << n_total_mapped_bytes << std::endl;
        test_pattern.fill(ptr, n_total_mapped_bytes);

        out_log << "Generating unique data, with_sgl=Y, filled ptr with unique data\n" << std::endl;

        nixlTime::us_t total_time(0);
        double total_data_gb(0);
        if (1) {
            for (int with_sgl = 1; with_sgl >= 0;
                 with_sgl--) { // Test with sgl dummy range then remove it and test without it
                single_bdev_request_build(bdev_io_src, bdev_io_dst, with_sgl);

                out_log << "Building single bdev request, with_sgl=" << (with_sgl ? 'Y' : 'N')
                        << std::endl;

                const int n_ranges = (int)bdev_io_src.descCount() - with_sgl;

                out_log << "Creating 2 transfers, with_sgl=" << (with_sgl ? 'Y' : 'N') << std::endl;

                for (int i = 0; i < 2; i++, treq = nullptr) {
                    const std::string io_t_str = nixlEnumStrings::xferOpStr(io_phases[i]);
                    print_segment_title(phase_title(absl::StrFormat("%s Test, nIOs=%u, with_sgl=%c",
                                                                    io_t_str.c_str(),
                                                                    n_ranges,
                                                                    (with_sgl ? 'Y' : 'N'))));
                    status = agent.createXferReq(io_phases[i],
                                                 bdev_io_src,
                                                 bdev_io_dst,
                                                 agent_name,
                                                 treq,
                                                 (with_sgl ? &extra_params : nullptr));
                    QUIT_ON_ERR("Failed to create req, rv=", status);
                    const nixlTime::us_t time_start = nixlTime::getUs();
                    status = agent.postXferReq(treq);
                    QUIT_ON_ERR("Failed to post req, rv=", status);
                    do { // Busy loop wait for transfer to complete
                        status = agent.getXferStatus(treq);
                        QUIT_ON_ERR("Failed during transfer req, rv=", status);
                    } while (status == NIXL_IN_PROG);
                    const nixlTime::us_t time_end = nixlTime::getUs();
                    const nixlTime::us_t micro_secs = (time_end - time_start);
                    const double data_gb = (double)n_total_mapped_bytes / (double)gb_size;
                    out_log << "- Time: " << format_time(micro_secs) << std::endl;
                    out_log << "- Data: " << std::fixed << std::setprecision(2) << data_gb
                            << "[GB]\n";
                    out_log << "- Speed: " << ((data_gb * 1000000.0) / micro_secs) << "[GB/s]\n";
                    total_time += micro_secs;
                    total_data_gb += data_gb;
                    agent.releaseXferReq(treq);
                    if (io_phases[i] == NIXL_WRITE) // Clear buffers before read
                        test_pattern.clear(ptr, n_total_mapped_bytes);
                }
                print_segment_title(phase_title("Validating read data"));
                if (!test_pattern.verify(ptr, n_total_mapped_bytes)) return -__LINE__;
            }
        }

        {
            print_segment_title(phase_title("Un-Registering memory with NIXL"));
            status = agent.deregisterMem(dram_reg);
            QUIT_ON_ERR(absl::StrFormat("Failed de-reg=%s, rv=",
                                        nixlEnumStrings::memTypeStr(dram_reg.getType())),
                        status);
            status = agent.deregisterMem(bdev_reg);
            QUIT_ON_ERR(absl::StrFormat("Failed de-reg=%s, rv=",
                                        nixlEnumStrings::memTypeStr(bdev_reg.getType())),
                        status);
        }
        print_segment_title("TEST write-read summary");
        out_log << "Total time: " << format_time(total_time) << std::endl;
        out_log << "Total data: " << std::fixed << std::setprecision(2) << total_data_gb << "[GB]"
                << line_str;

        if (1) { // Multi-bdev IO tests
            test_pattern.change_unique();
            print_segment_title(phase_title("register-mem on multi-bdevs"));
            if (register_bufs_on_multi_bdev(agent, true) < 0) return -__LINE__;
            test_pattern.fill(ptr, n_total_mapped_bytes);
            int with_sgl = 1;
            print_segment_title(phase_title(absl::StrFormat(
                "Write dummy info to bdev[0] with_sgl=%c", (with_sgl ? 'Y' : 'N'))));
            multi_bdev_single_request_build(bdev_io_src, bdev_io_dst, with_sgl, true);
            status = agent.createXferReq(NIXL_WRITE,
                                         bdev_io_src,
                                         bdev_io_dst,
                                         agent_name,
                                         treq,
                                         (with_sgl ? &extra_params : nullptr));
            QUIT_ON_ERR("Failed to create req, rv=", status);
            status = agent.postXferReq(treq);
            QUIT_ON_ERR("Failed to post req, rv=", status);
            do { // Busy loop wait for transfer to complete
                status = agent.getXferStatus(treq);
                QUIT_ON_ERR("Failed during transfer req, rv=", status);
            } while (status == NIXL_IN_PROG);
            agent.releaseXferReq(treq);
            for (with_sgl = 1; with_sgl >= 0;
                 with_sgl--) { // Test with sgl dummy range then remove it and test without it
                print_segment_title(phase_title(absl::StrFormat(
                    "TEST 1-transfer-multi-bdevs, with_sgl=%c", (with_sgl ? 'Y' : 'N'))));
                test_pattern.change_unique();
                multi_bdev_single_request_build(bdev_io_src, bdev_io_dst, with_sgl);
                for (int i = 0; i < 2; i++, treq = nullptr) {
                    const std::string io_t_str = nixlEnumStrings::xferOpStr(io_phases[i]);
                    out_log << phase_title(absl::StrFormat("%s nRanges=%u, with_sgl=%c\n",
                                                           io_t_str.c_str(),
                                                           (unsigned)bdev_io_src.descCount(),
                                                           (with_sgl ? 'Y' : 'N')));
                    status = agent.createXferReq(io_phases[i],
                                                 bdev_io_src,
                                                 bdev_io_dst,
                                                 agent_name,
                                                 treq,
                                                 (with_sgl ? &extra_params : nullptr));
                    QUIT_ON_ERR("Failed to create req, rv=", status);
                    status = agent.postXferReq(treq);
                    QUIT_ON_ERR("Failed to post req, rv=", status);
                    do { // Busy loop wait for transfer to complete
                        status = agent.getXferStatus(treq);
                        QUIT_ON_ERR("Failed during transfer req, rv=", status);
                    } while (status == NIXL_IN_PROG);
                    agent.releaseXferReq(treq);
                    if (io_phases[i] == NIXL_WRITE) // Clear buffers before read
                        test_pattern.clear(ptr, n_total_mapped_bytes);
                }
                out_log << phase_title(absl::StrFormat("Verify\n"));
                if (!multi_bdev_single_request_verify(bdev_io_src, with_sgl)) return -__LINE__;
            }
            print_segment_title(phase_title("unegister-mem on multi-bdevs"));
            if (register_bufs_on_multi_bdev(agent, false) < 0) return -__LINE__;
            print_segment_title(phase_title("Done! Success"));
        }
        return 0;
    }
};

int
main(int argc, char *argv[]) {
    static constexpr const int default_num_transfers = 50;
    static constexpr const size_t default_transfer_size = (1UL << 4); // 2[MB]
    int opt, num_transfers = default_num_transfers;
    size_t transfer_size = default_transfer_size;
    while ((opt = getopt(argc, argv, "n:s:h")) != -1) {
        switch (opt) {
        case 'n':
            num_transfers = std::stoi(optarg);
            break;
        case 's':
            transfer_size = std::stoull(optarg);
            break;
        case 'h':
        default:
            out_log << absl::StrFormat("Usage: %s [-n num_transfers] [-s transfer_size] [-h]\n",
                                       argv[0]);
            out_log << absl::StrFormat(
                "  -n num_transfers      Number of transfers (default: %d)\n",
                default_num_transfers);
            out_log << absl::StrFormat(
                "  -s transfer_size      Size of each transfer[Bytes] (default: %zu)\n",
                default_transfer_size);
            out_log << absl::StrFormat("  -h                    Show this help message\n");
            return (opt == 'h') ? 0 : 1;
        }
    }
    gtest base_test(num_transfers, transfer_size);
    const int rv = base_test.run_write_read_verify();
    // out_log << absl::StrFormat ("agent destroyed. test_rv=%d\n", rv);
    // out_log.flush();
    return rv;
}
