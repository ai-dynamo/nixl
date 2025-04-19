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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>
#include <sys/time.h>
#include <boost/asio.hpp>

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nixl.h"

using namespace boost::asio;
using ip::tcp;

DEFINE_string(mode, "initiator",
              "Running mode: initiator or target. Initiator node read/write "
              "data blocks from target node");
DEFINE_string(operation, "read", "Operation type: read or write");
DEFINE_string(protocol, "UCX", "Protocol: UCX or Mooncake");

DEFINE_string(segment_id, "192.168.3.76", "Segment ID to access data");
DEFINE_uint64(buffer_size, 1ull << 30, "total size of data buffer");
DEFINE_int32(batch_size, 128, "Batch size");
DEFINE_uint64(block_size, 4096, "Block size for each transfer request");
DEFINE_int32(duration, 10, "Test duration in seconds");
DEFINE_int32(threads, 4, "Task submission threads");
DEFINE_string(report_unit, "GB", "Report unit: GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb");
DEFINE_uint32(report_precision, 2, "Report precision");

const static std::unordered_map<std::string, uint64_t> RATE_UNIT_MP = {
    {"GB", 1000ull * 1000ull * 1000ull},
    {"GiB", 1ull << 30},
    {"Gb", 1000ull * 1000ull * 1000ull / 8},
    {"MB", 1000ull * 1000ull},
    {"MiB", 1ull << 20},
    {"Mb", 1000ull * 1000ull / 8},
    {"KB", 1000ull},
    {"KiB", 1ull << 10},
    {"Kb", 1000ull / 8}};

static inline std::string calculateRate(uint64_t data_bytes, double duration) {
    if (std::fabs(duration) < 1e-10) {
        LOG(ERROR) << "Invalid args: duration shouldn't be 0";
        return "";
    }
    if (!RATE_UNIT_MP.count(FLAGS_report_unit)) {
        LOG(WARNING) << "Invalid flag: report_unit only support "
                        "GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb, not support "
                     << FLAGS_report_unit
                     << " . Now use GB(default) as report_unit";
        FLAGS_report_unit = "GB";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(FLAGS_report_precision)
        << 1.0 * data_bytes / duration / RATE_UNIT_MP.at(FLAGS_report_unit)
        << " " << FLAGS_report_unit << "/s";
    return oss.str();
}

volatile bool running = true;
std::atomic<size_t> total_batch_count(0);

void initiatorWorker(int thread_id, void* local_addr, uint64_t remote_base,
                     nixlAgent *agent, nixl_opt_args_t *extra_params) {
    nixl_xfer_op_t opcode;
    if (FLAGS_operation == "read")
        opcode = NIXL_READ;
    else if (FLAGS_operation == "write")
        opcode = NIXL_WRITE;
    else {
        LOG(ERROR) << "Unsupported operation: must be 'read' or 'write'";
        exit(EXIT_FAILURE);
    }

    size_t batch_count = 0;
    while (running) {
        nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
        nixl_xfer_dlist_t req_dst_descs(DRAM_SEG);

        for (int i = 0; i < FLAGS_batch_size; ++i) {
            nixlBasicDesc req_src;
            req_src.addr = (uintptr_t)(local_addr) +
                           FLAGS_block_size * (i * FLAGS_threads + thread_id);
            req_src.len = FLAGS_block_size;
            req_src.devId = 0;
            req_src_descs.addDesc(req_src);

            nixlBasicDesc req_dst;
            req_dst.addr = remote_base +
                           FLAGS_block_size * (i * FLAGS_threads + thread_id);
            req_dst.len = FLAGS_block_size;
            req_dst.devId = 0;
            req_dst_descs.addDesc(req_dst);
        }

        nixlXferReqH* req_handle;
        auto ret = agent->createXferReq(opcode, req_src_descs, req_dst_descs,
                                       "target", req_handle, extra_params);

        assert(ret == NIXL_SUCCESS);

        nixl_status_t status = agent->postXferReq(req_handle);
        while (status != NIXL_SUCCESS) {
            status = agent->getXferStatus(req_handle);
            assert(status >= 0);
        }

        agent->releaseXferReq(req_handle);
        batch_count++;
    }

    LOG(INFO) << "Worker " << thread_id << " stopped!";
    total_batch_count.fetch_add(batch_count);
}

int initiator() {
    nixlAgentConfig cfg(true);
    nixlAgent agent("initiator", cfg);
    nixl_opt_args_t extra_params;
    nixl_b_params_t init;
    nixl_mem_list_t mems;
    std::vector<nixl_backend_t> plugins;
    auto ret = agent.getAvailPlugins(plugins);
    assert(ret == NIXL_SUCCESS);
    nixlBackendH* backend;
    ret = agent.createBackend(FLAGS_protocol, init, backend);
    assert(ret == NIXL_SUCCESS);
    extra_params.backends.push_back(backend);

    nixl_reg_dlist_t dlist(DRAM_SEG);
    void* addr = malloc(FLAGS_buffer_size);
    nixlBlobDesc buff;
    buff.addr = (uintptr_t)addr;
    buff.len = FLAGS_buffer_size;
    buff.devId = 0;
    dlist.addDesc(buff);

    ret = agent.registerMem(dlist, &extra_params);
    assert(ret == NIXL_SUCCESS);

    std::string meta;
    ret = agent.getLocalMD(meta);
    assert(ret == NIXL_SUCCESS);

    const short port = 12345;
    std::vector<char> buf(4096);
    try {
        io_context io;
        tcp::socket socket(io);
        socket.connect(tcp::endpoint(ip::make_address(FLAGS_segment_id), port));
        read(socket, buffer(buf), transfer_exactly(4096));

        socket.shutdown(tcp::socket::shutdown_both);
        socket.close();
    } catch (std::exception& e) {
        std::cerr << "Initiator error: " << e.what() << std::endl;
    }

    std::string target_meta, ret_string;
    uint64_t remote_base;

    remote_base = *(uint64_t *) ((char *) buf.data());
    auto target_meta_length = *(uint64_t *) ((char *) buf.data() + sizeof(uint64_t));
    LOG(INFO) << (void *) remote_base << " " << target_meta_length;
    target_meta = std::string((char *) buf.data() + 2 * sizeof(uint64_t), target_meta_length);

    ret = agent.loadRemoteMD(target_meta, ret_string);
    assert(ret == NIXL_SUCCESS);

    std::thread workers[FLAGS_threads];

    struct timeval start_tv, stop_tv;
    gettimeofday(&start_tv, nullptr);

    for (int i = 0; i < FLAGS_threads; ++i)
        workers[i] = std::thread(initiatorWorker, i, addr, remote_base, &agent, &extra_params);

    sleep(FLAGS_duration);
    running = false;

    for (int i = 0; i < FLAGS_threads; ++i) workers[i].join();

    gettimeofday(&stop_tv, nullptr);
    auto duration = (stop_tv.tv_sec - start_tv.tv_sec) +
                    (stop_tv.tv_usec - start_tv.tv_usec) / 1000000.0;
    auto batch_count = total_batch_count.load();

    LOG(INFO) << "Test completed: duration " << std::fixed
              << std::setprecision(2) << duration << ", batch count "
              << batch_count << ", throughput "
              << calculateRate(
                     batch_count * FLAGS_batch_size * FLAGS_block_size,
                     duration);

    free(addr);
    return 0;
}

volatile bool target_running = true;

int target() {
    nixlAgentConfig cfg(true);
    nixlAgent agent("target", cfg);
    nixl_b_params_t init;
    nixl_mem_list_t mems;
    std::vector<nixl_backend_t> plugins;
    auto ret = agent.getAvailPlugins(plugins);
    assert(ret == NIXL_SUCCESS);
    nixlBackendH* backend;
    ret = agent.createBackend(FLAGS_protocol, init, backend);
    assert(ret == NIXL_SUCCESS);
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend);

    nixl_reg_dlist_t dlist(DRAM_SEG);
    void* addr = malloc(FLAGS_buffer_size);
    nixlBlobDesc buff;
    buff.addr = (uintptr_t)addr;
    buff.len = FLAGS_buffer_size;
    buff.devId = 0;
    dlist.addDesc(buff);

    ret = agent.registerMem(dlist, &extra_params);
    assert(ret == NIXL_SUCCESS);

    std::string meta;
    ret = agent.getLocalMD(meta);
    assert(ret == NIXL_SUCCESS);

    const short port = 12345;
    try {
        io_context io;
        tcp::acceptor acceptor(io, tcp::endpoint(tcp::v4(), port));

        LOG(INFO) << "Start listening";

        tcp::socket socket(io);
        acceptor.accept(socket);

        std::vector<char> data(4096, 0);
        size_t meta_size = meta.size();
        memcpy(data.data(), &addr, sizeof(uint64_t));
        memcpy((char *) data.data() + sizeof(uint64_t), &meta_size, sizeof(uint64_t));
        memcpy((char *) data.data() + 2 * sizeof(uint64_t), meta.data(), meta_size);
        LOG(INFO) << addr << " " << meta_size;
        write(socket, buffer(data));

        socket.shutdown(tcp::socket::shutdown_both);
        socket.close();
    } catch (std::exception& e) {
        std::cerr << "Target error: " << e.what() << std::endl;
    }

    while (target_running) sleep(1);

    return 0;
}

void check_total_buffer_size() {
    uint64_t require_size = FLAGS_block_size * FLAGS_batch_size * FLAGS_threads;
    if (FLAGS_buffer_size < require_size) {
        FLAGS_buffer_size = require_size;
        LOG(WARNING) << "Invalid flag: buffer size is smaller than "
                        "require_size, adjust to "
                     << require_size;
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    check_total_buffer_size();

    if (FLAGS_mode == "initiator")
        return initiator();
    else if (FLAGS_mode == "target")
        return target();

    LOG(ERROR) << "Unsupported mode: must be 'initiator' or 'target'";
    exit(EXIT_FAILURE);
}
