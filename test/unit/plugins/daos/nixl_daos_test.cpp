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

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "nixl.h"

namespace {

constexpr size_t DEFAULT_TRANSFER_SIZE = 1 << 20;
constexpr size_t DEFAULT_NUM_TRANSFERS = 8;
constexpr size_t DEFAULT_ITERATIONS = 3;

void
usage(const char *program) {
    std::cerr << "Usage: " << program
              << " POOL CONTAINER [TRANSFER_SIZE] [NUM_TRANSFERS] [ITERATIONS]\n"
              << "Example: " << program << " nixl_pool nixl_cont 1048576 32 10\n";
}

bool
waitForTransfer(nixlAgent &agent, nixlXferReqH *request) {
    nixl_status_t status = agent.postXferReq(request);
    while (status == NIXL_IN_PROG) {
        status = agent.getXferStatus(request);
    }
    return status == NIXL_SUCCESS;
}

} // namespace

int
main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        usage(argv[0]);
        return 2;
    }

    const std::string pool = argv[1];
    const std::string container = argv[2];
    const size_t transfer_size = argc > 3 ? std::stoull(argv[3]) : DEFAULT_TRANSFER_SIZE;
    const size_t num_transfers = argc > 4 ? std::stoull(argv[4]) : DEFAULT_NUM_TRANSFERS;
    const size_t iterations = argc > 5 ? std::stoull(argv[5]) : DEFAULT_ITERATIONS;
    if (transfer_size == 0 || num_transfers == 0 || iterations == 0) {
        usage(argv[0]);
        return 2;
    }

    nixlAgentConfig config;
    config.useProgThread = true;
    nixlAgent agent("DaosTester", config);
    nixlBackendH *backend = nullptr;
    nixl_b_params_t params = {{"pool", pool}, {"container", container}};
    if (agent.createBackend("DAOS", params, backend) != NIXL_SUCCESS || backend == nullptr) {
        std::cerr << "Failed to create DAOS backend. Check NIXL_PLUGIN_DIR and DAOS connectivity."
                  << std::endl;
        return 1;
    }

    std::vector<std::unique_ptr<char[]>> write_buffers;
    std::vector<std::unique_ptr<char[]>> read_buffers;
    std::vector<nixlBlobDesc> write_descs(num_transfers);
    std::vector<nixlBlobDesc> read_descs(num_transfers);
    std::vector<nixlBlobDesc> object_descs(num_transfers);
    nixl_reg_dlist_t write_reg(DRAM_SEG);
    nixl_reg_dlist_t read_reg(DRAM_SEG);
    nixl_reg_dlist_t object_reg(OBJ_SEG);

    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string prefix =
        "nixl-daos-test-" + std::to_string(getpid()) + "-" + std::to_string(timestamp);
    for (size_t i = 0; i < num_transfers; ++i) {
        write_buffers.emplace_back(std::make_unique<char[]>(transfer_size));
        read_buffers.emplace_back(std::make_unique<char[]>(transfer_size));
        std::memset(write_buffers.back().get(), static_cast<int>((i % 251) + 1), transfer_size);
        std::memset(read_buffers.back().get(), 0, transfer_size);

        write_descs[i] =
            nixlBlobDesc(reinterpret_cast<uintptr_t>(write_buffers.back().get()), transfer_size, 0);
        read_descs[i] =
            nixlBlobDesc(reinterpret_cast<uintptr_t>(read_buffers.back().get()), transfer_size, 0);
        object_descs[i] = nixlBlobDesc(0, transfer_size, i, prefix + "-" + std::to_string(i));
        write_reg.addDesc(write_descs[i]);
        read_reg.addDesc(read_descs[i]);
        object_reg.addDesc(object_descs[i]);
    }

    nixl_opt_args_t registration_options;
    registration_options.backends.push_back(backend);
    bool write_registered = false;
    bool read_registered = false;
    bool object_registered = false;
    nixlXferReqH *write_request = nullptr;
    nixlXferReqH *read_request = nullptr;
    int result = 1;

    if (agent.registerMem(write_reg, &registration_options) != NIXL_SUCCESS) {
        std::cerr << "Failed to register write buffers" << std::endl;
        goto cleanup;
    }
    write_registered = true;
    if (agent.registerMem(read_reg, &registration_options) != NIXL_SUCCESS) {
        std::cerr << "Failed to register read buffers" << std::endl;
        goto cleanup;
    }
    read_registered = true;
    if (agent.registerMem(object_reg, &registration_options) != NIXL_SUCCESS) {
        std::cerr << "Failed to register DAOS objects" << std::endl;
        goto cleanup;
    }
    object_registered = true;

    {
        nixl_xfer_dlist_t write_list = write_reg.trim();
        nixl_xfer_dlist_t read_list = read_reg.trim();
        nixl_xfer_dlist_t object_list = object_reg.trim();
        if (agent.createXferReq(NIXL_WRITE, write_list, object_list, "DaosTester", write_request) !=
                NIXL_SUCCESS ||
            agent.createXferReq(NIXL_READ, read_list, object_list, "DaosTester", read_request) !=
                NIXL_SUCCESS) {
            std::cerr << "Failed to create DAOS transfer requests" << std::endl;
            goto cleanup;
        }
    }

    for (size_t i = 0; i < iterations; ++i) {
        if (!waitForTransfer(agent, write_request)) {
            std::cerr << "DAOS write failed in iteration " << i << std::endl;
            goto cleanup;
        }
    }
    for (auto &buffer : read_buffers) {
        std::memset(buffer.get(), 0, transfer_size);
    }
    for (size_t i = 0; i < iterations; ++i) {
        if (!waitForTransfer(agent, read_request)) {
            std::cerr << "DAOS read failed in iteration " << i << std::endl;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < num_transfers; ++i) {
        if (std::memcmp(write_buffers[i].get(), read_buffers[i].get(), transfer_size) != 0) {
            std::cerr << "Data verification failed for object " << object_descs[i].metaInfo
                      << std::endl;
            goto cleanup;
        }
    }

    std::cout << "DAOS write/read verification passed for " << num_transfers << " objects ("
              << transfer_size << " bytes each, " << iterations << " iterations)." << std::endl;
    std::cout << "Object prefix: " << prefix << std::endl;
    result = 0;

cleanup:
    if (write_request != nullptr) {
        agent.releaseXferReq(write_request);
    }
    if (read_request != nullptr) {
        agent.releaseXferReq(read_request);
    }
    if (object_registered) {
        agent.deregisterMem(object_reg);
    }
    if (read_registered) {
        agent.deregisterMem(read_reg);
    }
    if (write_registered) {
        agent.deregisterMem(write_reg);
    }
    return result;
}
