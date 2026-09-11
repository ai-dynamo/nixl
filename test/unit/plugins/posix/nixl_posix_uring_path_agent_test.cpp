/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <iostream>

namespace {

constexpr char agent_name[] = "POSIXUringEagerOpenReuse";

bool
runXfer(nixlAgent &agent,
        nixl_xfer_op_t operation,
        const nixl_xfer_dlist_t &dram,
        const nixl_xfer_dlist_t &file) {
    nixlXferReqH *request = nullptr;
    nixl_status_t status = agent.createXferReq(operation, dram, file, agent_name, request);
    if (status == NIXL_SUCCESS) {
        status = agent.postXferReq(request);
        while (status == NIXL_IN_PROG) {
            status = agent.getXferStatus(request);
        }
        agent.releaseXferReq(request);
    }
    return status == NIXL_SUCCESS;
}

int
fail(const char *message, const char *path) {
    unlink(path);
    std::cerr << message << std::endl;
    return 1;
}

} // namespace

int
runPosixUringPathAgentTest() {
    nixlAgentConfig config;
    nixlAgent agent(agent_name, config);
    nixl_b_params_t params;
    params["use_uring"] = "true";

    nixlBackendH *backend = nullptr;
    if (agent.createBackend("POSIX", params, backend) != NIXL_SUCCESS || !backend) {
        std::cout << "SKIP: kernel lacks asynchronous io_uring open support" << std::endl;
        return 77;
    }

    char path[] = "/tmp/nixl-uring-eager-agent-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        return 1;
    }
    std::array<unsigned char, 4096> expected;
    expected.fill(0x6b);
    if (write(fd, expected.data(), expected.size()) != static_cast<ssize_t>(expected.size())) {
        close(fd);
        return fail("failed to initialize agent eager-open test", path);
    }
    close(fd);

    const uint64_t wide_dev_id = static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1;
    nixl_reg_dlist_t wide_path_reg(FILE_SEG);
    nixlBlobDesc wide_path_desc;
    wide_path_desc.addr = 0;
    wide_path_desc.len = expected.size();
    wide_path_desc.devId = wide_dev_id;
    wide_path_desc.metaInfo = std::string("ro:") + path;
    wide_path_reg.addDesc(wide_path_desc);
    if (agent.registerMem(wide_path_reg) != NIXL_SUCCESS) {
        return fail("64-bit path-mode devId registration failed", path);
    }
    if (agent.registerMem(wide_path_reg) == NIXL_SUCCESS) {
        return fail("duplicate path-mode devId registration was not rejected", path);
    }
    if (agent.deregisterMem(wide_path_reg) != NIXL_SUCCESS) {
        return fail("64-bit path-mode devId deregistration failed", path);
    }
    if (agent.registerMem(wide_path_reg) != NIXL_SUCCESS) {
        return fail("released path-mode devId could not be registered again", path);
    }
    if (agent.deregisterMem(wide_path_reg) != NIXL_SUCCESS) {
        return fail("re-registered path-mode devId deregistration failed", path);
    }

    nixl_reg_dlist_t invalid_fd_reg(FILE_SEG);
    nixlBlobDesc invalid_fd_desc;
    invalid_fd_desc.addr = 0;
    invalid_fd_desc.len = expected.size();
    invalid_fd_desc.devId = wide_dev_id;
    invalid_fd_reg.addDesc(invalid_fd_desc);
    if (agent.registerMem(invalid_fd_reg) == NIXL_SUCCESS) {
        return fail("out-of-range fd-mode devId was not rejected", path);
    }

    nixl_reg_dlist_t file_reg(FILE_SEG);
    nixlBlobDesc file_desc;
    file_desc.addr = 0;
    file_desc.len = expected.size();
    file_desc.devId = 1;
    file_desc.metaInfo = std::string("ro:") + path;
    file_reg.addDesc(file_desc);

    std::array<unsigned char, 4096> buffer{};
    nixl_reg_dlist_t dram_reg(DRAM_SEG);
    nixlBlobDesc dram_desc;
    dram_desc.addr = reinterpret_cast<uintptr_t>(buffer.data());
    dram_desc.len = buffer.size();
    dram_desc.devId = 0;
    dram_reg.addDesc(dram_desc);

    if (agent.registerMem(file_reg) != NIXL_SUCCESS ||
        agent.registerMem(dram_reg) != NIXL_SUCCESS) {
        return fail("eager asynchronous path registration failed", path);
    }

    const nixl_xfer_dlist_t file_xfer = file_reg.trim();
    const nixl_xfer_dlist_t dram_xfer = dram_reg.trim();
    if (!runXfer(agent, NIXL_READ, dram_xfer, file_xfer) ||
        !std::equal(buffer.begin(), buffer.end(), expected.begin())) {
        return fail("first registered eager-open transfer failed", path);
    }

    if (unlink(path) != 0) {
        return fail("failed to unlink after first eager-open transfer", path);
    }
    buffer.fill(0);
    if (!runXfer(agent, NIXL_READ, dram_xfer, file_xfer) ||
        !std::equal(buffer.begin(), buffer.end(), expected.begin())) {
        return fail("second transfer did not reuse the registered open file", path);
    }

    char create_path[] = "/tmp/nixl-uring-eager-create-XXXXXX";
    int create_fd = mkstemp(create_path);
    if (create_fd < 0) {
        return fail("failed to reserve eager-create path", path);
    }
    close(create_fd);
    unlink(create_path);

    nixl_reg_dlist_t create_reg(FILE_SEG);
    nixlBlobDesc create_desc;
    create_desc.addr = 0;
    create_desc.len = buffer.size();
    create_desc.devId = 2;
    create_desc.metaInfo = std::string("rw,create:") + create_path;
    create_reg.addDesc(create_desc);
    buffer.fill(0x3c);
    if (agent.registerMem(create_reg) != NIXL_SUCCESS) {
        return fail("eager rw,create registration failed", create_path);
    }
    if (!runXfer(agent, NIXL_WRITE, dram_xfer, create_reg.trim()) ||
        access(create_path, F_OK) != 0) {
        return fail("transfer waiting on eager create failed", create_path);
    }
    if (agent.deregisterMem(create_reg) != NIXL_SUCCESS) {
        return fail("eager-created file deregistration failed", create_path);
    }
    unlink(create_path);

    char missing_path[] = "/tmp/nixl-uring-eager-missing-XXXXXX";
    int missing_fd = mkstemp(missing_path);
    if (missing_fd < 0) {
        return fail("failed to reserve missing-file path", path);
    }
    close(missing_fd);
    unlink(missing_path);

    nixl_reg_dlist_t missing_reg(FILE_SEG);
    nixlBlobDesc missing_desc;
    missing_desc.addr = 0;
    missing_desc.len = buffer.size();
    missing_desc.devId = 3;
    missing_desc.metaInfo = std::string("ro:") + missing_path;
    missing_reg.addDesc(missing_desc);
    if (agent.registerMem(missing_reg) != NIXL_SUCCESS) {
        return fail("missing path failed during eager asynchronous registration", missing_path);
    }
    if (runXfer(agent, NIXL_READ, dram_xfer, missing_reg.trim())) {
        return fail("missing path unexpectedly transferred successfully", missing_path);
    }

    std::array<unsigned char, 4096> sibling_buffer{};
    nixl_reg_dlist_t sibling_dram_reg(DRAM_SEG);
    nixlBlobDesc sibling_dram_desc;
    sibling_dram_desc.addr = reinterpret_cast<uintptr_t>(sibling_buffer.data());
    sibling_dram_desc.len = sibling_buffer.size();
    sibling_dram_desc.devId = 0;
    sibling_dram_reg.addDesc(sibling_dram_desc);
    if (agent.registerMem(sibling_dram_reg) != NIXL_SUCCESS) {
        return fail("failed to register mixed-transfer buffer", missing_path);
    }

    nixl_xfer_dlist_t mixed_dram(DRAM_SEG);
    mixed_dram.addDesc(*dram_xfer.begin());
    mixed_dram.addDesc(*sibling_dram_reg.trim().begin());
    nixl_xfer_dlist_t mixed_file(FILE_SEG);
    mixed_file.addDesc(*file_xfer.begin());
    mixed_file.addDesc(*missing_reg.trim().begin());
    buffer.fill(0);
    if (runXfer(agent, NIXL_READ, mixed_dram, mixed_file) ||
        std::any_of(buffer.begin(), buffer.end(), [](unsigned char value) { return value != 0; })) {
        return fail("failed-open transfer did not cancel its valid sibling", missing_path);
    }
    if (!runXfer(agent, NIXL_READ, dram_xfer, file_xfer)) {
        return fail("queue was unusable after failed-open cancellation", missing_path);
    }

    if (agent.deregisterMem(missing_reg) != NIXL_SUCCESS ||
        agent.deregisterMem(file_reg) != NIXL_SUCCESS ||
        agent.deregisterMem(sibling_dram_reg) != NIXL_SUCCESS ||
        agent.deregisterMem(dram_reg) != NIXL_SUCCESS) {
        return fail("failure-path deregistration failed", missing_path);
    }

    nixlAgent synchronous_agent("POSIXUringOpenSynchronous", config);
    nixl_b_params_t synchronous_params;
    synchronous_params["use_uring"] = "true";
    synchronous_params["uring_open_synchronous"] = "true";
    nixlBackendH *synchronous_backend = nullptr;
    if (synchronous_agent.createBackend("POSIX", synchronous_params, synchronous_backend) !=
            NIXL_SUCCESS ||
        !synchronous_backend) {
        return fail("failed to create synchronous-open backend", missing_path);
    }

    char synchronous_missing_path[] = "/tmp/nixl-uring-synchronous-missing-XXXXXX";
    int synchronous_missing_fd = mkstemp(synchronous_missing_path);
    if (synchronous_missing_fd < 0) {
        return fail("failed to reserve synchronous missing-file path", missing_path);
    }
    close(synchronous_missing_fd);
    unlink(synchronous_missing_path);

    nixl_reg_dlist_t synchronous_missing_reg(FILE_SEG);
    nixlBlobDesc synchronous_missing_desc;
    synchronous_missing_desc.addr = 0;
    synchronous_missing_desc.len = buffer.size();
    synchronous_missing_desc.devId = 4;
    synchronous_missing_desc.metaInfo = std::string("ro:") + synchronous_missing_path;
    synchronous_missing_reg.addDesc(synchronous_missing_desc);
    if (synchronous_agent.registerMem(synchronous_missing_reg) == NIXL_SUCCESS) {
        return fail("synchronous open did not report the failure", synchronous_missing_path);
    }

    std::cout << "asynchronous open and synchronous open error reporting: OK" << std::endl;
    return 0;
}
