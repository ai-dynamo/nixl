/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/odm_consistency.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "odm_ioctl.h"

OdmConsistencyContext::OdmConsistencyContext(
    const std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    if (xferBenchConfig::backend != XFERBENCH_BACKEND_ODM ||
        xferBenchConfig::op_type != XFERBENCH_OP_WRITE) {
        return;
    }
    active = true;
    dpa_base = xferBenchConfig::odm_dpa_base;
    if (xferBenchConfig::odm_use_get_iova) {
        return;
    }
    for (const auto &l : iov_lists) {
        for (const auto &v : l) {
            dpa_base = std::min<uint64_t>(dpa_base, v.addr);
        }
    }
    uint64_t hi = 0;
    for (const auto &l : iov_lists) {
        for (const auto &v : l) {
            hi = std::max<uint64_t>(hi, v.addr + v.len);
        }
    }
    dax_map_size = (hi - dpa_base + (2 << 20) - 1) & ~static_cast<size_t>((2 << 20) - 1);
    dax_fd = open(xferBenchConfig::dax_device.c_str(), O_RDWR | O_SYNC);
    if (dax_fd < 0) {
        std::cerr << "ODM: consistency: open(" << xferBenchConfig::dax_device
                  << ") failed: " << strerror(errno) << " (run as root for the DAX window)"
                  << std::endl;
        return;
    }
    dax_map = mmap(nullptr, dax_map_size, PROT_READ | PROT_WRITE, MAP_SHARED, dax_fd, 0);
    if (dax_map == MAP_FAILED) {
        std::cerr << "ODM: consistency: mmap(" << xferBenchConfig::dax_device
                  << ") failed: " << strerror(errno) << std::endl;
        dax_map = nullptr;
        close(dax_fd);
        dax_fd = -1;
    }
}

OdmConsistencyContext::~OdmConsistencyContext() {
    if (dax_map) {
        munmap(dax_map, dax_map_size);
        close(dax_fd);
    }
}

bool
OdmConsistencyContext::fetchWriteBuffer(const xferBenchIOV &iov,
                                        void **addr_out,
                                        bool *allocated_out) {
    *addr_out = nullptr;
    *allocated_out = false;
    if (!active) {
        return false;
    }
    if (dax_map) {
        *addr_out = static_cast<char *>(dax_map) + (iov.addr - dpa_base);
        return true;
    }
    if (!xferBenchConfig::odm_use_get_iova) {
        return true;
    }
    if (iov.len > UINT32_MAX) {
        std::cerr << "ODM: consistency: iov length " << iov.len
                  << " exceeds 32-bit ioctl field limit" << std::endl;
        exit(EXIT_FAILURE);
    }
    void *host = nullptr;
    if (posix_memalign(&host, xferBenchConfig::page_size, iov.len) != 0) {
        std::cerr << "ODM: consistency: host buffer alloc failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    *allocated_out = true;
    struct mrvl_dma_xfer_commands cmd{};
    cmd.host_va_addr = reinterpret_cast<uint64_t>(host);
    cmd.target_iova_addr = iov.addr;
    cmd.tranfer_size = static_cast<uint32_t>(iov.len);
    cmd.tranfer_type = ODM_XTYPE_OUTBOUND;
    cmd.qid = 0;
    int odm_fd = open(xferBenchConfig::odm_device_path.c_str(), O_RDWR);
    if (odm_fd < 0 || ioctl(odm_fd, MRVL_CXL_DMA_READ_COMMAND, &cmd) < 0) {
        std::cerr << "ODM: consistency: host READ ioctl from IOVA 0x" << std::hex << iov.addr
                  << std::dec << " failed" << std::endl;
        if (odm_fd >= 0) {
            close(odm_fd);
        }
        free(host);
        return true;
    }
    close(odm_fd);
    *addr_out = host;
    return true;
}
