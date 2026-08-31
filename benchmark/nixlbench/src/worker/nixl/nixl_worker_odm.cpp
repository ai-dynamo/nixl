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

#include "worker/nixl/nixl_worker_odm.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <linux/cxl_mem.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "odm_ioctl.h"
#include "utils/utils.h"

namespace {

constexpr uint64_t kCxlCapUnitBytes = 256ULL * 1024 * 1024;
constexpr size_t kCxlIdentifyPayloadSize = 0x43;
constexpr size_t kCxlIdentifyTotalCapOffset = 0x10;
constexpr size_t kCxlIdentifyPersistentCapOffset = 0x20;
constexpr uint64_t kOdmFallbackAddr = 0x800000000ULL;

bool
odmIoctlSizeOk(size_t size) {
    if (size > UINT32_MAX) {
        std::cerr << "ODM: size " << size << " exceeds 32-bit ioctl field limit" << std::endl;
        return false;
    }
    return true;
}

uint64_t
readCxlCapacityFieldBytes(const unsigned char *id, size_t offset) {
    uint64_t units = 0;

    for (int i = 7; i >= 0; i--) {
        units = (units << 8) | id[offset + i];
    }
    return units * kCxlCapUnitBytes;
}

uint64_t
readCxlOdmCapacityBytes(const std::string &dev_path, uint64_t *total_bytes_out) {
    int fd = open(dev_path.c_str(), O_RDWR);
    if (fd < 0) {
        return 0;
    }

    unsigned char id[kCxlIdentifyPayloadSize];
    memset(id, 0, sizeof(id));
    struct cxl_send_command sc;
    memset(&sc, 0, sizeof(sc));
    sc.id = CXL_MEM_COMMAND_ID_IDENTIFY;
    sc.out.size = sizeof(id);
    sc.out.payload = reinterpret_cast<uint64_t>(id);

    int rc = ioctl(fd, CXL_MEM_SEND_COMMAND, &sc);
    close(fd);
    if (rc < 0 || sc.retval != 0) {
        return 0;
    }

    const uint64_t total_bytes = readCxlCapacityFieldBytes(id, kCxlIdentifyTotalCapOffset);
    const uint64_t persistent_bytes =
        readCxlCapacityFieldBytes(id, kCxlIdentifyPersistentCapOffset);

    if (total_bytes_out) {
        *total_bytes_out = total_bytes;
    }

    if (persistent_bytes > 0) {
        return persistent_bytes;
    }
    if (total_bytes > 0) {
        return total_bytes;
    }
    return 0;
}

} // namespace

namespace xferBenchOdm {

void
configureBackend(const std::vector<std::string> &devices,
                 State &state,
                 std::map<std::string, std::string> &backend_params) {
    const std::string odm_device = (devices.empty() || devices[0] == "all") ? "odm0" : devices[0];
    backend_params["dmadev_param"] = odm_device;
    state.device_path_ = (odm_device[0] == '/') ? odm_device : ("/dev/" + odm_device);
    backend_params["odm_qid"] = std::to_string(kQidStart);
    backend_params["odm_qid_start"] = std::to_string(kQidStart);
    backend_params["odm_qid_end"] = std::to_string(kQidEnd);
    backend_params["num_threads"] = std::to_string(xferBenchConfig::num_threads);
    backend_params["dax_device"] = xferBenchConfig::dax_device;
    std::cout << "ODM backend: dma_device=" << odm_device << " qid=" << kQidStart
              << " qid_range=" << kQidStart << ".." << kQidEnd
              << " threads=" << xferBenchConfig::num_threads
              << " dax_device=" << xferBenchConfig::dax_device
              << " engine=ODM/dma-buf (both directions)"
              << " addr=auto(GET_IOVA)" << std::endl;
}

void
State::freeIova() {
    if (iova_fd_ < 0 || base_addr_ == 0 || !use_get_iova_) {
        return;
    }
    struct mrvl_dma_iova_commands cmd{};
    cmd.target_iova_addr = base_addr_;
    cmd.target_iova_size = iova_size_;
    ioctl(iova_fd_, MRVL_CXL_FREE_IOVA_COMMAND, &cmd);
    close(iova_fd_);
    iova_fd_ = -1;
    iova_size_ = 0;
    use_get_iova_ = false;
}

void
State::seedViaHostWrite(size_t total_size, uint8_t pattern) {
    if (!odmIoctlSizeOk(total_size)) {
        return;
    }
    const std::string &dev = device_path_.empty() ? xferBenchConfig::odm_device_path : device_path_;
    void *host = nullptr;
    if (posix_memalign(&host, xferBenchConfig::page_size, total_size) != 0) {
        std::cerr << "ODM: host seed: allocation failed" << std::endl;
        return;
    }
    memset(host, pattern, total_size);

    int fd = open(dev.c_str(), O_RDWR);
    if (fd < 0) {
        std::cerr << "ODM: host seed: open(" << dev << ") failed: " << strerror(errno) << std::endl;
        free(host);
        return;
    }
    struct mrvl_dma_xfer_commands cmd{};
    cmd.host_va_addr = reinterpret_cast<uint64_t>(host);
    cmd.target_iova_addr = base_addr_;
    cmd.tranfer_size = static_cast<uint32_t>(total_size);
    cmd.tranfer_type = ODM_XTYPE_INBOUND;
    cmd.qid = 0;
    if (ioctl(fd, MRVL_CXL_DMA_WRITE_COMMAND, &cmd) < 0) {
        std::cerr << "ODM: host seed: WRITE ioctl failed: " << strerror(errno) << std::endl;
    } else {
        std::cout << "ODM: seeded " << total_size << " bytes at IOVA 0x" << std::hex << base_addr_
                  << std::dec << " with 0x" << std::hex << static_cast<unsigned>(pattern)
                  << std::dec << " (host WRITE)" << std::endl;
    }
    close(fd);
    free(host);
}

uint64_t
State::discoverBaseAddr() {
    if (base_addr_ != 0) {
        return base_addr_;
    }

    const std::string &odm_dev = device_path_.empty() ? "/dev/odm0" : device_path_;
    xferBenchConfig::odm_device_path = odm_dev;

    uint64_t total_bytes = 0;
    const uint64_t identify_capacity = readCxlOdmCapacityBytes(odm_dev, &total_bytes);
    if (const char *dpa_env = getenv("ODM_DPA_BASE")) {
        const uint64_t v = strtoull(dpa_env, nullptr, 0);
        if (v != 0) {
            dpa_base_ = v;
        }
    }
    if (dpa_base_ == 0) {
        dpa_base_ = kOdmFallbackAddr;
    }
    xferBenchConfig::odm_dpa_base = dpa_base_;
    if (identify_capacity > 0 && xferBenchConfig::total_buffer_size > identify_capacity) {
        std::cerr << "ODM: buffer size " << xferBenchConfig::total_buffer_size
                  << " exceeds CXL IDENTIFY capacity " << identify_capacity << std::endl;
    }

    if (const char *e = getenv("ODM_ADDR")) {
        const uint64_t v = strtoull(e, nullptr, 0);
        if (v != 0) {
            base_addr_ = v;
            use_get_iova_ = false;
            xferBenchConfig::odm_use_get_iova = false;
            std::cout << "ODM: base 0x" << std::hex << base_addr_ << std::dec
                      << " (from ODM_ADDR env)" << std::endl;
            return base_addr_;
        }
    }

    iova_fd_ = open(odm_dev.c_str(), O_RDWR);
    if (iova_fd_ < 0) {
        std::cerr << "ODM: open(" << odm_dev << ") for GET_IOVA failed: " << strerror(errno)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!odmIoctlSizeOk(xferBenchConfig::total_buffer_size)) {
        close(iova_fd_);
        iova_fd_ = -1;
        std::cerr << "ODM: buffer size " << xferBenchConfig::total_buffer_size
                  << " exceeds GET_IOVA ioctl limit" << std::endl;
        exit(EXIT_FAILURE);
    }
    struct mrvl_dma_iova_commands iova_cmd{};
    iova_cmd.target_iova_size = static_cast<uint32_t>(xferBenchConfig::total_buffer_size);
    if (ioctl(iova_fd_, MRVL_CXL_GET_IOVA_COMMAND, &iova_cmd) < 0) {
        std::cerr << "ODM: GET_IOVA failed: " << strerror(errno) << std::endl;
        close(iova_fd_);
        iova_fd_ = -1;
        exit(EXIT_FAILURE);
    }
    base_addr_ = iova_cmd.target_iova_addr;
    iova_size_ = iova_cmd.target_iova_size;
    use_get_iova_ = true;
    xferBenchConfig::odm_use_get_iova = true;
    std::cout << "ODM: allocated IOVA 0x" << std::hex << base_addr_ << std::dec << " size "
              << iova_size_ << " via GET_IOVA on " << odm_dev << " (DPA base 0x" << std::hex
              << dpa_base_ << std::dec << " used for DAX alias only)" << std::endl;
    return base_addr_;
}

void
State::seedDramForRead(size_t total_size) {
    if (use_get_iova_) {
        seedViaHostWrite(total_size, XFERBENCH_TARGET_BUFFER_ELEMENT);
        return;
    }
    const std::string &dax = xferBenchConfig::dax_device;
    if (dax.empty()) {
        std::cerr << "ODM: consistency seed: --dax_device not set, skipping seed" << std::endl;
        return;
    }
    int fd = open(dax.c_str(), O_RDWR | O_SYNC);
    if (fd < 0) {
        std::cerr << "ODM: consistency seed: open(" << dax << ") failed: " << strerror(errno)
                  << " (run as root for the DAX window)" << std::endl;
        return;
    }
    if (base_addr_ < dpa_base_) {
        std::cerr << "ODM: consistency seed: base 0x" << std::hex << base_addr_
                  << " is below DPA base 0x" << dpa_base_ << std::dec << ", skipping seed"
                  << std::endl;
        close(fd);
        return;
    }
    const uint64_t dax_offset = base_addr_ - dpa_base_;
    size_t map_size = (static_cast<size_t>(dax_offset) + total_size + (2 << 20) - 1) &
        ~static_cast<size_t>((2 << 20) - 1);
    void *p = mmap(
        nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, static_cast<off_t>(dax_offset));
    if (p == MAP_FAILED) {
        std::cerr << "ODM: consistency seed: mmap(" << dax << ", " << map_size
                  << ") failed: " << strerror(errno) << std::endl;
        close(fd);
        return;
    }
    memset(p, XFERBENCH_TARGET_BUFFER_ELEMENT, total_size);
    __sync_synchronize();
    munmap(p, map_size);
    close(fd);
    std::cout << "ODM: seeded " << total_size << " bytes of Iliad DRAM with 0x" << std::hex
              << (int)XFERBENCH_TARGET_BUFFER_ELEMENT << std::dec << " via " << dax
              << " for READ consistency" << std::endl;
}

} // namespace xferBenchOdm
