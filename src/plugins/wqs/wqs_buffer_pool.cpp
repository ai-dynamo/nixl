// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "wqs_buffer_pool.h"

#include <linux/memfd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace {
constexpr size_t alignment = 512;
}

nixlWqsBufferPool::nixlWqsBufferPool(size_t bytes, bool hugepages) {
    if (bytes == 0 || bytes > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        throw std::invalid_argument("Invalid WQS staging pool size");
    }
    unsigned flags = MFD_CLOEXEC;
    if (hugepages) {
        flags |= MFD_HUGETLB | (21U << MFD_HUGE_SHIFT);
    }
    fd_ = nixl::scopedFd(static_cast<int>(syscall(SYS_memfd_create, "nixl_wqs", flags)));
    if (!fd_.valid()) {
        throw std::system_error(errno, std::generic_category(), "WQS memfd_create");
    }
    if (ftruncate(fd_.get(), static_cast<off_t>(bytes)) != 0) {
        throw std::system_error(errno, std::generic_category(), "WQS ftruncate");
    }
    data_ = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_.get(), 0);
    if (data_ == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "WQS mmap");
    }
    size_ = bytes;
}

nixlWqsBufferPool::~nixlWqsBufferPool() {
    munmap(data_, size_);
}

void
nixlWqsBufferPool::reset() {
    used_ = 0;
}

bool
nixlWqsBufferPool::canFit(size_t bytes) const {
    const size_t padding = (alignment - used_ % alignment) % alignment;
    return (padding <= size_ - used_) && (bytes <= size_ - used_ - padding);
}

void *
nixlWqsBufferPool::allocate(size_t bytes) {
    if (!canFit(bytes)) {
        throw std::length_error("WQS object exceeds staging pool capacity");
    }
    used_ += (alignment - used_ % alignment) % alignment;
    void *buffer = static_cast<char *>(data_) + used_;
    used_ += bytes;
    return buffer;
}
