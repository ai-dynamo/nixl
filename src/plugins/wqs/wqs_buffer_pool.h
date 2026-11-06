// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_WQS_WQS_BUFFER_POOL_H
#define NIXL_SRC_PLUGINS_WQS_WQS_BUFFER_POOL_H

#include "common/scoped_fd.h"

#include <cstddef>

/** A memfd-backed staging region with 512-byte-aligned allocations per sub-batch. */
class nixlWqsBufferPool {
public:
    explicit nixlWqsBufferPool(size_t bytes, bool hugepages = true);
    ~nixlWqsBufferPool();
    nixlWqsBufferPool(const nixlWqsBufferPool &) = delete;
    nixlWqsBufferPool &
    operator=(const nixlWqsBufferPool &) = delete;

    void
    reset();
    [[nodiscard]] bool
    canFit(size_t bytes) const;
    [[nodiscard]] void *
    allocate(size_t bytes);

    [[nodiscard]] void *
    data() const noexcept {
        return data_;
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return size_;
    }

    [[nodiscard]] int
    fd() const noexcept {
        return fd_.get();
    }

private:
    nixl::scopedFd fd_;
    void *data_ = nullptr;
    size_t size_ = 0;
    size_t used_ = 0;
};

#endif // NIXL_SRC_PLUGINS_WQS_WQS_BUFFER_POOL_H
