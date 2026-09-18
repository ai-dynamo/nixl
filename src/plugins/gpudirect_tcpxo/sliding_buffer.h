/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#ifndef GPUDIRECT_TCPXO_SLIDING_BUFFER_H_
#define GPUDIRECT_TCPXO_SLIDING_BUFFER_H_

#include <cmath>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <vector>

namespace tcpxo {

/**
 * @brief A non-thread safe buffer with a sliding head and tail
 * @details
 * There's always some contiguous region of valid data in this buffer, and only one such region.
 * Data can be read from the start of the region, and written to the end. If the buffer gets full,
 * but we have enough space at the front of the buffer to satisfy the incoming data, we slide the
 * entire region down to the front. If the buffer doesn't have enough space even after sliding the
 * entire region, we also increase the size.
 *
 * TODO: consider replacing usage with absl::Cord when we have protobuf deserialization support
 */
class SlidingBuffer {
public:
    static constexpr double kGrowthFactor = 1.2;

    uint8_t *
    write_head() {
        return buffer_.data() + write_head_;
    }

    size_t
    GetFreeSpace() const {
        return buffer_.size() - write_head_;
    }

    /**
     * @brief Inform the buffer that bytes have been written to the write_head
     */
    void
    CommitWrite(size_t bytes_written) {
        write_head_ += bytes_written;
    }

    const uint8_t *const
    read_head() const {
        return buffer_.data() + read_head_;
    }

    size_t
    GetReadableSpace() const {
        return write_head_ - read_head_;
    }

    /**
     * @brief Inform the buffer that bytes have been read from the read_head
     */
    void
    Consume(size_t bytes_parsed) {
        read_head_ += bytes_parsed;

        if (read_head_ == write_head_) {
            read_head_ = 0;
            write_head_ = 0;
        }
    }

    void
    reserve(size_t required_space) {
        if (GetFreeSpace() >= required_space) {
            return;
        }

        auto pending_data_len = GetReadableSpace();
        // compact the buffer, this might be all we need to do to get enough free space
        if (pending_data_len > 0 && read_head_ > 0) {
            std::memmove(buffer_.data(), buffer_.data() + read_head_, pending_data_len);
        }
        // we may also need to resize
        if (buffer_.size() - pending_data_len < required_space) {
            buffer_.resize(
                std::max(static_cast<size_t>(std::llround(buffer_.size() * kGrowthFactor)),
                         pending_data_len + required_space));
        }

        // Reset heads after sliding/resizing
        read_head_ = 0;
        write_head_ = pending_data_len;
    }

private:
    std::vector<uint8_t> buffer_;
    size_t read_head_{0};
    size_t write_head_{0};
};

} // namespace tcpxo

#endif // GPUDIRECT_TCPXO_SLIDING_BUFFER_H_