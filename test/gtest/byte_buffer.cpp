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

#include "common/byte_buffer.h"
#include "gtest/gtest.h"
#include <cstdint>

namespace nixl {

TEST(ByteBuffer, Put) {
    inputByteBuffer b;
    b.put(static_cast<std::uint8_t>(1));
    EXPECT_EQ(b.size(), 1);
    EXPECT_EQ(b.data()[0], 1);

    b.put(static_cast<std::int32_t>(2));
    EXPECT_EQ(b.size(), 5);
    std::int32_t v32;
    memcpy(&v32, &b.data()[1], sizeof(std::int32_t));
    EXPECT_EQ(v32, 2);

    b.put("/" + std::string("Bar"));
    EXPECT_EQ(b.size(), 17);
    std::uint64_t s;
    memcpy(&s, &b.data()[5], sizeof(std::uint64_t));
    EXPECT_EQ(s, 4);
    EXPECT_EQ(b.data()[13], '/');
    EXPECT_EQ(b.data()[14], 'B');
    EXPECT_EQ(b.data()[15], 'a');
    EXPECT_EQ(b.data()[16], 'r');

    b.put(std::string("Foo"));
    EXPECT_EQ(b.size(), 28);
    memcpy(&s, &b.data()[17], sizeof(std::uint64_t));
    EXPECT_EQ(s, 3);
    EXPECT_EQ(b.data()[25], 'F');
    EXPECT_EQ(b.data()[26], 'o');
    EXPECT_EQ(b.data()[27], 'o');

    b.put(std::vector<std::int8_t>{1, 2, 3});
    EXPECT_EQ(b.size(), 39);
    memcpy(&s, &b.data()[28], sizeof(std::uint64_t));
    EXPECT_EQ(s, 3);
    EXPECT_EQ(b.data()[36], 1);
    EXPECT_EQ(b.data()[37], 2);
    EXPECT_EQ(b.data()[38], 3);

    std::int8_t arr[] = {1, 2, 3};
    b.put(arr);
    EXPECT_EQ(b.size(), 50);
    memcpy(&s, &b.data()[39], sizeof(std::uint64_t));
    EXPECT_EQ(s, 3);
    EXPECT_EQ(b.data()[47], 1);
    EXPECT_EQ(b.data()[48], 2);
    EXPECT_EQ(b.data()[49], 3);

    // Enums can be put into the buffer using views::transform.
    b.put(std::array<std::byte, 3>{std::byte(1), std::byte(2), std::byte(3)} |
          std::views::transform([](std::byte v) { return static_cast<std::int8_t>(v); }));
    EXPECT_EQ(b.size(), 61);
    memcpy(&s, &b.data()[50], sizeof(std::uint64_t));
    EXPECT_EQ(s, 3);
    EXPECT_EQ(b.data()[58], 1);
    EXPECT_EQ(b.data()[59], 2);
    EXPECT_EQ(b.data()[60], 3);
}

TEST(ByteBuffer, VarArg) {
    inputByteBuffer b;
    b.putValues(static_cast<std::uint8_t>(1),
                static_cast<std::int32_t>(2),
                "/" + std::string("Bar"),
                std::string("Foo"),
                std::vector<std::int8_t>{1, 2, 3},
                std::array<std::byte, 3>{std::byte(1), std::byte(2), std::byte(3)} |
                    std::views::transform([](std::byte v) { return static_cast<std::int8_t>(v); }));
    EXPECT_EQ(b.size(), 50);

    EXPECT_EQ(*&b.data()[0], 1);

    std::int32_t v32;
    memcpy(&v32, &b.data()[1], sizeof(std::int32_t));
    EXPECT_EQ(v32, 2);

    std::uint64_t s;
    memcpy(&s, &b.data()[5], sizeof(std::uint64_t));
    EXPECT_EQ(s, 4);
    EXPECT_EQ(b.data()[13], '/');
    EXPECT_EQ(b.data()[14], 'B');
    EXPECT_EQ(b.data()[15], 'a');
    EXPECT_EQ(b.data()[16], 'r');

    memcpy(&s, &b.data()[17], sizeof(std::uint64_t));
    EXPECT_EQ(s, 3);
    EXPECT_EQ(b.data()[25], 'F');
    EXPECT_EQ(b.data()[26], 'o');
    EXPECT_EQ(b.data()[27], 'o');

    memcpy(&s, &b.data()[28], sizeof(std::uint64_t));
    EXPECT_EQ(s, 3);
    EXPECT_EQ(b.data()[36], 1);
    EXPECT_EQ(b.data()[37], 2);
    EXPECT_EQ(b.data()[38], 3);

    memcpy(&s, &b.data()[39], sizeof(std::uint64_t));
    EXPECT_EQ(s, 3);
    EXPECT_EQ(b.data()[47], 1);
    EXPECT_EQ(b.data()[48], 2);
    EXPECT_EQ(b.data()[49], 3);
}

TEST(ByteBuffer, EmptyString) {
    inputByteBuffer b;
    b.put(std::string(""));
    EXPECT_EQ(b.size(), 8);
    std::uint64_t s;
    memcpy(&s, &b.data()[0], sizeof(std::uint64_t));
    EXPECT_EQ(s, 0);
}

TEST(ByteBuffer, OutputByteBuffer) {
    nixl::outputByteBuffer b(10);
    EXPECT_EQ(b.capacity(), 10);
    EXPECT_EQ(b.size(), 0);
    EXPECT_NE(b.data(), nullptr);

    b.setSize(5);
    EXPECT_EQ(b.size(), 5);
}

} // namespace nixl
