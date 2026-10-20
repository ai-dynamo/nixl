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
    byteBuffer b;
    b.put(static_cast<std::uint8_t>(1));
    EXPECT_EQ(b.getRef().size(), 1);
    EXPECT_EQ(b.getRef()[0], 1);

    b.put(static_cast<std::int32_t>(2));
    EXPECT_EQ(b.getRef().size(), 5);
    EXPECT_EQ(*reinterpret_cast<std::int32_t *>(&b.getRef()[1]), 2);

    b.put("/" + std::string("Bar"));
    EXPECT_EQ(b.getRef().size(), 17);
    EXPECT_EQ(*reinterpret_cast<std::uint64_t *>(&b.getRef()[5]), 4);
    EXPECT_EQ(b.getRef()[13], '/');
    EXPECT_EQ(b.getRef()[14], 'B');
    EXPECT_EQ(b.getRef()[15], 'a');
    EXPECT_EQ(b.getRef()[16], 'r');

    b.put(std::string("Foo"));
    EXPECT_EQ(b.getRef().size(), 28);
    EXPECT_EQ(*reinterpret_cast<std::uint64_t *>(&b.getRef()[17]), 3);
    EXPECT_EQ(b.getRef()[25], 'F');
    EXPECT_EQ(b.getRef()[26], 'o');
    EXPECT_EQ(b.getRef()[27], 'o');
}

TEST(ByteBuffer, VarArg) {
    byteBuffer b;
    b.put(static_cast<std::uint8_t>(1),
          static_cast<std::int32_t>(2),
          "/" + std::string("Bar"),
          std::string("Foo"));
    EXPECT_EQ(b.getRef().size(), 28);
    EXPECT_EQ(*reinterpret_cast<std::uint8_t *>(&b.getRef()[0]), 1);
    EXPECT_EQ(*reinterpret_cast<std::int32_t *>(&b.getRef()[1]), 2);
    EXPECT_EQ(*reinterpret_cast<std::uint64_t *>(&b.getRef()[5]), 4);
    EXPECT_EQ(b.getRef()[13], '/');
    EXPECT_EQ(b.getRef()[14], 'B');
    EXPECT_EQ(b.getRef()[15], 'a');
    EXPECT_EQ(b.getRef()[16], 'r');
    EXPECT_EQ(*reinterpret_cast<std::uint64_t *>(&b.getRef()[17]), 3);
    EXPECT_EQ(b.getRef()[25], 'F');
    EXPECT_EQ(b.getRef()[26], 'o');
    EXPECT_EQ(b.getRef()[27], 'o');
}

TEST(ByteBuffer, VarArgConstructor) {
    byteBuffer b{static_cast<std::uint8_t>(1),
                 static_cast<std::int32_t>(2),
                 "/" + std::string("Bar"),
                 std::string("Foo")};
    EXPECT_EQ(b.getRef().size(), 28);
    EXPECT_EQ(*reinterpret_cast<std::uint8_t *>(&b.getRef()[0]), 1);
    EXPECT_EQ(*reinterpret_cast<std::int32_t *>(&b.getRef()[1]), 2);
    EXPECT_EQ(*reinterpret_cast<std::uint64_t *>(&b.getRef()[5]), 4);
    EXPECT_EQ(b.getRef()[13], '/');
    EXPECT_EQ(b.getRef()[14], 'B');
    EXPECT_EQ(b.getRef()[15], 'a');
    EXPECT_EQ(b.getRef()[16], 'r');
    EXPECT_EQ(*reinterpret_cast<std::uint64_t *>(&b.getRef()[17]), 3);
    EXPECT_EQ(b.getRef()[25], 'F');
    EXPECT_EQ(b.getRef()[26], 'o');
    EXPECT_EQ(b.getRef()[27], 'o');
}
} // namespace nixl
