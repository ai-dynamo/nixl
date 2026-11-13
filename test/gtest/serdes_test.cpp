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
#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <string>
#include <string_view>

#include "common.h"
#include "serdes/serdes.h"

namespace {

const std::string invalid_field_log =
    "^Deserialization of tag .* failed for invalid or incomplete field$";
const std::string invalid_length_log =
    "^Deserialization of tag .* failed for invalid field or data length mismatch$";

// Construct malformed fields without asking the serializer to allocate their declared size.
std::string
makeField(std::string_view tag, size_t length, std::string_view payload) {
    std::string wire = "nixlSerDes|";
    wire.append(tag);
    wire.append(reinterpret_cast<const char *>(&length), sizeof(length));
    wire.append(payload);
    wire.push_back('|');
    return wire;
}

void
expectRejectedField(const std::string &wire, const std::string &tag) {
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(wire), NIXL_SUCCESS);
    const gtest::LogIgnoreGuard field_log(invalid_field_log);
    const gtest::LogIgnoreGuard length_log(invalid_length_log);
    EXPECT_FALSE(reader.peekBufLen(tag).has_value());
    EXPECT_EQ(reader.getBufLen(tag), -1);
    EXPECT_TRUE(reader.getStr(tag).empty());

    ASSERT_EQ(reader.importStr(wire), NIXL_SUCCESS);
    std::array<char, 4> output{'a', 'b', 'c', 'd'};
    const auto original = output;
    EXPECT_EQ(reader.getBuf(tag, output.data(), output.size()), NIXL_ERR_MISMATCH);
    EXPECT_EQ(output, original);
}

TEST(SerDesTest, RoundTripsMixedBinaryFields) {
    const std::string payload("a\0b|c", 5);
    const uint64_t number = 0x123456789abcdef0;
    nixlSerDes writer;
    ASSERT_EQ(writer.addStr("", payload), NIXL_SUCCESS);
    ASSERT_EQ(writer.addBuf("number", &number, sizeof(number)), NIXL_SUCCESS);
    ASSERT_EQ(writer.addStr("last", "value"), NIXL_SUCCESS);

    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(writer.exportStr()), NIXL_SUCCESS);
    EXPECT_EQ(reader.peekBufLen({}), payload.size());
    EXPECT_EQ(reader.peekBufLen(""), payload.size());
    EXPECT_EQ(reader.getBufLen(""), payload.size());
    EXPECT_EQ(reader.getStr(""), payload);

    uint64_t decoded = 0;
    EXPECT_EQ(reader.getBuf("number", &decoded, sizeof(decoded)), NIXL_SUCCESS);
    EXPECT_EQ(decoded, number);
    EXPECT_EQ(reader.getStr("last"), "value");
    EXPECT_FALSE(reader.peekBufLen("last").has_value());
    const gtest::LogIgnoreGuard field_log(invalid_field_log);
    EXPECT_EQ(reader.getBufLen("last"), -1);
    EXPECT_TRUE(reader.getStr("last").empty());
}

TEST(SerDesTest, EmptyFieldsConsumeTheirTrailingByte) {
    nixlSerDes writer;
    ASSERT_EQ(writer.addStr("msg", ""), NIXL_SUCCESS);
    ASSERT_EQ(writer.addStr("buffer", ""), NIXL_SUCCESS);
    ASSERT_EQ(writer.addStr("last", "value"), NIXL_SUCCESS);

    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(writer.exportStr()), NIXL_SUCCESS);
    EXPECT_EQ(reader.peekBufLen("msg"), 0u);
    EXPECT_EQ(reader.getStr("msg"), "");
    const gtest::LogIgnoreGuard empty_log("^Deserialization of tag buffer has data length zero$");
    EXPECT_EQ(reader.getBufLen("buffer"), 0);
    EXPECT_EQ(reader.getBuf("buffer", nullptr, 0), NIXL_SUCCESS);
    EXPECT_EQ(reader.getStr("last"), "value");
}

TEST(SerDesTest, WrongTagDoesNotConsumeOrWrite) {
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(makeField("field", 4, "data")), NIXL_SUCCESS);
    for (const std::string tag : {"other", "a tag longer than the entire serialized field"}) {
        SCOPED_TRACE(tag);
        const gtest::LogIgnoreGuard field_log(invalid_field_log);
        const gtest::LogIgnoreGuard length_log(invalid_length_log);
        EXPECT_FALSE(reader.peekBufLen(tag).has_value());
        EXPECT_EQ(reader.getBufLen(tag), -1);
        EXPECT_TRUE(reader.getStr(tag).empty());
        std::array<char, 4> output{};
        EXPECT_EQ(reader.getBuf(tag, output.data(), output.size()), NIXL_ERR_MISMATCH);
        EXPECT_EQ(output, (std::array<char, 4>{}));
    }
    EXPECT_EQ(reader.getStr("field"), "data");
}

TEST(SerDesTest, WrongBufferLengthDoesNotConsumeOrWrite) {
    for (const ssize_t length : {-1, 0, 3, 5}) {
        SCOPED_TRACE(length);
        nixlSerDes reader;
        ASSERT_EQ(reader.importStr(makeField("field", 4, "data")), NIXL_SUCCESS);
        std::array<char, 5> output{};
        const gtest::LogIgnoreGuard length_log(invalid_length_log);
        EXPECT_EQ(reader.getBuf("field", output.data(), length), NIXL_ERR_MISMATCH);
        EXPECT_EQ(output, (std::array<char, 5>{}));
        EXPECT_EQ(reader.peekBufLen("field"), 4u);
        EXPECT_EQ(reader.getStr("field"), "data");
    }
}

TEST(SerDesTest, RejectsTruncatedFields) {
    for (const std::string tag : {"", "field"}) {
        const auto complete = makeField(tag, 4, "data");
        for (size_t length = std::string_view("nixlSerDes|").size(); length < complete.size();
             ++length) {
            SCOPED_TRACE(tag);
            SCOPED_TRACE(length);
            expectRejectedField(complete.substr(0, length), tag);
        }
    }
}

TEST(SerDesTest, RejectsOverflowingLengths) {
    const size_t max_length = std::numeric_limits<size_t>::max();
    const size_t max_signed_length = std::numeric_limits<ssize_t>::max();
    for (const size_t length :
         {max_length, max_length - 1, max_signed_length, max_signed_length + 1}) {
        SCOPED_TRACE(length);
        expectRejectedField(makeField("field", length, "data"), "field");
    }
}

TEST(SerDesTest, RejectsNegativeBufferLengthMatchingWireBits) {
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(makeField("field", std::numeric_limits<size_t>::max(), "data")),
              NIXL_SUCCESS);
    std::array<char, 4> output{};
    const gtest::LogIgnoreGuard length_log(invalid_length_log);
    EXPECT_EQ(reader.getBuf("field", output.data(), -1), NIXL_ERR_MISMATCH);
    EXPECT_EQ(output, (std::array<char, 4>{}));
}

TEST(SerDesTest, RejectsOverflowingLengthAfterValidField) {
    nixlSerDes writer;
    ASSERT_EQ(writer.addStr("first", "prefix"), NIXL_SUCCESS);
    const auto malformed = makeField("field", std::numeric_limits<size_t>::max(), "data");
    const auto wire = writer.exportStr() + malformed.substr(std::string_view("nixlSerDes|").size());
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(wire), NIXL_SUCCESS);
    EXPECT_EQ(reader.getStr("first"), "prefix");
    EXPECT_FALSE(reader.peekBufLen("field").has_value());
    const gtest::LogIgnoreGuard field_log(invalid_field_log);
    EXPECT_TRUE(reader.getStr("field").empty());
}

} // namespace
