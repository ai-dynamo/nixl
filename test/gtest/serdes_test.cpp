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
expectRejectedField(nixlSerDes &reader, const std::string &tag, const char *reason) {
    SCOPED_TRACE(tag);
    EXPECT_FALSE(reader.peekBufLen(tag).has_value());
    const std::string expected = "^Deserialization of tag " + tag + " failed for " + reason + "$";
    const gtest::LogIgnoreGuard log(expected);
    EXPECT_EQ(reader.getBufLen(tag), -1);
    EXPECT_EQ(log.getIgnoredCount(), 1u);
    EXPECT_TRUE(reader.getStr(tag).empty());
    EXPECT_EQ(log.getIgnoredCount(), 2u);

    std::array<char, 4> output{'a', 'b', 'c', 'd'};
    const auto original = output;
    EXPECT_EQ(reader.getBuf(tag, output.data(), output.size()), NIXL_ERR_MISMATCH);
    EXPECT_EQ(output, original);
    EXPECT_EQ(log.getIgnoredCount(), 3u);
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
    expectRejectedField(reader, "last", "incomplete or missing header");
}

TEST(SerDesTest, EmptyFieldsConsumeTheirTrailingByte) {
    nixlSerDes writer;
    ASSERT_EQ(writer.addStr("msg", ""), NIXL_SUCCESS);
    ASSERT_EQ(writer.addStr("empty", ""), NIXL_SUCCESS);
    ASSERT_EQ(writer.addStr("buffer", ""), NIXL_SUCCESS);
    ASSERT_EQ(writer.addStr("last", "value"), NIXL_SUCCESS);

    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(writer.exportStr()), NIXL_SUCCESS);
    EXPECT_EQ(reader.peekBufLen("msg"), 0u);
    EXPECT_EQ(reader.getStr("msg"), "");
    const gtest::LogIgnoreGuard empty_log("^Deserialization of tag empty failed for empty data$");
    EXPECT_EQ(reader.getStr("empty"), "");
    EXPECT_EQ(empty_log.getIgnoredCount(), 1u);
    const gtest::LogIgnoreGuard zero_log("^Deserialization of tag buffer has data length zero$");
    EXPECT_EQ(reader.getBufLen("buffer"), 0);
    EXPECT_EQ(zero_log.getIgnoredCount(), 1u);
    EXPECT_EQ(reader.getBuf("buffer", nullptr, 0), NIXL_SUCCESS);
    EXPECT_EQ(reader.getStr("last"), "value");
}

TEST(SerDesTest, WrongTagDoesNotConsumeOrWrite) {
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(makeField("field", 4, "data")), NIXL_SUCCESS);
    expectRejectedField(reader, "other", "tag mismatch");
    expectRejectedField(
        reader, "a tag longer than the entire serialized field", "incomplete or missing header");
    EXPECT_EQ(reader.getStr("field"), "data");
}

TEST(SerDesTest, WrongBufferLengthDoesNotConsumeOrWrite) {
    for (const ssize_t length : {-1, 0, 3, 5}) {
        SCOPED_TRACE(length);
        nixlSerDes reader;
        ASSERT_EQ(reader.importStr(makeField("field", 4, "data")), NIXL_SUCCESS);
        std::array<char, 5> output{};
        const gtest::LogIgnoreGuard length_log(
            "^Deserialization of tag field failed for data length mismatch$");
        EXPECT_EQ(reader.getBuf("field", output.data(), length), NIXL_ERR_MISMATCH);
        EXPECT_EQ(length_log.getIgnoredCount(), 1u);
        EXPECT_EQ(output, (std::array<char, 5>{}));
        EXPECT_EQ(reader.peekBufLen("field"), 4u);
        EXPECT_EQ(reader.getStr("field"), "data");
    }
}

TEST(SerDesTest, RejectsTruncatedFields) {
    const size_t prefix_size = std::string_view("nixlSerDes|").size();
    for (const std::string tag : {"", "field"}) {
        const auto complete = makeField(tag, 4, "data");
        const size_t header_end = prefix_size + tag.size() + sizeof(size_t);
        for (size_t length = prefix_size; length < complete.size(); ++length) {
            SCOPED_TRACE(length);
            nixlSerDes reader;
            ASSERT_EQ(reader.importStr(complete.substr(0, length)), NIXL_SUCCESS);
            const char *reason =
                length < header_end ? "incomplete or missing header" : "incomplete data";
            expectRejectedField(reader, tag, reason);
        }
    }
}

TEST(SerDesTest, RejectsOverflowingLengths) {
    const size_t max_length = std::numeric_limits<size_t>::max();
    const size_t max_signed_length = std::numeric_limits<ssize_t>::max();
    for (const size_t length :
         {max_length, max_length - 1, max_signed_length, max_signed_length + 1}) {
        SCOPED_TRACE(length);
        nixlSerDes reader;
        ASSERT_EQ(reader.importStr(makeField("field", length, "data")), NIXL_SUCCESS);
        expectRejectedField(reader, "field", "incomplete data");
    }
}

TEST(SerDesTest, RejectsNegativeBufferLengthMatchingWireBits) {
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(makeField("field", std::numeric_limits<size_t>::max(), "data")),
              NIXL_SUCCESS);
    std::array<char, 4> output{};
    const gtest::LogIgnoreGuard field_log(
        "^Deserialization of tag field failed for incomplete data$");
    EXPECT_EQ(reader.getBuf("field", output.data(), -1), NIXL_ERR_MISMATCH);
    EXPECT_EQ(field_log.getIgnoredCount(), 1u);
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
    expectRejectedField(reader, "field", "incomplete data");
}

} // namespace
