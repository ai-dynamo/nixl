/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the dependency-free S3-over-RDMA wire-protocol helpers. These
// exercise the parts of the RDMA path that do not require the AWS SDK or the
// cuObjClient library, so they run on any build.

#include <gtest/gtest.h>

#include "object/rdma_protocol.h"

namespace {

using namespace nixl_obj_rdma;

TEST(RdmaProtocol, FormatTokenIsDescColonAddrColonSize) {
    // Wire format: "<descriptor>:<addr:016x>:<size:016x>" (lowercase, zero-padded).
    EXPECT_EQ(formatRdmaToken("DESC", 0x10, 0x200), "DESC:0000000000000010:0000000000000200");
    EXPECT_EQ(formatRdmaToken("", 0, 0), ":0000000000000000:0000000000000000");
}

TEST(RdmaProtocol, FormatTokenHandlesFullWidthAndNullDescriptor) {
    // Full 64-bit values render as 16 lowercase hex digits (no truncation).
    EXPECT_EQ(formatRdmaToken("D", 0xffffffffffffffffULL, 0xdeadbeefcafef00dULL),
              "D:ffffffffffffffff:deadbeefcafef00d");
    // A null descriptor is treated as an empty one (no crash, leading ':').
    EXPECT_EQ(formatRdmaToken(nullptr, 0x1, 0x2), ":0000000000000001:0000000000000002");
}

TEST(RdmaProtocol, FormatTokenDoesNotTruncateLongDescriptor) {
    // The descriptor is opaque and has no fixed length; it must never be cut off
    // (a fixed stack buffer used to truncate it). The suffix is always
    // ":<16 hex>:<16 hex>" = 34 chars.
    const std::string desc(1024, 'a');
    const std::string token = formatRdmaToken(desc.c_str(), 0xabc, 0xdef);
    EXPECT_EQ(token, desc + ":0000000000000abc:0000000000000def");
    EXPECT_EQ(token.size(), desc.size() + 34);
}

TEST(RdmaProtocol, ParseReplySuccessCodes) {
    EXPECT_EQ(parseRdmaReply("200"), rdma_reply_success);
    EXPECT_EQ(parseRdmaReply("204"), rdma_reply_no_content);
    EXPECT_EQ(parseRdmaReply("206"), rdma_reply_partial_content);
}

TEST(RdmaProtocol, ParseReplyDeclinedAndAbsentMapToNotSupported) {
    // Explicit decline.
    EXPECT_EQ(parseRdmaReply("501"), static_cast<int>(rdma_not_supported));
    // An absent header (a non-RDMA server never sets it) reads as "declined".
    // This drives GET decline detection; PUT success is decided separately by
    // HTTP 200 + ETag, not by this header.
    EXPECT_EQ(parseRdmaReply(""), static_cast<int>(rdma_not_supported));
    // Garbage parses to 0 (caller treats as failure).
    EXPECT_EQ(parseRdmaReply("not-a-number"), 0);
}

TEST(RdmaProtocol, ParseReplyRejectsTrailingJunk) {
    // A reply with trailing junk must NOT be accepted as a success code: std::stoi
    // would return 200 for "200xyz" and mask a malformed reply as RDMA success.
    EXPECT_EQ(parseRdmaReply("200xyz"), 0);
    EXPECT_EQ(parseRdmaReply("200 "), 0);
    EXPECT_EQ(parseRdmaReply("2 06"), 0);
    EXPECT_EQ(parseRdmaReply("0x200"), 0);
    // "501" with trailing junk is not the exact decline token, so it is malformed.
    EXPECT_EQ(parseRdmaReply("501x"), 0);
}

TEST(RdmaProtocol, ParseReplyRejectsLeadingWhitespaceAndSign) {
    // from_chars does not skip leading whitespace or accept a leading '+'.
    EXPECT_EQ(parseRdmaReply(" 200"), 0);
    EXPECT_EQ(parseRdmaReply("+200"), 0);
    EXPECT_EQ(parseRdmaReply("\t206"), 0);
}

TEST(RdmaProtocol, ProtocolConstantsHaveExpectedValues) {
    // Guard the wire contract: header names and status codes must not drift.
    EXPECT_STREQ(amz_rdma_token, "x-amz-rdma-token");
    EXPECT_STREQ(amz_rdma_reply, "x-amz-rdma-reply");
    EXPECT_STREQ(amz_rdma_bytes_transferred, "x-amz-rdma-bytes-transferred");
    EXPECT_STREQ(unsigned_payload, "UNSIGNED-PAYLOAD");
    EXPECT_EQ(rdma_reply_success, 200);
    EXPECT_EQ(rdma_reply_no_content, 204);
    EXPECT_EQ(rdma_reply_partial_content, 206);
    EXPECT_EQ(rdma_reply_not_implemented, 501);
    EXPECT_EQ(rdma_not_supported, -2);
    EXPECT_EQ(rdma_error, -1);
}

} // namespace
