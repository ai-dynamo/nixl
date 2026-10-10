/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the dependency-free S3-over-RDMA wire-protocol helpers. These
// exercise the parts of the RDMA path that do not require the AWS SDK or the
// cuObjClient library, so they run on any build.

#include <gtest/gtest.h>

#include "object/rdma/rdma_protocol.h"

namespace {

using namespace nixl_obj_rdma;

TEST(RdmaProtocol, ParseReplyRejectsOutOfRangeCodes) {
    // Values outside the HTTP status range are malformed, so a negative reply
    // cannot alias rdma_not_supported (-2) nor a huge value look like success.
    EXPECT_EQ(parseRdmaReply("-2"), 0);
    EXPECT_EQ(parseRdmaReply("99"), 0);
    EXPECT_EQ(parseRdmaReply("600"), 0);
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

TEST(RdmaProtocol, RegistrationPiecesBelowLimitIsOnePiece) {
    const auto pieces = registrationPieces(0x10000, 4096, 8192, 2048);
    ASSERT_EQ(pieces.size(), 1u);
    EXPECT_EQ(pieces[0], std::make_pair(uintptr_t{0x10000}, size_t{4096}));
}

TEST(RdmaProtocol, RegistrationPiecesCoverTheBufferInOrder) {
    // At the limit and above it: consecutive pieces, the last one shorter.
    const auto at_limit = registrationPieces(0x10000, 8192, 8192, 2048);
    ASSERT_EQ(at_limit.size(), 4u);
    const auto pieces = registrationPieces(0x10000, 8192 + 100, 8192, 2048);
    ASSERT_EQ(pieces.size(), 5u);
    uintptr_t next = 0x10000;
    for (const auto &[start, len] : pieces) {
        EXPECT_EQ(start, next);
        next = start + len;
    }
    EXPECT_EQ(next, uintptr_t{0x10000 + 8192 + 100});
    EXPECT_EQ(pieces.back().second, 100u);
}

TEST(RdmaProtocol, RegistrationPiecesRealLimits) {
    // 5 GiB with the cuObject limit (4 GiB) and 2 GiB pieces: 2 + 2 + 1 GiB.
    constexpr size_t gib = size_t{1} << 30;
    const auto pieces = registrationPieces(0, 5 * gib, 4 * gib, 2 * gib);
    ASSERT_EQ(pieces.size(), 3u);
    EXPECT_EQ(pieces[2], std::make_pair(uintptr_t{4 * gib}, gib));
}

TEST(RdmaProtocol, RegistrationPiecesUnusablePieceLengthKeepsOnePiece) {
    EXPECT_EQ(registrationPieces(0x10000, 8192, 8192, 0).size(), 1u);
    EXPECT_EQ(registrationPieces(0x10000, 8192, 8192, 8192).size(), 1u);
}

} // namespace
