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

#include <set>
#include <string>

#include "tracing/trace_context.h"

constexpr char kCanonicalTraceparent[] = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

TEST(TraceContext, ParsesAndFormatsCanonicalTraceparent) {
    const auto context = nixl::trace::parseTraceparent(kCanonicalTraceparent);

    ASSERT_TRUE(context.has_value());
    EXPECT_TRUE(context->valid());
    EXPECT_EQ(nixl::trace::formatTraceparent(*context), kCanonicalTraceparent);
}

TEST(TraceContext, RejectsUppercaseHex) {
    std::string value = kCanonicalTraceparent;
    value[4] = 'B';
    EXPECT_FALSE(nixl::trace::parseTraceparent(value).has_value());

    value = kCanonicalTraceparent;
    value[38] = 'F';
    EXPECT_FALSE(nixl::trace::parseTraceparent(value).has_value());

    value = kCanonicalTraceparent;
    value[53] = 'A';
    EXPECT_FALSE(nixl::trace::parseTraceparent(value).has_value());
}

TEST(TraceContext, RejectsInvalidLengths) {
    const std::string canonical = kCanonicalTraceparent;

    EXPECT_FALSE(nixl::trace::parseTraceparent(canonical.substr(1)).has_value());
    EXPECT_FALSE(nixl::trace::parseTraceparent(canonical + "0").has_value());
}

TEST(TraceContext, RejectsInvalidSeparators) {
    std::string value = kCanonicalTraceparent;
    value[2] = ':';
    EXPECT_FALSE(nixl::trace::parseTraceparent(value).has_value());

    value = kCanonicalTraceparent;
    value[35] = ':';
    EXPECT_FALSE(nixl::trace::parseTraceparent(value).has_value());

    value = kCanonicalTraceparent;
    value[52] = ':';
    EXPECT_FALSE(nixl::trace::parseTraceparent(value).has_value());
}

TEST(TraceContext, RejectsInvalidHex) {
    for (const std::size_t offset : {3u, 36u, 53u}) {
        std::string value = kCanonicalTraceparent;
        value[offset] = 'g';
        EXPECT_FALSE(nixl::trace::parseTraceparent(value).has_value());
    }
}

TEST(TraceContext, RejectsUnsupportedVersion) {
    std::string value = kCanonicalTraceparent;
    value[1] = '1';

    EXPECT_FALSE(nixl::trace::parseTraceparent(value).has_value());
}

TEST(TraceContext, RejectsZeroTraceId) {
    EXPECT_FALSE(
        nixl::trace::parseTraceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01")
            .has_value());
}

TEST(TraceContext, RejectsZeroSpanId) {
    EXPECT_FALSE(
        nixl::trace::parseTraceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01")
            .has_value());
}

TEST(TraceContext, PreservesFlagsAndNormalizesOutput) {
    const auto sampled =
        nixl::trace::parseTraceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-ff");
    const auto unsampled =
        nixl::trace::parseTraceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-fe");

    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(unsampled.has_value());
    EXPECT_EQ(sampled->flags, 0xff);
    EXPECT_TRUE(sampled->sampled());
    EXPECT_EQ(unsampled->flags, 0xfe);
    EXPECT_FALSE(unsampled->sampled());
    EXPECT_EQ(nixl::trace::formatTraceparent(*sampled),
              "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-03");
    EXPECT_EQ(nixl::trace::formatTraceparent(*unsampled),
              "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-02");

    auto with_reserved_flags = *sampled;
    with_reserved_flags.flags = 0xff;
    EXPECT_EQ(nixl::trace::formatTraceparent(with_reserved_flags),
              "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-03");
}

TEST(TraceContext, RoundTripsFixedContextWithoutFieldDrift) {
    const nixl::trace::TraceContext expected{{0x4b,
                                              0xf9,
                                              0x2f,
                                              0x35,
                                              0x77,
                                              0xb3,
                                              0x4d,
                                              0xa6,
                                              0xa3,
                                              0xce,
                                              0x92,
                                              0x9d,
                                              0x0e,
                                              0x0e,
                                              0x47,
                                              0x36},
                                             {0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7},
                                             0x03};

    const auto parsed = nixl::trace::parseTraceparent(nixl::trace::formatTraceparent(expected));

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->traceId, expected.traceId);
    EXPECT_EQ(parsed->spanId, expected.spanId);
    EXPECT_EQ(parsed->flags, expected.flags);
}

TEST(TraceContext, InvalidContextHasNoTextRepresentation) {
    EXPECT_TRUE(nixl::trace::formatTraceparent({}).empty());
}

TEST(TraceContext, ProjectsTraceIdBigEndian) {
    const auto context = nixl::trace::parseTraceparent(kCanonicalTraceparent);

    ASSERT_TRUE(context.has_value());
    EXPECT_EQ(context->correlationId64(), 0x4bf92f3577b34da6ULL);
}

TEST(TraceContext, GeneratesDistinctValidContexts) {
    std::set<std::string> generated;
    constexpr std::size_t count = 64;

    for (std::size_t index = 0; index < count; ++index) {
        const auto context = nixl::trace::generateTraceContext();
        EXPECT_TRUE(context.valid());
        EXPECT_EQ(context.flags, 0x02);
        EXPECT_FALSE(context.sampled());
        generated.insert(nixl::trace::formatTraceparent(context));
    }

    EXPECT_EQ(generated.size(), count);
}
