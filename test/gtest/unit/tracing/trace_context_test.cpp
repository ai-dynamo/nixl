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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "tracing/trace_context.h"

constexpr char kCanonicalTraceparent[] = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

namespace {

constexpr std::array<std::uint8_t, nixl::trace::traceContextWireSize> kCanonicalWireRecord{
    0x01, 0x01, 0x4b, 0xf9, 0x2f, 0x35, 0x77, 0xb3, 0x4d, 0xa6, 0xa3, 0xce, 0x92,
    0x9d, 0x0e, 0x0e, 0x47, 0x36, 0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7};

[[nodiscard]] nixl::trace::TraceContext
canonicalContext() {
    const auto context = nixl::trace::parseTraceparent(kCanonicalTraceparent);
    return context.value();
}

// Fills the trailing bytes the sampling decision is derived from, keeping the
// trace id non-zero (hence valid) through a leading byte those bytes exclude.
[[nodiscard]] nixl::trace::TraceContext
contextWithTraceIdRandomness(std::uint8_t randomness_byte) {
    nixl::trace::TraceContext context;
    context.traceId.fill(randomness_byte);
    context.traceId[0] = 0x01;
    context.spanId.fill(0x01);
    context.flags = 0x02;
    return context;
}

// ScopedEnv can only set a variable, so the genuinely-unset path needs its own
// guard; the ambient value is restored either way.
class ScopedUnsetEnv {
public:
    explicit ScopedUnsetEnv(std::string name) : name_(std::move(name)) {
        if (const char *value = std::getenv(name_.c_str()); value != nullptr) {
            previous_ = value;
        }
        ::unsetenv(name_.c_str());
    }

    ~ScopedUnsetEnv() {
        if (previous_) {
            ::setenv(name_.c_str(), previous_->c_str(), 1);
        }
    }

    ScopedUnsetEnv(const ScopedUnsetEnv &) = delete;
    ScopedUnsetEnv &
    operator=(const ScopedUnsetEnv &) = delete;

private:
    std::string name_;
    std::optional<std::string> previous_;
};

} // namespace

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
    nixl::trace::TraceContext expected;
    expected.traceId = {0x4b,
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
                        0x36};
    expected.spanId = {0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7};
    expected.flags = 0x03;

    const auto parsed = nixl::trace::parseTraceparent(nixl::trace::formatTraceparent(expected));

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->traceId, expected.traceId);
    EXPECT_EQ(parsed->spanId, expected.spanId);
    EXPECT_EQ(parsed->flags, expected.flags);
}

TEST(TraceContext, InvalidContextHasNoTextRepresentation) {
    EXPECT_TRUE(nixl::trace::formatTraceparent({}).empty());
}

TEST(TraceContext, ProjectsSpanIdBigEndian) {
    const auto context = nixl::trace::parseTraceparent(kCanonicalTraceparent);

    ASSERT_TRUE(context.has_value());
    EXPECT_EQ(context->correlationId64(), 0x00f067aa0ba902b7ULL);
}

TEST(TraceContext, InvalidContextsProjectZero) {
    EXPECT_EQ(nixl::trace::TraceContext{}.correlationId64(), 0ULL);

    nixl::trace::TraceContext no_span;
    no_span.traceId = {0x4b, 0xf9, 0x2f, 0x35, 0x77, 0xb3, 0x4d, 0xa6};
    ASSERT_FALSE(no_span.valid());
    EXPECT_EQ(no_span.correlationId64(), 0ULL);
}

TEST(TraceContext, NullTracerYieldsInertContext) {
    const nixl::trace::TraceContext context{nullptr};

    EXPECT_FALSE(context.valid());
    EXPECT_EQ(context.correlationId64(), 0ULL);
}

TEST(TraceContext, ValidSpanIdProjectsNonzero) {
    nixl::trace::TraceContext context;
    context.traceId[15] = 0x01;
    context.spanId[7] = 0x01;

    ASSERT_TRUE(context.valid());
    EXPECT_EQ(context.correlationId64(), 0x01ULL);
}

TEST(TraceContext, ProjectionFollowsSpanIdAndIsStable) {
    auto context = nixl::trace::parseTraceparent(kCanonicalTraceparent);
    ASSERT_TRUE(context.has_value());
    const auto baseline = context->correlationId64();

    EXPECT_EQ(context->correlationId64(), baseline);

    auto trace_changed = *context;
    trace_changed.traceId[0] ^= 0xFF;
    trace_changed.traceId[15] ^= 0xFF;
    EXPECT_EQ(trace_changed.correlationId64(), baseline);

    auto span_changed = *context;
    span_changed.spanId[7] ^= 0xFF;
    EXPECT_NE(span_changed.correlationId64(), baseline);
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

// Pins the byte layout: version first, then flags, then trace id and span id in
// big-endian order. A dropped version byte, a reversed id, or a shifted offset
// all change these bytes.
TEST(TraceContext, EncodesCanonicalContextToFixedBytes) {
    std::array<std::uint8_t, nixl::trace::traceContextWireSize> buffer{};

    ASSERT_EQ(nixl::trace::encodeTraceContext(canonicalContext(), buffer),
              nixl::trace::traceContextWireSize);
    EXPECT_EQ(buffer, kCanonicalWireRecord);
}

TEST(TraceContext, EncodeLeavesTrailingBytesUntouched) {
    constexpr std::uint8_t canary = 0xAA;
    std::array<std::uint8_t, nixl::trace::traceContextWireSize + 6> buffer{};
    buffer.fill(canary);

    const auto written = nixl::trace::encodeTraceContext(canonicalContext(), buffer);
    ASSERT_EQ(written, nixl::trace::traceContextWireSize);

    for (std::size_t index = *written; index < buffer.size(); ++index) {
        EXPECT_EQ(buffer[index], canary) << "index " << index;
    }
}

// Sentinel-filled rather than zeroed, so an implementation that clears the
// buffer before rejecting cannot pass.
TEST(TraceContext, RefusesToEncodeInvalidContextOrShortBuffer) {
    std::array<std::uint8_t, nixl::trace::traceContextWireSize> buffer{};
    buffer.fill(0x5a);
    const auto untouched = buffer;
    EXPECT_EQ(nixl::trace::encodeTraceContext(nixl::trace::TraceContext{}, buffer), std::nullopt);
    EXPECT_EQ(buffer, untouched);

    std::array<std::uint8_t, nixl::trace::traceContextWireSize - 1> short_buffer{};
    short_buffer.fill(0xa5);
    const auto short_untouched = short_buffer;
    EXPECT_EQ(nixl::trace::encodeTraceContext(canonicalContext(), short_buffer), std::nullopt);
    EXPECT_EQ(short_buffer, short_untouched);
}

TEST(TraceContext, RoundTripsThroughWireRecord) {
    nixl::trace::TraceContext minimal;
    minimal.traceId[15] = 0x01;
    minimal.spanId[7] = 0x01;
    minimal.flags = 0x03;

    nixl::trace::TraceContext maximal;
    maximal.traceId.fill(0xff);
    maximal.spanId.fill(0xff);
    maximal.flags = 0x01;

    for (const auto &expected :
         {canonicalContext(), nixl::trace::generateTraceContext(), minimal, maximal}) {
        std::array<std::uint8_t, nixl::trace::traceContextWireSize> buffer{};
        ASSERT_TRUE(nixl::trace::encodeTraceContext(expected, buffer).has_value());

        nixl::trace::TraceContext decoded;
        ASSERT_EQ(nixl::trace::decodeTraceContext(buffer, decoded),
                  nixl::trace::WireDecodeResult::Ok);
        EXPECT_EQ(decoded.traceId, expected.traceId);
        EXPECT_EQ(decoded.spanId, expected.spanId);
        EXPECT_EQ(decoded.flags, expected.flags);
    }
}

TEST(TraceContext, WireAndTextFormsAgree) {
    auto expected = nixl::trace::generateTraceContext();
    expected.flags |= 0x01;

    std::array<std::uint8_t, nixl::trace::traceContextWireSize> buffer{};
    ASSERT_TRUE(nixl::trace::encodeTraceContext(expected, buffer).has_value());
    nixl::trace::TraceContext from_wire;
    ASSERT_EQ(nixl::trace::decodeTraceContext(buffer, from_wire),
              nixl::trace::WireDecodeResult::Ok);

    const auto from_text = nixl::trace::parseTraceparent(nixl::trace::formatTraceparent(expected));

    ASSERT_TRUE(from_text.has_value());
    EXPECT_EQ(from_wire.traceId, from_text->traceId);
    EXPECT_EQ(from_wire.spanId, from_text->spanId);
    EXPECT_EQ(from_wire.flags, from_text->flags);
}

// A later peer's record must be skippable without the caller losing the context
// it already had, which is what an append-only carrier relies on.
TEST(TraceContext, SkipsUnknownVersionWithoutTouchingContext) {
    auto buffer = kCanonicalWireRecord;
    buffer[0] = nixl::trace::traceContextWireVersion + 1;

    auto context = nixl::trace::generateTraceContext();
    const auto untouched = context;

    EXPECT_EQ(nixl::trace::decodeTraceContext(buffer, context),
              nixl::trace::WireDecodeResult::UnknownVersion);
    EXPECT_EQ(context.traceId, untouched.traceId);
    EXPECT_EQ(context.spanId, untouched.spanId);
    EXPECT_EQ(context.flags, untouched.flags);
}

TEST(TraceContext, RejectsTruncatedRecords) {
    for (std::size_t length = 0; length < nixl::trace::traceContextWireSize; ++length) {
        nixl::trace::TraceContext decoded;
        const std::span<const std::uint8_t> truncated{kCanonicalWireRecord.data(), length};

        EXPECT_EQ(nixl::trace::decodeTraceContext(truncated, decoded),
                  nixl::trace::WireDecodeResult::Malformed)
            << "length " << length;
    }
}

TEST(TraceContext, RejectsOversizedRecord) {
    std::vector<std::uint8_t> buffer(kCanonicalWireRecord.begin(), kCanonicalWireRecord.end());
    buffer.push_back(0x00);

    nixl::trace::TraceContext decoded;
    EXPECT_EQ(nixl::trace::decodeTraceContext(buffer, decoded),
              nixl::trace::WireDecodeResult::Malformed);
}

TEST(TraceContext, RejectsZeroIdsOnDecode) {
    auto zero_trace = kCanonicalWireRecord;
    std::fill(zero_trace.begin() + 2, zero_trace.begin() + 18, 0x00);

    auto zero_span = kCanonicalWireRecord;
    std::fill(zero_span.begin() + 18, zero_span.end(), 0x00);

    for (const auto &buffer : {zero_trace, zero_span}) {
        nixl::trace::TraceContext decoded;
        EXPECT_EQ(nixl::trace::decodeTraceContext(buffer, decoded),
                  nixl::trace::WireDecodeResult::Malformed);
    }
}

TEST(TraceContext, DropsReservedFlagBitsAcrossWireRoundTrip) {
    auto context = canonicalContext();
    context.flags = 0xff;

    std::array<std::uint8_t, nixl::trace::traceContextWireSize> buffer{};
    ASSERT_TRUE(nixl::trace::encodeTraceContext(context, buffer).has_value());
    EXPECT_EQ(buffer[1], 0x03);

    nixl::trace::TraceContext decoded;
    ASSERT_EQ(nixl::trace::decodeTraceContext(buffer, decoded), nixl::trace::WireDecodeResult::Ok);
    EXPECT_EQ(decoded.flags, 0x03);
}

// A peer's record reaches the decoder without passing through the encoder, so
// its reserved bits are only dropped if decoding masks them itself.
TEST(TraceContext, NormalizesReservedFlagBitsFromRawRecord) {
    auto buffer = kCanonicalWireRecord;
    buffer[1] = 0xff;

    nixl::trace::TraceContext decoded;
    ASSERT_EQ(nixl::trace::decodeTraceContext(buffer, decoded), nixl::trace::WireDecodeResult::Ok);
    EXPECT_EQ(decoded.flags, 0x03);
    EXPECT_TRUE(decoded.sampled());
}

TEST(TraceContext, SampleRatioBoundsDecideEverythingOrNothing) {
    for (std::size_t index = 0; index < 32; ++index) {
        const auto context = nixl::trace::generateTraceContext();
        EXPECT_FALSE(nixl::trace::sampledByRatio(context, 0.0));
        EXPECT_TRUE(nixl::trace::sampledByRatio(context, 1.0));
    }
}

// The kept half is the top one, so the highest randomness survives the smallest
// ratio and a downstream sampler keeps a subset rather than a disjoint set.
TEST(TraceContext, FractionalRatioKeepsHighestRandomness) {
    EXPECT_FALSE(nixl::trace::sampledByRatio(contextWithTraceIdRandomness(0x00), 0.5));
    EXPECT_TRUE(nixl::trace::sampledByRatio(contextWithTraceIdRandomness(0xff), 0.5));

    const auto maximal = contextWithTraceIdRandomness(0xff);
    EXPECT_TRUE(nixl::trace::sampledByRatio(maximal, 0.5));
    EXPECT_TRUE(nixl::trace::sampledByRatio(maximal, 0.001));
}

// Ratios below 2^-53 are where computing the threshold as (1 - ratio) * 2^56 in
// floating point collapses to "never", so the smallest usable ones are pinned.
TEST(TraceContext, RatioBelowDoublePrecisionStillSamplesHighestRandomness) {
    EXPECT_TRUE(
        nixl::trace::sampledByRatio(contextWithTraceIdRandomness(0xff), std::ldexp(1.0, -54)));
}

// What a low ratio keeps, a higher one keeps too: the property that lets a
// second stage thin an already-sampled stream instead of emptying it.
TEST(TraceContext, LowerRatioKeepsSubsetOfHigherRatio) {
    for (std::size_t index = 0; index < 256; ++index) {
        const auto context = nixl::trace::generateTraceContext();
        if (!nixl::trace::sampledByRatio(context, 0.25)) {
            continue;
        }
        EXPECT_TRUE(nixl::trace::sampledByRatio(context, 0.5));
        EXPECT_TRUE(nixl::trace::sampledByRatio(context, 1.0));
    }
}

// The random flag generated contexts set covers only the trailing 7 bytes, so
// a peer's non-random leading prefix must not move the decision.
TEST(TraceContext, SampleDecisionIgnoresLeadingTraceIdBytes) {
    const auto context = contextWithTraceIdRandomness(0xff);
    auto prefixed = context;
    std::fill(prefixed.traceId.begin(), prefixed.traceId.end() - 7, 0x00);

    ASSERT_TRUE(prefixed.valid());
    EXPECT_EQ(nixl::trace::sampledByRatio(prefixed, 0.5),
              nixl::trace::sampledByRatio(context, 0.5));
    EXPECT_TRUE(nixl::trace::sampledByRatio(prefixed, 0.5));
}

TEST(TraceContext, FractionalRatioProducesBothOutcomes) {
    bool sampled_seen = false;
    bool unsampled_seen = false;

    for (std::size_t index = 0; index < 512 && !(sampled_seen && unsampled_seen); ++index) {
        if (nixl::trace::generateTraceContext(0.5).sampled()) {
            sampled_seen = true;
        } else {
            unsampled_seen = true;
        }
    }

    EXPECT_TRUE(sampled_seen);
    EXPECT_TRUE(unsampled_seen);
}

TEST(TraceContext, SampleDecisionIsStableAndIgnoresSpanId) {
    const auto context = canonicalContext();
    const bool decision = nixl::trace::sampledByRatio(context, 0.5);

    EXPECT_EQ(nixl::trace::sampledByRatio(context, 0.5), decision);

    auto span_changed = context;
    span_changed.spanId[0] ^= 0xff;
    EXPECT_EQ(nixl::trace::sampledByRatio(span_changed, 0.5), decision);
}

TEST(TraceContext, InvalidContextIsNeverSampled) {
    EXPECT_FALSE(nixl::trace::sampledByRatio(nixl::trace::TraceContext{}, 1.0));
}

TEST(TraceContext, GeneratedContextCarriesSampledFlag) {
    const auto sampled = nixl::trace::generateTraceContext(1.0);
    EXPECT_EQ(sampled.flags, 0x03);
    EXPECT_TRUE(sampled.sampled());

    const auto unsampled = nixl::trace::generateTraceContext();
    EXPECT_EQ(unsampled.flags, 0x02);
    EXPECT_FALSE(unsampled.sampled());
}

TEST(TraceContext, SampleRatioIsOffWhenUnset) {
    const ScopedUnsetEnv no_nixl_ratio{std::string(nixl::trace::traceSampleRatioVar)};
    const ScopedUnsetEnv no_otel_ratio{std::string(nixl::trace::otelTracesSampleRatioVar)};

    EXPECT_EQ(nixl::trace::resolveTraceSampleRatio(), 0.0);
}

TEST(TraceContext, SampleRatioParsesOwnVariable) {
    const ScopedUnsetEnv no_otel_ratio{std::string(nixl::trace::otelTracesSampleRatioVar)};
    gtest::ScopedEnv env;

    env.addVar(std::string(nixl::trace::traceSampleRatioVar), "0.25");
    EXPECT_EQ(nixl::trace::resolveTraceSampleRatio(), 0.25);
    env.popVar();

    env.addVar(std::string(nixl::trace::traceSampleRatioVar), "1");
    EXPECT_EQ(nixl::trace::resolveTraceSampleRatio(), 1.0);
    env.popVar();

    env.addVar(std::string(nixl::trace::traceSampleRatioVar), "0");
    EXPECT_EQ(nixl::trace::resolveTraceSampleRatio(), 0.0);
}

TEST(TraceContext, SampleRatioRejectsOwnInvalidVariable) {
    const ScopedUnsetEnv no_otel_ratio{std::string(nixl::trace::otelTracesSampleRatioVar)};
    gtest::ScopedEnv env;

    for (const char *value : {"abc", "1.5", "-0.1", "nan", "0.5x"}) {
        env.addVar(std::string(nixl::trace::traceSampleRatioVar), value);
        EXPECT_THROW(static_cast<void>(nixl::trace::resolveTraceSampleRatio()),
                     std::invalid_argument)
            << "value " << value;
        env.popVar();
    }
}

// An OpenTelemetry deployment (Dynamo) already sets this process-wide with the
// same meaning, so NIXL inherits it instead of requiring a second knob.
TEST(TraceContext, SampleRatioFallsBackToOtelVariable) {
    const ScopedUnsetEnv no_nixl_ratio{std::string(nixl::trace::traceSampleRatioVar)};
    gtest::ScopedEnv env;

    env.addVar(std::string(nixl::trace::otelTracesSampleRatioVar), "0.5");
    EXPECT_EQ(nixl::trace::resolveTraceSampleRatio(), 0.5);
}

TEST(TraceContext, OwnVariableOverridesOtelVariable) {
    gtest::ScopedEnv env;
    env.addVar(std::string(nixl::trace::otelTracesSampleRatioVar), "1");

    env.addVar(std::string(nixl::trace::traceSampleRatioVar), "0.25");
    EXPECT_EQ(nixl::trace::resolveTraceSampleRatio(), 0.25);
    env.popVar();

    env.addVar(std::string(nixl::trace::traceSampleRatioVar), "");
    EXPECT_EQ(nixl::trace::resolveTraceSampleRatio(), 0.0);
}

TEST(TraceContext, InvalidOtelVariableIsIgnoredNotFatal) {
    const ScopedUnsetEnv no_nixl_ratio{std::string(nixl::trace::traceSampleRatioVar)};
    gtest::ScopedEnv env;

    env.addVar(std::string(nixl::trace::otelTracesSampleRatioVar), "not-a-ratio");
    EXPECT_EQ(nixl::trace::resolveTraceSampleRatio(), 0.0);
}
