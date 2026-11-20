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

#include "ucx_split_failure.h"

#include <gtest/gtest.h>

#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common/nixl_time.h"

namespace {

class DeferringSink final : public nixlBackendTraceSink {
public:
    DeferringSink() {
        identity_.context.traceId->fill(0x11);
        identity_.context.spanId->fill(0x22);
        identity_.context.flags = nixl::trace::sampledTraceFlag;
        identity_.requestId = nixl::trace::RequestId{1};
        identity_.execution = 1;
    }

    using nixlBackendTraceSink::recordPhase;

    void
    recordPhase(nixl_trace_phase_t,
                nixlTime::us_t,
                const nixl::trace::TraceIdentity &,
                std::span<const nixlBackendTraceAttr>,
                std::string_view) noexcept override {}

    void
    recordSpan(nixl_trace_span_t,
               nixl_xfer_op_t,
               nixlTime::us_t,
               nixlTime::us_t,
               const nixl::trace::TraceIdentity &,
               std::span<const nixlBackendTraceAttr>,
               std::string_view) noexcept override {}

    [[nodiscard]] bool
    needsTimestamps() const noexcept override {
        return false;
    }

    [[nodiscard]] const nixl::trace::TraceIdentity *
    currentIdentity() const noexcept override {
        return &identity_;
    }

    [[nodiscard]] std::string_view
    currentBaggage() const noexcept override {
        return {};
    }

    void
    writeTraceTags(nixlSerDes &,
                   const nixl::trace::TraceIdentity &,
                   std::string_view) const override {}

    [[nodiscard]] std::optional<nixl::trace::TraceIdentity>
    readTraceTags(nixlSerDes &) const override {
        return std::nullopt;
    }

    [[nodiscard]] std::string
    readTraceBaggage(nixlSerDes &) const override {
        return {};
    }

    void
    deferPhase(const nixl::trace::DeferredPhase &phase) noexcept override {
        deferred.push_back(phase);
    }

    void
    recordDeferredPhases() noexcept override {}

    std::vector<nixl::trace::DeferredPhase> deferred;

private:
    nixl::trace::TraceIdentity identity_;
};

[[nodiscard]] nixlBackendPostPhases
splitPostTrace(DeferringSink &sink) {
    nixlBackendPostPhases trace;
    trace.begin(&sink, NIXL_WRITE, nixl_trace_completion_t::LAST_PART);
    return trace;
}

} // namespace

TEST(UcxSplitFailure, ResetReturnsTheSubmissionToWaiting) {
    DeferringSink sink;
    nixlBackendPostPhases trace = splitPostTrace(sink);
    nixlUcxSplitFailure failure;
    failure.finishSubmission(NIXL_ERR_BACKEND);
    failure.reset();
    EXPECT_FALSE(failure.observe(UCS_ERR_CONNECTION_RESET, trace));
}

TEST(UcxSplitFailure, FirstChunkWaitsWhileSubmittingAndRecordsWhenItSawTheFailure) {
    DeferringSink sink;
    nixlBackendPostPhases trace = splitPostTrace(sink);
    nixlUcxSplitFailure failure;
    failure.reset();

    const nixlTime::us_t before_seen = nixlTime::getUs();
    EXPECT_FALSE(failure.observe(UCS_ERR_CONNECTION_RESET, trace));
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const nixlTime::us_t after_wait = nixlTime::getUs();
    EXPECT_FALSE(failure.observe(UCS_ERR_CONNECTION_RESET, trace));
    EXPECT_TRUE(sink.deferred.empty());

    failure.finishSubmission(NIXL_SUCCESS);
    EXPECT_TRUE(failure.observe(UCS_ERR_CONNECTION_RESET, trace));
    ASSERT_EQ(sink.deferred.size(), 1u);
    EXPECT_EQ(sink.deferred[0].phase, nixl_trace_phase_t::WIRE_COMPLETED);
    EXPECT_EQ(sink.deferred[0].status, NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(sink.deferred[0].phaseAt, 0u);
    EXPECT_GE(sink.deferred[0].completedAt, before_seen);
    EXPECT_LT(sink.deferred[0].completedAt, after_wait);
    EXPECT_FALSE(trace.active());
}

TEST(UcxSplitFailure, FirstChunkDropsTheFailureOfARejectedPost) {
    DeferringSink sink;
    nixlBackendPostPhases trace = splitPostTrace(sink);
    nixlUcxSplitFailure failure;
    failure.reset();

    EXPECT_FALSE(failure.observe(UCS_ERR_CONNECTION_RESET, trace));
    failure.finishSubmission(NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_TRUE(failure.observe(UCS_ERR_CONNECTION_RESET, trace));
    EXPECT_TRUE(sink.deferred.empty());
    EXPECT_FALSE(trace.active());
}

TEST(UcxSplitFailure, ResetForgetsWhenTheLastPostsFailureWasSeen) {
    DeferringSink sink;
    nixlBackendPostPhases trace = splitPostTrace(sink);
    nixlUcxSplitFailure failure;
    failure.reset();
    EXPECT_FALSE(failure.observe(UCS_ERR_CONNECTION_RESET, trace));

    failure.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const nixlTime::us_t after_reset = nixlTime::getUs();
    failure.finishSubmission(NIXL_SUCCESS);
    EXPECT_TRUE(failure.observe(UCS_ERR_CONNECTION_RESET, trace));
    ASSERT_EQ(sink.deferred.size(), 1u);
    EXPECT_GE(sink.deferred[0].completedAt, after_reset);
}

TEST(UcxSplitFailure, TracedChunkRetiresAtOnceWithoutANativeStatus) {
    DeferringSink sink;
    nixlBackendPostPhases trace = splitPostTrace(sink);
    nixlUcxSplitFailure failure;
    failure.reset();
    EXPECT_TRUE(failure.observe(UCS_OK, trace));
    EXPECT_TRUE(sink.deferred.empty());
    EXPECT_TRUE(trace.active());
}
