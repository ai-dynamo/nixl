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
#include "common.h"
#include "tracing/backend_trace.h"
#include "tracing/trace.h"
#include "tracing/trace_context.h"
#include "tracing/trace_sink.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using record_t = std::vector<std::string>;

constexpr const char *recorder_agent = "recorder_agent";

class traceRecorder : public testing::Test {
protected:
    void
    SetUp() override {
        traceFile = std::filesystem::temp_directory_path() /
            ("nixl_trace_recorder_emitted_" + std::to_string(::getpid()) + ".spans");
        std::filesystem::remove(traceFile);
        env.addVar("NIXL_TEST_TRACE_FILE", traceFile.string());
    }

    void
    TearDown() override {
        std::filesystem::remove(traceFile);
    }

    [[nodiscard]] std::vector<record_t>
    records() const {
        std::vector<record_t> lines;
        std::ifstream file(traceFile);
        for (std::string line; std::getline(file, line);) {
            std::istringstream fields(line);
            lines.emplace_back(std::istream_iterator<std::string>(fields),
                               std::istream_iterator<std::string>());
        }
        return lines;
    }

    gtest::ScopedEnv env;
    std::filesystem::path traceFile;
};

} // namespace

TEST_F(traceRecorder, EmittedSpanRecordsItsTimesAndContext) {
    const auto context = nixl::trace::generateTraceContext(1.0);
    {
        const auto tracer = nixl::trace::makeTracer({recorder_agent, {"recorder"}});
        ASSERT_NE(tracer, nullptr);
        const std::array attrs{
            nixl::trace::Attribute{"nixl.bytes", std::int64_t{4096}},
            nixl::trace::Attribute{"nixl.remote_agent", std::string_view{"peer"}}};
        tracer->emitSpan("nixl::test.span", nixl::trace::Kind::CommSend, context, 100, 250, attrs);
        tracer->emitSpan("nixl::test.untraced",
                         nixl::trace::Kind::Generic,
                         nixl::trace::TraceContext{},
                         7,
                         9,
                         {});
    }

    EXPECT_EQ(
        records(),
        (std::vector<record_t>{{recorder_agent,
                                "nixl::test.span",
                                "correlation=" + std::to_string(context.correlationId64()),
                                "context=" + nixl::trace::formatTraceparent(context),
                                "start_us=100",
                                "end_us=250",
                                "nixl.bytes=4096",
                                "nixl.remote_agent=peer"},
                               {recorder_agent, "nixl::test.untraced", "start_us=7", "end_us=9"}}));
}

TEST_F(traceRecorder, RecordedPhaseIsAnEventAtThePluginTimestamp) {
    const auto context = nixl::trace::generateTraceContext(1.0);
    {
        const auto tracer = nixl::trace::makeTracer({recorder_agent, {"recorder"}});
        ASSERT_NE(tracer, nullptr);
        nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
        const std::array attrs{nixlBackendTraceAttr{"nixl.remote_agent", "sender"}};
        sink.recordPhase(nixl_trace_phase_t::NOTIF_RECEIVED, {}, 1234, attrs, &context);
    }

    const std::string traceparent = nixl::trace::formatTraceparent(context);
    EXPECT_EQ(records(),
              (std::vector<record_t>{{recorder_agent,
                                      "nixl::notif.received",
                                      "correlation=" + std::to_string(context.correlationId64()),
                                      "context=" + traceparent,
                                      "at_us=1234",
                                      "nixl.backend=UCX",
                                      "nixl.phase.timestamp_us=1234",
                                      "nixl.traceparent=" + traceparent,
                                      "nixl.remote_agent=sender"}}));
}
