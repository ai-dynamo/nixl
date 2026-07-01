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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common.h"
#include "tracing/trace.h"
#include "tracing/trace_selection.h"

namespace {

// Records the calls a single backend receives, so tests can assert that one
// call site fanned out to every backend.
struct CallLog {
    int spansBegun = 0;
    int spansEnded = 0;
    int marks = 0;
    std::vector<std::pair<std::string, std::string>> strAttrs;
    std::vector<std::pair<std::string, std::int64_t>> intAttrs;
    std::vector<std::pair<std::string, double>> dblAttrs;
};

class MockSpan final : public nixl::trace::SpanBackend {
public:
    MockSpan(CallLog *log, std::uint64_t id) : log_(log), id_(id) {}

    ~MockSpan() override {
        ++log_->spansEnded;
    }

    void
    addAttribute(std::string_view key, std::string_view value) override {
        log_->strAttrs.emplace_back(std::string(key), std::string(value));
    }

    void
    addAttribute(std::string_view key, std::int64_t value) override {
        log_->intAttrs.emplace_back(std::string(key), value);
    }

    void
    addAttribute(std::string_view key, double value) override {
        log_->dblAttrs.emplace_back(std::string(key), value);
    }

    void
    addCtrlDep(nixl::trace::SpanId) override {}

    void
    addDataDep(nixl::trace::SpanId) override {}

    [[nodiscard]] nixl::trace::SpanId
    id() const noexcept override {
        return {id_};
    }

private:
    CallLog *log_;
    std::uint64_t id_;
};

class MockBackend final : public nixl::trace::TraceBackend {
public:
    MockBackend(std::string name, CallLog *log, std::uint64_t span_id)
        : name_(std::move(name)),
          log_(log),
          spanId_(span_id) {}

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(std::string_view, nixl::trace::Kind) override {
        ++log_->spansBegun;
        return std::make_unique<MockSpan>(log_, spanId_);
    }

    void
    mark(std::string_view, nixl::trace::Kind) override {
        ++log_->marks;
    }

    void
    pushCorrelationId(std::uint64_t) override {}

    void
    popCorrelationId() override {}

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return name_;
    }

private:
    std::string name_;
    CallLog *log_;
    std::uint64_t spanId_;
};

[[nodiscard]] std::unique_ptr<nixl::trace::Tracer>
makeMockTracer(CallLog &a, CallLog &b, std::uint64_t id_a = 0, std::uint64_t id_b = 0) {
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<MockBackend>("a", &a, id_a));
    backends.push_back(std::make_unique<MockBackend>("b", &b, id_b));
    return std::make_unique<nixl::trace::Tracer>(std::move(backends));
}

} // namespace

// A single beginSpan() must fan out to every enabled backend, and the Span's
// destructor must end all of them. This is the core requirement.
TEST(Tracing, FanOutToAllBackends) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    EXPECT_FALSE(tracer->empty());

    {
        auto span = tracer->beginSpan("op", nixl::trace::Kind::CommSend);
        EXPECT_TRUE(span.active());
        span.addAttribute("key", std::string_view{"val"});
    } // span ends here

    EXPECT_EQ(a.spansBegun, 1);
    EXPECT_EQ(b.spansBegun, 1);
    EXPECT_EQ(a.spansEnded, 1);
    EXPECT_EQ(b.spansEnded, 1);

    ASSERT_EQ(a.strAttrs.size(), 1u);
    ASSERT_EQ(b.strAttrs.size(), 1u);
    EXPECT_EQ(a.strAttrs[0].first, "key");
    EXPECT_EQ(a.strAttrs[0].second, "val");
    EXPECT_EQ(b.strAttrs[0].first, "key");
    EXPECT_EQ(b.strAttrs[0].second, "val");
}

// Typed attribute overloads must reach every backend with the correct type.
TEST(Tracing, TypedAttributesPropagate) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);

    {
        auto span = tracer->beginSpan("op");
        span.addAttribute("bytes", std::int64_t{4096});
        span.addAttribute("ratio", 0.5);
        span.addAttribute("peer", std::string_view{"agent_1"});
    }

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->intAttrs.size(), 1u);
        EXPECT_EQ(log->intAttrs[0].first, "bytes");
        EXPECT_EQ(log->intAttrs[0].second, 4096);
        ASSERT_EQ(log->dblAttrs.size(), 1u);
        EXPECT_EQ(log->dblAttrs[0].first, "ratio");
        EXPECT_DOUBLE_EQ(log->dblAttrs[0].second, 0.5);
        ASSERT_EQ(log->strAttrs.size(), 1u);
        EXPECT_EQ(log->strAttrs[0].second, "agent_1");
    }
}

// mark() must fan out to every backend.
TEST(Tracing, MarkFanOut) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    tracer->mark("connected");
    tracer->mark("done");
    EXPECT_EQ(a.marks, 2);
    EXPECT_EQ(b.marks, 2);
}

// Span::id() reports the first non-zero backend id, independent of ordering:
// a backend with no id (e.g. NVTX, id {0}) must not mask a later backend's id.
TEST(Tracing, SpanIdFromFirstNonZeroBackend) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b, /*id_a=*/42, /*id_b=*/99);
    auto span = tracer->beginSpan("op");
    EXPECT_EQ(span.id().value, 42u);

    CallLog c, d;
    auto tracer2 = makeMockTracer(c, d, /*id_a=*/0, /*id_b=*/99);
    auto span2 = tracer2->beginSpan("op");
    EXPECT_EQ(span2.id().value, 99u);
}

// A tracer with no backends is empty; spans are inactive and all calls are
// safe no-ops (this is the zero-overhead, nothing-requested path).
TEST(Tracing, EmptyTracerIsInert) {
    nixl::trace::Tracer tracer{std::vector<std::unique_ptr<nixl::trace::TraceBackend>>{}};
    EXPECT_TRUE(tracer.empty());

    auto span = tracer.beginSpan("op", nixl::trace::Kind::CommRecv);
    EXPECT_FALSE(span.active());
    span.addAttribute("k", std::string_view{"v"});
    span.addAttribute("n", std::int64_t{1});
    span.addCtrlDep(nixl::trace::SpanId{7});
    EXPECT_EQ(span.id().value, 0u);
    tracer.mark("m"); // no crash
}

// A default-constructed Span (what the macros bind when the tracer is null) is
// inert.
TEST(Tracing, DefaultSpanIsInert) {
    nixl::trace::Span span;
    EXPECT_FALSE(span.active());
    span.addAttribute("k", std::string_view{"v"});
    EXPECT_EQ(span.id().value, 0u);
}

// NVTX auto-enable precedence (NIX-1576). resolveTraceBackends is the pure
// policy behind makeAgentTracer: an explicit NIXL_TRACE_BACKENDS list is always
// honored, nsys auto-enables NVTX *in addition to* it, and a set-but-empty value
// forces tracing off.
TEST(Tracing, ResolveBackendsExplicitListIsHonored) {
    using nixl::trace::resolveTraceBackends;

    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"nvtx"}, /*under_nsys=*/false),
              (std::vector<std::string>{"nvtx"}));
    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"nvtx,chakra"}, false),
              (std::vector<std::string>{"nvtx", "chakra"}));
    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"chakra"}, false),
              (std::vector<std::string>{"chakra"}));
}

// nsys auto-enable must not suppress an explicitly requested backend: it adds
// NVTX to the explicit list (and dedups when NVTX is already requested).
TEST(Tracing, ResolveBackendsNsysAddsNvtxToExplicitList) {
    using nixl::trace::resolveTraceBackends;

    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"chakra"}, /*under_nsys=*/true),
              (std::vector<std::string>{"chakra", "nvtx"}));
    // Already-requested NVTX is not loaded twice.
    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"nvtx"}, /*under_nsys=*/true),
              (std::vector<std::string>{"nvtx"}));
    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"nvtx,chakra"}, /*under_nsys=*/true),
              (std::vector<std::string>{"nvtx", "chakra"}));
}

TEST(Tracing, ResolveBackendsExplicitEmptyIsOff) {
    using nixl::trace::resolveTraceBackends;

    // Set-but-empty is an explicit "off" that must beat the nsys auto-enable.
    EXPECT_TRUE(resolveTraceBackends(std::optional<std::string>{""}, /*under_nsys=*/true).empty());
    EXPECT_TRUE(resolveTraceBackends(std::optional<std::string>{""}, /*under_nsys=*/false).empty());
    // All-whitespace entries are dropped, so this is still an explicit "off".
    EXPECT_TRUE(resolveTraceBackends(std::optional<std::string>{" , "}, /*under_nsys=*/true).empty());
}

// Entries are trimmed, so surrounding whitespace does not create bogus backend
// names and does not defeat the nsys dedup ("chakra, nvtx" under nsys stays two).
TEST(Tracing, ResolveBackendsTrimsWhitespace) {
    using nixl::trace::resolveTraceBackends;

    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{" nvtx , chakra "}, /*under_nsys=*/false),
              (std::vector<std::string>{"nvtx", "chakra"}));
    // Trimmed "nvtx" is recognized by the dedup, so nsys does not add it again.
    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"chakra, nvtx"}, /*under_nsys=*/true),
              (std::vector<std::string>{"chakra", "nvtx"}));
    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"  nvtx  "}, /*under_nsys=*/true),
              (std::vector<std::string>{"nvtx"}));
}

TEST(Tracing, ResolveBackendsAutoEnablesNvtxUnderNsys) {
    using nixl::trace::resolveTraceBackends;

    // Unset + under nsys -> auto-enable NVTX; unset + not under nsys -> inert.
    EXPECT_EQ(resolveTraceBackends(std::nullopt, /*under_nsys=*/true),
              (std::vector<std::string>{"nvtx"}));
    EXPECT_TRUE(resolveTraceBackends(std::nullopt, /*under_nsys=*/false).empty());
}

// runningUnderNsys keys off NVTX_INJECTION64_PATH, the variable nsys injects
// into the profiled process's environment.
TEST(Tracing, RunningUnderNsysDetectsInjectionVar) {
    gtest::ScopedEnv env;
    env.addVar("NVTX_INJECTION64_PATH", "/opt/nvidia/nsight/libInjectionNvtx64.so");
    EXPECT_TRUE(nixl::trace::runningUnderNsys());
}

// makeTracer ignores empty entries and returns null when no backend resolves to
// a loadable plugin (so callers can cheaply null-check).
TEST(Tracing, MakeTracerUnknownBackendReturnsNull) {
    {
        // No plugin exists for this backend, so it resolves to null with a warning.
        const gtest::LogIgnoreGuard ignore("requested but plugin");
        const auto tracer = nixl::trace::makeTracer({"agent", {"definitely-not-a-backend"}});
        EXPECT_EQ(tracer, nullptr);
    }

    const auto none = nixl::trace::makeTracer({"agent", {}});
    EXPECT_EQ(none, nullptr);
}

// The no-backend runtime path: a backend is requested but its plugin is not
// available (no libtrace_backend_*.so is registered in this unit binary), so
// makeTracer yields a null tracer. Call sites then fall back to a
// default-constructed Span -- the stub path that does essentially nothing -- and
// it must stay safe. NVTX's functional coverage lives in the e2e transfer tests,
// which load the real plugin.
TEST(Tracing, RequestedBackendWithoutPluginIsInert) {
    {
        const gtest::LogIgnoreGuard ignore("requested but plugin");
        const auto tracer = nixl::trace::makeTracer({"stub_agent", {"nvtx"}});
        EXPECT_EQ(tracer, nullptr);
    }

    // What every traced call site binds to when the tracer is null.
    nixl::trace::Span span;
    EXPECT_FALSE(span.active());
    span.addAttribute("bytes", std::int64_t{1024});
    span.addAttribute("peer", std::string_view{"peer_agent"});
    EXPECT_EQ(span.id().value, 0u);
}
