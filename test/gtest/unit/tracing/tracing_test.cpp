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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common.h"
#include "tracing/trace.h"

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
    beginSpan(std::string_view, nixl::trace::Kind, nixl::trace::Color) override {
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

std::unique_ptr<nixl::trace::Tracer>
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

// makeTracer ignores empty entries and returns null when nothing resolves to a
// compiled-in backend (so callers can cheaply null-check).
TEST(Tracing, MakeTracerUnknownBackendReturnsNull) {
    {
        // An unknown backend is expected to emit a warning and resolve to null.
        const gtest::LogIgnoreGuard ignore("unknown backend");
        auto tracer = nixl::trace::makeTracer({"agent", {"definitely-not-a-backend"}});
        EXPECT_EQ(tracer, nullptr);
    }

    auto none = nixl::trace::makeTracer({"agent", {}});
    EXPECT_EQ(none, nullptr);
}

// NVTX smoke test: when compiled in, makeTracer({"nvtx"}) yields a working
// tracer; begin/attr/mark/end must not crash with no profiler attached. When
// NVTX is not compiled in, makeTracer returns null and we simply skip.
TEST(Tracing, NvtxBackendSmoke) {
    auto tracer = nixl::trace::makeTracer({"nvtx_smoke_agent", {"nvtx"}});
#ifdef NIXL_TRACE_BACKEND_NVTX
    ASSERT_NE(tracer, nullptr);
    EXPECT_FALSE(tracer->empty());
    {
        auto span = tracer->beginSpan("nixl::test.span", nixl::trace::Kind::CommSend);
        span.addAttribute("bytes", std::int64_t{1024});
        span.addAttribute("peer", std::string_view{"peer_agent"});
        span.addAttribute("ratio", 0.5);
    }
    tracer->mark("nixl::test.mark");
#else
    EXPECT_EQ(tracer, nullptr);
    GTEST_SKIP() << "NVTX backend not compiled in";
#endif
}
