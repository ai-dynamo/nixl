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
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "common.h"
#include "nixl.h"
#include "transfer_request.h"
#include "tracing/trace.h"
#include "tracing/trace_context.h"
#include "tracing/trace_sink.h"

namespace nixl::trace {
// Agent-wiring backend-selection policy, defined in nixl_agent.cpp (not exposed
// via a header — declared here so the unit test can exercise it directly).
[[nodiscard]] bool
runningUnderNsys();
[[nodiscard]] std::vector<std::string>
resolveTraceBackends(const std::optional<std::string> &explicit_spec, bool under_nsys);
} // namespace nixl::trace

namespace {

constexpr const char *kUnloadableBackend = "definitely-not-a-backend";

using attr_value_t = std::variant<std::string, std::int64_t, double>;
using attr_t = std::pair<std::string, attr_value_t>;
using attr_list_t = std::vector<attr_t>;

struct EmittedRecord {
    std::string name;
    nixl::trace::Kind kind;
    nixl::trace::TraceContext context;
    std::vector<nixlTime::us_t> times;
    attr_list_t attrs;
};

// Records the calls a single backend receives, so tests can assert that one
// call site fanned out to every backend.
struct CallLog {
    int spansBegun = 0;
    int spansEnded = 0;
    int marks = 0;
    std::vector<std::string> spanNames;
    std::vector<std::pair<std::string, std::string>> strAttrs;
    std::vector<std::pair<std::string, std::int64_t>> intAttrs;
    std::vector<std::pair<std::string, double>> dblAttrs;
    std::vector<nixl::trace::TraceContext> pushedContexts;
    std::vector<nixl::trace::TraceContext> activeContexts;
    std::vector<nixl::trace::TraceContext> poppedContexts;
    std::vector<EmittedRecord> emittedSpans;
    std::vector<EmittedRecord> emittedEvents;
};

[[nodiscard]] EmittedRecord
makeRecord(std::string_view name,
           nixl::trace::Kind kind,
           const nixl::trace::TraceContext &context,
           std::vector<nixlTime::us_t> times,
           std::span<const nixl::trace::Attribute> attrs) {
    EmittedRecord record{std::string(name), kind, context, std::move(times), {}};
    for (const auto &attr : attrs) {
        record.attrs.emplace_back(
            std::string(attr.key),
            std::visit(
                [](const auto value) -> attr_value_t {
                    if constexpr (std::is_same_v<decltype(value), const std::string_view>) {
                        return std::string(value);
                    } else {
                        return value;
                    }
                },
                attr.value));
    }
    return record;
}

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
    MockBackend(std::string name, CallLog *log, std::uint64_t span_id, bool null_spans = false)
        : name_(std::move(name)),
          log_(log),
          spanId_(span_id),
          nullSpans_(null_spans) {}

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(std::string_view name, nixl::trace::Kind) override {
        ++log_->spansBegun;
        log_->spanNames.emplace_back(name);
        if (nullSpans_) {
            return nullptr;
        }
        return std::make_unique<MockSpan>(log_, spanId_);
    }

    void
    mark(std::string_view, nixl::trace::Kind) override {
        ++log_->marks;
    }

    void
    emitSpan(std::string_view name,
             nixl::trace::Kind kind,
             const nixl::trace::TraceContext &context,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const nixl::trace::Attribute> attrs) override {
        log_->emittedSpans.push_back(makeRecord(name, kind, context, {start, end}, attrs));
    }

    void
    emitEvent(std::string_view name,
              nixl::trace::Kind kind,
              const nixl::trace::TraceContext &context,
              nixlTime::us_t at,
              std::span<const nixl::trace::Attribute> attrs) override {
        log_->emittedEvents.push_back(makeRecord(name, kind, context, {at}, attrs));
    }

    void
    pushContext(const nixl::trace::TraceContext &context) override {
        log_->pushedContexts.push_back(context);
        log_->activeContexts.push_back(context);
    }

    void
    popContext() override {
        ASSERT_FALSE(log_->activeContexts.empty());
        log_->poppedContexts.push_back(log_->activeContexts.back());
        log_->activeContexts.pop_back();
    }

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return name_;
    }

private:
    std::string name_;
    CallLog *log_;
    std::uint64_t spanId_;
    bool nullSpans_;
};

class ThrowingBackend final : public nixl::trace::TraceBackend {
public:
    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(std::string_view, nixl::trace::Kind) override {
        throw std::runtime_error("beginSpan failed");
    }

    void
    mark(std::string_view, nixl::trace::Kind) override {}

    void
    emitSpan(std::string_view,
             nixl::trace::Kind,
             const nixl::trace::TraceContext &,
             nixlTime::us_t,
             nixlTime::us_t,
             std::span<const nixl::trace::Attribute>) override {
        throw std::runtime_error("emitSpan failed");
    }

    void
    emitEvent(std::string_view,
              nixl::trace::Kind,
              const nixl::trace::TraceContext &,
              nixlTime::us_t,
              std::span<const nixl::trace::Attribute>) override {
        throw std::runtime_error("emitEvent failed");
    }

    void
    pushContext(const nixl::trace::TraceContext &) override {}

    void
    popContext() override {}

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return "throwing";
    }
};

void
failRandomBytes(std::uint8_t *, std::size_t) {
    throw std::system_error(ENOSYS, std::generic_category(), "getentropy failed");
}

int zero_draws_left = 0;

void
zeroFirstRandomBytes(std::uint8_t *bytes, std::size_t size) {
    std::fill_n(bytes, size, (zero_draws_left-- > 0) ? 0 : 0x5a);
}

[[nodiscard]] std::unique_ptr<nixl::trace::Tracer>
makeMockTracer(CallLog &a,
               CallLog &b,
               std::uint64_t id_a = 0,
               std::uint64_t id_b = 0,
               double sample_ratio = 0.0,
               nixl::trace::random_bytes_fn_t random_bytes = nullptr) {
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<MockBackend>("a", &a, id_a));
    backends.push_back(std::make_unique<MockBackend>("b", &b, id_b));
    return std::make_unique<nixl::trace::Tracer>(std::move(backends), sample_ratio, random_bytes);
}

[[nodiscard]] std::unique_ptr<nixl::trace::Tracer>
makeThrowingTracer() {
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<ThrowingBackend>());
    return std::make_unique<nixl::trace::Tracer>(std::move(backends), 0.0);
}

[[nodiscard]] nixl::trace::TraceContext
testContext(std::uint8_t id_byte, std::uint8_t flags = 0x01) {
    nixl::trace::TraceContext context;
    context.traceId.fill(id_byte);
    context.spanId.fill(id_byte);
    context.flags = flags;
    return context;
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

TEST(Tracing, EmittedRecordsFanOutToAllBackends) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto context = testContext(0x21, 0x02);
    const nixl::trace::Attribute attrs[] = {
        {"peer", std::string_view{"agent_1"}}, {"bytes", std::int64_t{4096}}, {"ratio", 0.5}};
    const attr_list_t expected{
        {"peer", std::string("agent_1")}, {"bytes", std::int64_t{4096}}, {"ratio", 0.5}};

    tracer->emitSpan("span", nixl::trace::Kind::CommSend, context, 10, 25, attrs);
    tracer->emitEvent("event", nixl::trace::Kind::CommRecv, context, 42, attrs);

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedSpans.size(), 1u);
        EXPECT_EQ(log->emittedSpans[0].name, "span");
        EXPECT_EQ(log->emittedSpans[0].kind, nixl::trace::Kind::CommSend);
        EXPECT_EQ(log->emittedSpans[0].context, context);
        EXPECT_EQ(log->emittedSpans[0].times, (std::vector<nixlTime::us_t>{10, 25}));
        EXPECT_EQ(log->emittedSpans[0].attrs, expected);
        ASSERT_EQ(log->emittedEvents.size(), 1u);
        EXPECT_EQ(log->emittedEvents[0].name, "event");
        EXPECT_EQ(log->emittedEvents[0].kind, nixl::trace::Kind::CommRecv);
        EXPECT_EQ(log->emittedEvents[0].context, context);
        EXPECT_EQ(log->emittedEvents[0].times, std::vector<nixlTime::us_t>{42});
        EXPECT_EQ(log->emittedEvents[0].attrs, expected);
    }
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
    tracer.emitSpan("s", nixl::trace::Kind::Generic, testContext(0x23), 1, 2, {});
    tracer.emitEvent("e", nixl::trace::Kind::Metadata, testContext(0x23), 3, {});
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
// forces tracing off. The result is deduplicated (backends are collected in a
// std::set), so the returned list is sorted.
TEST(Tracing, ResolveBackendsExplicitListIsHonored) {
    using nixl::trace::resolveTraceBackends;

    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"nvtx"}, /*under_nsys=*/false),
              (std::vector<std::string>{"nvtx"}));
    EXPECT_EQ(resolveTraceBackends(std::optional<std::string>{"nvtx,chakra"}, false),
              (std::vector<std::string>{"chakra", "nvtx"}));
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
              (std::vector<std::string>{"chakra", "nvtx"}));
}

TEST(Tracing, ResolveBackendsExplicitEmptyIsOff) {
    using nixl::trace::resolveTraceBackends;

    // Set-but-empty is an explicit "off" that must beat the nsys auto-enable.
    EXPECT_TRUE(resolveTraceBackends(std::optional<std::string>{""}, /*under_nsys=*/true).empty());
    EXPECT_TRUE(resolveTraceBackends(std::optional<std::string>{""}, /*under_nsys=*/false).empty());
    // All-whitespace entries are dropped, so this is still an explicit "off".
    EXPECT_TRUE(
        resolveTraceBackends(std::optional<std::string>{" , "}, /*under_nsys=*/true).empty());
}

// Entries are trimmed, so surrounding whitespace does not create bogus backend
// names and does not defeat the nsys dedup ("chakra, nvtx" under nsys stays two).
TEST(Tracing, ResolveBackendsTrimsWhitespace) {
    using nixl::trace::resolveTraceBackends;

    EXPECT_EQ(
        resolveTraceBackends(std::optional<std::string>{" nvtx , chakra "}, /*under_nsys=*/false),
        (std::vector<std::string>{"chakra", "nvtx"}));
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

// makeAgentTracer resolves the ratio before it decides whether any backend is
// active, so an unusable value is reported rather than accepted silently on the
// path where no tracer is built at all.
TEST(Tracing, InvalidSampleRatioIsRejectedWithTracingOff) {
    gtest::ScopedEnv env;
    env.addVar("NIXL_TRACE_BACKENDS", "");
    env.addVar(std::string(nixl::trace::traceSampleRatioVar), "abc");

    const nixlAgentConfig config;
    EXPECT_THROW(nixlAgent("sample_ratio_agent", config), std::invalid_argument);
}

// makeTracer ignores empty entries and returns null when no backend resolves to
// a loadable plugin (so callers can cheaply null-check).
TEST(Tracing, MakeTracerUnknownBackendReturnsNull) {
    {
        // No plugin exists for this backend, so it resolves to null with a warning.
        const gtest::LogIgnoreGuard ignore("requested but plugin");
        const auto tracer = nixl::trace::makeTracer({"agent", {kUnloadableBackend}});
        EXPECT_EQ(tracer, nullptr);
    }

    const auto none = nixl::trace::makeTracer({"agent", {}});
    EXPECT_EQ(none, nullptr);
}

// The no-backend runtime path: a backend is requested but no plugin provides
// it, so makeTracer yields a null tracer. Call sites then fall back to a
// default-constructed Span -- the stub path that does essentially nothing --
// and it must stay safe. NVTX's functional coverage lives in the e2e transfer
// tests, which load the real plugin.
TEST(Tracing, RequestedBackendWithoutPluginIsInert) {
    {
        const gtest::LogIgnoreGuard ignore("requested but plugin");
        const auto tracer = nixl::trace::makeTracer({"stub_agent", {kUnloadableBackend}});
        EXPECT_EQ(tracer, nullptr);
    }

    // What every traced call site binds to when the tracer is null.
    nixl::trace::Span span;
    EXPECT_FALSE(span.active());
    span.addAttribute("bytes", std::int64_t{1024});
    span.addAttribute("peer", std::string_view{"peer_agent"});
    EXPECT_EQ(span.id().value, 0u);
}

// Cross-thread correlation: push/pop must fan out to every backend, so a
// context set at the request boundary reaches all active backends.
TEST(Tracing, ContextFansOutToAllBackends) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto context = testContext(0x11);

    tracer->pushContext(context);
    tracer->popContext();

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, std::vector{context});
        EXPECT_EQ(log->poppedContexts, std::vector{context});
    }
}

// CorrelationScope is the RAII guard used at call sites: it pushes on
// construction and pops on destruction, across every backend.
TEST(Tracing, CorrelationScopeBalancesPushPop) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto context = testContext(0x12);

    {
        const nixl::trace::CorrelationScope scope(tracer.get(), context);
        const auto span = tracer->beginSpan("op");
        for (const CallLog *log : {&a, &b}) {
            EXPECT_EQ(log->activeContexts, std::vector{context});
            EXPECT_TRUE(log->poppedContexts.empty());
        }
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, std::vector{context});
        EXPECT_EQ(log->poppedContexts, std::vector{context});
        EXPECT_TRUE(log->activeContexts.empty());
    }
}

TEST(Tracing, CorrelationScopePushesUnsampledContext) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto unsampled = testContext(0x13, 0x02);
    ASSERT_TRUE(unsampled.valid());
    ASSERT_FALSE(unsampled.sampled());

    { const nixl::trace::CorrelationScope scope(tracer.get(), unsampled); }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, std::vector{unsampled});
        EXPECT_EQ(log->poppedContexts, std::vector{unsampled});
    }
}

// A null tracer (no active backend) must make CorrelationScope a safe no-op.
TEST(Tracing, CorrelationScopeNullTracerIsInert) {
    nixl::trace::Tracer *tracer = nullptr;
    { const nixl::trace::CorrelationScope scope(tracer, testContext(0x14)); }
    SUCCEED();
}

TEST(Tracing, CorrelationScopeInvalidContextIsInert) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    auto no_trace_id = testContext(0x15);
    no_trace_id.traceId.fill(0);
    auto no_span_id = testContext(0x15);
    no_span_id.spanId.fill(0);

    for (const auto &context : {nixl::trace::TraceContext{}, no_trace_id, no_span_id}) {
        const nixl::trace::CorrelationScope scope(tracer.get(), context);
        const auto span = tracer->beginSpan("op");
    }

    EXPECT_EQ(a.spansBegun, 3);
    for (const CallLog *log : {&a, &b}) {
        EXPECT_TRUE(log->pushedContexts.empty());
        EXPECT_TRUE(log->poppedContexts.empty());
    }
}

TEST(Tracing, NestedCorrelationScopesPopInReverseOrder) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto outer = testContext(0x16);
    const auto inner = testContext(0x17);

    {
        const nixl::trace::CorrelationScope outer_scope(tracer.get(), outer);
        {
            const nixl::trace::CorrelationScope inner_scope(tracer.get(), inner);
            EXPECT_EQ(a.activeContexts, (std::vector{outer, inner}));
        }
        EXPECT_EQ(a.activeContexts, std::vector{outer});
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, (std::vector{outer, inner}));
        EXPECT_EQ(log->poppedContexts, (std::vector{inner, outer}));
        EXPECT_TRUE(log->activeContexts.empty());
    }
}

TEST(Tracing, RequestStoresFixedCorrelationContext) {
    nixl::trace::TraceContext context;
    context.traceId = {0x4b, 0xf9, 0x2f, 0x35, 0x77, 0xb3, 0x4d, 0xa6};
    context.spanId = {0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7};
    context.flags = 0x03;

    const nixlXferReqH request("remote",
                               NIXL_WRITE,
                               DRAM_SEG,
                               DRAM_SEG,
                               0,
                               nixl_remote_section_weak_t{},
                               context,
                               0x0123456789abcdefULL);

    EXPECT_EQ(request.traceCorrelationId64(), 0x00f067aa0ba902b7ULL);
    EXPECT_EQ(request.traceContext(), context);
    EXPECT_EQ(request.traceRequestId(), 0x0123456789abcdefULL);
    EXPECT_EQ(request.traceExecution(), 0u);
}

TEST(Tracing, RequestContextsAreDistinctAndStable) {
    const nixlXferReqH first("remote",
                             NIXL_WRITE,
                             DRAM_SEG,
                             DRAM_SEG,
                             0,
                             nixl_remote_section_weak_t{},
                             nixl::trace::generateTraceContext(),
                             0);
    const nixlXferReqH second("remote",
                              NIXL_WRITE,
                              DRAM_SEG,
                              DRAM_SEG,
                              0,
                              nixl_remote_section_weak_t{},
                              nixl::trace::generateTraceContext(),
                              0);
    const auto first_id = first.traceCorrelationId64();

    EXPECT_NE(first_id, second.traceCorrelationId64());
    EXPECT_EQ(first.traceCorrelationId64(), first_id);
}

TEST(Tracing, RequestDefaultContextHasInertCorrelation) {
    const nixlXferReqH request("remote",
                               NIXL_WRITE,
                               DRAM_SEG,
                               DRAM_SEG,
                               0,
                               nixl_remote_section_weak_t{},
                               nixl::trace::TraceContext{},
                               0);

    EXPECT_EQ(request.traceCorrelationId64(), 0u);
}

TEST(Tracing, ActiveTracerConstructsGeneratedContext) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);

    const nixl::trace::TraceContext first{tracer.get()};
    const nixl::trace::TraceContext second{tracer.get()};

    EXPECT_TRUE(first.valid());
    EXPECT_TRUE(second.valid());
    EXPECT_NE(first.correlationId64(), second.correlationId64());
}

// The tracer carries the sampling ratio to the generation call sites, which only
// have the tracer pointer.
TEST(Tracing, TracerSampleRatioReachesGeneratedContexts) {
    CallLog a, b;
    const auto unsampled_tracer = makeMockTracer(a, b);
    EXPECT_EQ(unsampled_tracer->sampleRatio(), 0.0);
    EXPECT_FALSE(nixl::trace::TraceContext{unsampled_tracer.get()}.sampled());

    CallLog c, d;
    const auto sampled_tracer = makeMockTracer(c, d, 0, 0, /*sample_ratio=*/1.0);
    EXPECT_EQ(sampled_tracer->sampleRatio(), 1.0);

    const nixl::trace::TraceContext context{sampled_tracer.get()};
    EXPECT_TRUE(context.valid());
    EXPECT_TRUE(context.sampled());
}

TEST(Tracing, ContextGenerationFailureIsUntracedAndLoggedOncePerTracer) {
    gtest::LogMessageCounter failures("could not generate a trace context");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0, 0, 1.0, failRandomBytes);

    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(nixl::trace::TraceContext{tracer.get()}, nixl::trace::TraceContext{});
    }
    EXPECT_EQ(failures.count(), 1u);

    CallLog c, d;
    const auto other_tracer = makeMockTracer(c, d, 0, 0, 1.0, failRandomBytes);
    EXPECT_EQ(nixl::trace::TraceContext{other_tracer.get()}, nixl::trace::TraceContext{});
    EXPECT_EQ(failures.count(), 2u);
}

TEST(Tracing, RequestIdIsDrawnAgainWhileZero) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0, 0, 0.0, zeroFirstRandomBytes);
    zero_draws_left = 2;

    EXPECT_EQ(nixl::trace::generateRequestId(tracer.get()), 0x5a5a5a5a5a5a5a5aULL);
}

TEST(Tracing, RequestIdIsNoneWithoutTracerOrSource) {
    gtest::LogMessageCounter failures("could not generate a trace context");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0, 0, 0.0, failRandomBytes);

    EXPECT_EQ(nixl::trace::generateRequestId(nullptr), 0u);
    EXPECT_EQ(nixl::trace::generateRequestId(tracer.get()), 0u);
    EXPECT_EQ(failures.count(), 1u);
}

// A phase recorded through the plugin-facing sink reaches every enabled trace
// backend, tagged with the recording backend and the plugin's timestamp.
TEST(TracePhaseSink, RecordedPhaseReachesEveryBackend) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};

    sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, {}, 1234, {}, nullptr);

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->spansBegun, 0);
        ASSERT_EQ(log->emittedEvents.size(), 1u);
        const EmittedRecord &event = log->emittedEvents[0];
        EXPECT_EQ(event.name, "nixl::wire.submitted");
        EXPECT_EQ(event.times, std::vector<nixlTime::us_t>{1234});
        EXPECT_EQ(event.attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", std::int64_t{1234}}}));
    }
}

// Every enumerator maps to a distinct event name, so timelines stay comparable
// across plugins.
TEST(TracePhaseSink, PhaseVocabularyIsDistinct) {
    constexpr nixl_trace_phase_t kPhases[] = {nixl_trace_phase_t::SUBMIT,
                                              nixl_trace_phase_t::WIRE_SUBMITTED,
                                              nixl_trace_phase_t::WIRE_COMPLETED,
                                              nixl_trace_phase_t::NOTIF_SENT,
                                              nixl_trace_phase_t::NOTIF_RECEIVED,
                                              nixl_trace_phase_t::REMOTE_OBSERVED,
                                              nixl_trace_phase_t::OTHER};

    std::set<std::string_view> names;
    for (const auto phase : kPhases) {
        names.insert(toStringView(phase));
    }
    EXPECT_EQ(names.size(), std::size(kPhases));
}

TEST(TracePhaseSink, NotificationPhasesAreSendAndReceive) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    constexpr std::pair<nixl_trace_phase_t, nixl::trace::Kind> kKinds[] = {
        {nixl_trace_phase_t::SUBMIT, nixl::trace::Kind::Metadata},
        {nixl_trace_phase_t::WIRE_SUBMITTED, nixl::trace::Kind::Metadata},
        {nixl_trace_phase_t::WIRE_COMPLETED, nixl::trace::Kind::Metadata},
        {nixl_trace_phase_t::NOTIF_SENT, nixl::trace::Kind::CommSend},
        {nixl_trace_phase_t::NOTIF_RECEIVED, nixl::trace::Kind::CommRecv},
        {nixl_trace_phase_t::REMOTE_OBSERVED, nixl::trace::Kind::Metadata},
        {nixl_trace_phase_t::OTHER, nixl::trace::Kind::Metadata}};

    for (const auto &[phase, kind] : kKinds) {
        sink.recordPhase(phase, {}, 1, {}, nullptr);
    }

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), std::size(kKinds));
        for (std::size_t i = 0; i < std::size(kKinds); ++i) {
            EXPECT_EQ(log->emittedEvents[i].kind, kKinds[i].second) << log->emittedEvents[i].name;
        }
    }
}

// The generic phase carries a plugin-supplied label as the event name; an empty
// label falls back to the enum's own name.
TEST(TracePhaseSink, GenericPhaseUsesPluginLabel) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "LIBFABRIC"};

    sink.recordPhase(nixl_trace_phase_t::OTHER, "post_write", 1, {}, nullptr);
    sink.recordPhase(nixl_trace_phase_t::OTHER, {}, 2, {}, nullptr);

    ASSERT_EQ(a.emittedEvents.size(), 2u);
    EXPECT_EQ(a.emittedEvents[0].name, "post_write");
    EXPECT_EQ(a.emittedEvents[1].name, "nixl::phase");
}

// A label passed with any phase is recorded, not just the generic one: the
// event name stays the fixed vocabulary entry, and the label rides along as an
// attribute so it cannot be silently dropped.
TEST(TracePhaseSink, LabelIsRecordedForEveryPhase) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "LIBFABRIC"};

    sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, "rail0", 5, {}, nullptr);

    ASSERT_EQ(a.emittedEvents.size(), 1u);
    EXPECT_EQ(a.emittedEvents[0].name, "nixl::wire.submitted");
    EXPECT_EQ(a.emittedEvents[0].attrs,
              (attr_list_t{{"nixl.backend", std::string("LIBFABRIC")},
                           {"nixl.phase.timestamp_us", std::int64_t{5}},
                           {"nixl.phase.label", std::string("rail0")}}));
}

// An unlabelled phase adds no label attribute at all.
TEST(TracePhaseSink, EmptyLabelAddsNoAttribute) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};

    sink.recordPhase(nixl_trace_phase_t::SUBMIT, {}, 5, {}, nullptr);

    ASSERT_EQ(a.emittedEvents.size(), 1u);
    for (const auto &attr : a.emittedEvents[0].attrs) {
        EXPECT_NE(attr.first, "nixl.phase.label");
    }
}

// Plugin-supplied attributes are forwarded after the ones core adds.
TEST(TracePhaseSink, PluginAttributesAreForwarded) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "LIBFABRIC"};

    const nixlBackendTraceAttr attrs[] = {{"rail", "0"}, {"op", "write"}};
    sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, {}, 7, attrs, nullptr);

    ASSERT_EQ(a.emittedEvents.size(), 1u);
    EXPECT_EQ(a.emittedEvents[0].attrs,
              (attr_list_t{{"nixl.backend", std::string("LIBFABRIC")},
                           {"nixl.phase.timestamp_us", std::int64_t{7}},
                           {"rail", std::string("0")},
                           {"op", std::string("write")}}));
}

TEST(TracePhaseSink, ValidContextCorrelatesAndRecordsTraceparent) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    constexpr std::string_view traceparent =
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    const auto context = nixl::trace::parseTraceparent(traceparent);
    ASSERT_TRUE(context.has_value());
    const nixl::trace::TraceContext zeroed;

    sink.recordPhase(nixl_trace_phase_t::NOTIF_RECEIVED, {}, 1, {}, nullptr);
    sink.recordPhase(nixl_trace_phase_t::NOTIF_RECEIVED, {}, 2, {}, &zeroed);
    sink.recordPhase(nixl_trace_phase_t::NOTIF_RECEIVED, {}, 3, {}, &*context);

    for (const CallLog *log : {&a, &b}) {
        EXPECT_TRUE(log->pushedContexts.empty());
        ASSERT_EQ(log->emittedEvents.size(), 3u);
        EXPECT_EQ(log->emittedEvents[0].context, nixl::trace::TraceContext{});
        EXPECT_EQ(log->emittedEvents[1].context, zeroed);
        EXPECT_EQ(log->emittedEvents[2].context, *context);
        EXPECT_EQ(log->emittedEvents[0].attrs.size(), 2u);
        EXPECT_EQ(log->emittedEvents[1].attrs.size(), 2u);
        ASSERT_EQ(log->emittedEvents[2].attrs.size(), 3u);
        EXPECT_EQ(log->emittedEvents[2].attrs[2],
                  (attr_t{"nixl.traceparent", std::string(traceparent)}));
    }
}

// A backend may decline a span; Tracer drops the null, leaving the Span
// inactive even though the tracer has a backend. Nothing must be recorded and
// no null span may be dereferenced.
TEST(Tracing, InactiveSpanRecordsNothing) {
    CallLog a;
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<MockBackend>("a", &a, 0, /*null_spans=*/true));
    nixl::trace::Tracer tracer{std::move(backends)};
    auto span = tracer.beginSpan("op");

    span.addAttribute("rail", std::string_view{"0"});
    span.addAttribute("bytes", std::int64_t{1});

    EXPECT_FALSE(tracer.empty());
    EXPECT_FALSE(span.active());
    EXPECT_EQ(a.spansBegun, 1);
    EXPECT_EQ(a.spansEnded, 0);
    EXPECT_TRUE(a.strAttrs.empty());
    EXPECT_TRUE(a.intAttrs.empty());
}

// recordPhase() is noexcept: a backend that throws must not propagate, and the
// drop must be reported once per sink however many phases are dropped.
TEST(TracePhaseSink, ThrowingBackendIsReportedOnce) {
    const auto tracer = makeThrowingTracer();
    gtest::LogMessageCounter drops("Dropping a trace phase");
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};

    EXPECT_NO_THROW(sink.recordPhase(nixl_trace_phase_t::SUBMIT, {}, 1, {}, nullptr));
    EXPECT_NO_THROW(sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, {}, 2, {}, nullptr));

    EXPECT_EQ(drops.count(), 1u);
}
