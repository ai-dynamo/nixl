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
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "../allocation_counter.h"
#include "common.h"
#include "nixl.h"
#include "transfer_request.h"
#include "tracing/core/trace.h"
#include "tracing/core/trace_config.h"
#include "tracing/core/trace_context.h"
#include "tracing/api/tracing.h"
#include "tracing/core/trace_sink.h"

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
    std::uint64_t requestId;
    std::vector<nixlTime::us_t> times;
    attr_list_t attrs;
};

// Records the calls a single backend receives, so tests can assert that one
// call site fanned out to every backend.
struct CallLog {
    int spansBegun = 0;
    int spansEnded = 0;
    std::vector<std::string> spanNames;
    std::vector<std::pair<std::string, std::string>> strAttrs;
    std::vector<std::pair<std::string, std::int64_t>> intAttrs;
    std::vector<std::pair<std::string, double>> dblAttrs;
    std::vector<nixl::trace::TraceContext> pushedContexts;
    std::vector<std::uint64_t> pushedRequestIds;
    std::vector<nixl::trace::TraceContext> activeContexts;
    std::vector<nixl::trace::TraceContext> poppedContexts;
    std::vector<EmittedRecord> emittedSpans;
    std::vector<EmittedRecord> emittedEvents;
    std::vector<EmittedRecord> contextSpans;
    std::vector<EmittedRecord> emittedContextSpans;
};

[[nodiscard]] EmittedRecord
makeRecord(std::string_view name,
           nixl::trace::Kind kind,
           const nixl::trace::TraceContext &context,
           std::uint64_t request_id,
           std::vector<nixlTime::us_t> times,
           std::span<const nixl::trace::Attribute> attrs) {
    EmittedRecord record{std::string(name), kind, context, request_id, std::move(times), {}};
    for (const auto &attr : attrs) {
        record.attrs.emplace_back(
            std::string(nixl::trace::toStringView(attr.key)),
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
    explicit MockSpan(CallLog *log) : log_(log) {}

    ~MockSpan() override {
        ++log_->spansEnded;
    }

    void
    addAttribute(nixl::trace::AttributeKey key, std::string_view value) override {
        log_->strAttrs.emplace_back(std::string(nixl::trace::toStringView(key)),
                                    std::string(value));
    }

    void
    addAttribute(nixl::trace::AttributeKey key, std::int64_t value) override {
        log_->intAttrs.emplace_back(std::string(nixl::trace::toStringView(key)), value);
    }

    void
    addAttribute(nixl::trace::AttributeKey key, double value) override {
        log_->dblAttrs.emplace_back(std::string(nixl::trace::toStringView(key)), value);
    }

private:
    CallLog *log_;
};

class MockBackend : public nixl::trace::TraceBackend {
public:
    MockBackend(std::string name, CallLog *log, bool null_spans = false)
        : name_(std::move(name)),
          log_(log),
          nullSpans_(null_spans) {}

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(nixl::trace::Name name, nixl::trace::Kind) override {
        ++log_->spansBegun;
        log_->spanNames.emplace_back(name.view());
        if (nullSpans_) {
            return nullptr;
        }
        return std::make_unique<MockSpan>(log_);
    }

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginContextSpan(nixl::trace::Name name,
                     nixl::trace::Kind kind,
                     const nixl::trace::TraceIdentity &identity) override {
        log_->contextSpans.push_back(
            makeRecord(name.view(), kind, identity.context, *identity.requestId, {}, {}));
        if (nullSpans_) {
            return nullptr;
        }
        return std::make_unique<MockSpan>(log_);
    }

    void
    emitSpan(nixl::trace::Name name,
             nixl::trace::Kind kind,
             const nixl::trace::TraceIdentity &identity,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const nixl::trace::Attribute> attrs) override {
        log_->emittedSpans.push_back(makeRecord(
            name.view(), kind, identity.context, *identity.requestId, {start, end}, attrs));
    }

    void
    emitEvent(nixl::trace::Name name,
              nixl::trace::Kind kind,
              const nixl::trace::TraceIdentity &identity,
              nixlTime::us_t at,
              std::span<const nixl::trace::Attribute> attrs) override {
        log_->emittedEvents.push_back(
            makeRecord(name.view(), kind, identity.context, *identity.requestId, {at}, attrs));
    }

    void
    emitContextSpan(nixl::trace::Name name,
                    nixl::trace::Kind kind,
                    const nixl::trace::TraceIdentity &identity,
                    nixlTime::us_t start,
                    nixlTime::us_t end,
                    std::span<const nixl::trace::Attribute> attrs) override {
        log_->emittedContextSpans.push_back(makeRecord(
            name.view(), kind, identity.context, *identity.requestId, {start, end}, attrs));
    }

    void
    pushContext(const nixl::trace::TraceIdentity &identity) override {
        log_->pushedContexts.push_back(identity.context);
        log_->pushedRequestIds.push_back(*identity.requestId);
        log_->activeContexts.push_back(identity.context);
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
    bool nullSpans_;
};

class SelfTimedBackend final : public MockBackend {
public:
    using MockBackend::MockBackend;

    [[nodiscard]] bool
    ownsTimestamps() const noexcept override {
        return true;
    }
};

class TimestampQuestionSink final : public nixlBackendTraceSink {
public:
    using nixlBackendTraceSink::recordPhase;

    void
    recordPhase(nixl_trace_phase_t,
                nixlTime::us_t timestamp,
                const nixl::trace::TraceIdentity &,
                std::span<const nixlBackendTraceAttr>,
                std::string_view) noexcept override {
        timestamps.push_back(timestamp);
    }

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
        ++questions;
        return true;
    }

    [[nodiscard]] const nixl::trace::TraceIdentity *
    currentIdentity() const noexcept override {
        return nullptr;
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
    deferPhase(const nixl::trace::DeferredPhase &) noexcept override {}

    void
    recordDeferredPhases() noexcept override {}

    mutable int questions = 0;
    std::vector<nixlTime::us_t> timestamps;
};

class DrainCountingSink final : public nixlBackendTraceSink {
public:
    explicit DrainCountingSink(const std::atomic<std::uint64_t> *pending) noexcept
        : nixlBackendTraceSink(pending) {}

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

    [[nodiscard]] const nixl::trace::TraceIdentity *
    currentIdentity() const noexcept override {
        return nullptr;
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
    deferPhase(const nixl::trace::DeferredPhase &) noexcept override {}

    void
    recordDeferredPhases() noexcept override {
        ++drains;
    }

    int drains = 0;
};

class ThrowingBackend final : public nixl::trace::TraceBackend {
public:
    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(nixl::trace::Name, nixl::trace::Kind) override {
        throw std::runtime_error("beginSpan failed");
    }

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginContextSpan(nixl::trace::Name,
                     nixl::trace::Kind,
                     const nixl::trace::TraceIdentity &) override {
        throw std::runtime_error("beginContextSpan failed");
    }

    void
    emitSpan(nixl::trace::Name,
             nixl::trace::Kind,
             const nixl::trace::TraceIdentity &,
             nixlTime::us_t,
             nixlTime::us_t,
             std::span<const nixl::trace::Attribute>) override {
        throw std::runtime_error("emitSpan failed");
    }

    void
    emitEvent(nixl::trace::Name,
              nixl::trace::Kind,
              const nixl::trace::TraceIdentity &,
              nixlTime::us_t,
              std::span<const nixl::trace::Attribute>) override {
        throw std::runtime_error("emitEvent failed");
    }

    void
    emitContextSpan(nixl::trace::Name,
                    nixl::trace::Kind,
                    const nixl::trace::TraceIdentity &,
                    nixlTime::us_t,
                    nixlTime::us_t,
                    std::span<const nixl::trace::Attribute>) override {
        throw std::runtime_error("emitContextSpan failed");
    }

    void
    pushContext(const nixl::trace::TraceIdentity &) override {}

    void
    popContext() override {}

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return "throwing";
    }
};

class DiscardingBackend final : public nixl::trace::TraceBackend {
public:
    explicit DiscardingBackend(std::size_t &attrs_seen) noexcept : attrsSeen_(attrs_seen) {}

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(nixl::trace::Name, nixl::trace::Kind) override {
        return nullptr;
    }

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginContextSpan(nixl::trace::Name,
                     nixl::trace::Kind,
                     const nixl::trace::TraceIdentity &) override {
        return nullptr;
    }

    void
    emitSpan(nixl::trace::Name,
             nixl::trace::Kind,
             const nixl::trace::TraceIdentity &,
             nixlTime::us_t,
             nixlTime::us_t,
             std::span<const nixl::trace::Attribute> attrs) override {
        attrsSeen_ += attrs.size();
    }

    void
    emitEvent(nixl::trace::Name,
              nixl::trace::Kind,
              const nixl::trace::TraceIdentity &,
              nixlTime::us_t,
              std::span<const nixl::trace::Attribute> attrs) override {
        attrsSeen_ += attrs.size();
    }

    void
    emitContextSpan(nixl::trace::Name,
                    nixl::trace::Kind,
                    const nixl::trace::TraceIdentity &,
                    nixlTime::us_t,
                    nixlTime::us_t,
                    std::span<const nixl::trace::Attribute> attrs) override {
        attrsSeen_ += attrs.size();
    }

    void
    pushContext(const nixl::trace::TraceIdentity &) override {}

    void
    popContext() override {}

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return "discarding";
    }

private:
    std::size_t &attrsSeen_;
};

[[nodiscard]] std::unique_ptr<nixl::trace::CompositeTracer>
makeDiscardingTracer(std::size_t &attrs_seen) {
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<DiscardingBackend>(attrs_seen));
    return std::make_unique<nixl::trace::CompositeTracer>(std::move(backends), 1.0);
}

void
failRandomBytes(std::span<std::uint8_t>) {
    throw std::system_error(ENOSYS, std::generic_category(), "getentropy failed");
}

int zero_draws_left = 0;

void
zeroFirstRandomBytes(std::span<std::uint8_t> bytes) {
    std::ranges::fill(bytes, (zero_draws_left-- > 0) ? 0 : 0x5a);
}

std::size_t random_draws = 0;

void
countedSequenceRandomBytes(std::span<std::uint8_t> bytes) {
    ++random_draws;
    std::iota(bytes.begin(), bytes.end(), std::uint8_t{1});
}

void
zeroSpanIdOnceRandomBytes(std::span<std::uint8_t> bytes) {
    const bool first = (random_draws++ == 0);
    std::ranges::fill(bytes, 0x5a);
    if (first && (bytes.size() == 32)) {
        std::ranges::fill(bytes.first(16), 0x11);
        std::ranges::fill(bytes.subspan(16, 8), 0x00);
        std::ranges::fill(bytes.last(8), 0x22);
    }
}

[[nodiscard]] std::unique_ptr<nixl::trace::CompositeTracer>
makeMockTracer(CallLog &a,
               CallLog &b,
               double sample_ratio = 0.0,
               nixl::trace::random_bytes_fn_t random_bytes = nullptr) {
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<MockBackend>("a", &a));
    backends.push_back(std::make_unique<MockBackend>("b", &b));
    return std::make_unique<nixl::trace::CompositeTracer>(
        std::move(backends), sample_ratio, random_bytes);
}

[[nodiscard]] std::unique_ptr<nixl::trace::CompositeTracer>
makeThrowingTracer() {
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<ThrowingBackend>());
    return std::make_unique<nixl::trace::CompositeTracer>(std::move(backends), 0.0);
}

[[nodiscard]] const nixl::trace::TraceIdentity *
currentIdentity(nixl::trace::Tracer &tracer) {
    return tracer.makeSink("test", {})->currentIdentity();
}

[[nodiscard]] nixl::trace::TraceContext
testContext(std::uint8_t id_byte, std::uint8_t flags = 0x01) {
    nixl::trace::TraceContext context;
    context.traceId->fill(id_byte);
    context.spanId->fill(id_byte);
    context.flags = flags;
    return context;
}

template<typename T>
concept hasAttributeValue = requires(const T &value) { nixl::trace::attributeValue(value); };

static_assert(hasAttributeValue<float>);
static_assert(!hasAttributeValue<std::atomic<std::uint64_t>>);

} // namespace

TEST(Tracing, AttributeKeysHaveDistinctTerminatedNames) {
    std::set<std::string_view> names;
    for (std::size_t i = 0; i < nixl::trace::attributeKeyCount; ++i) {
        const auto name = nixl::trace::toStringView(static_cast<nixl::trace::AttributeKey>(i));
        EXPECT_NE(name, "nixl.attribute");
        EXPECT_EQ(std::string_view(name.data()), name);
        EXPECT_TRUE(names.insert(name).second) << name;
    }
}

TEST(Tracing, AttributeKeyPastTheListIsNamedGenerically) {
    const auto unknown = static_cast<nixl::trace::AttributeKey>(nixl::trace::attributeKeyCount);

    EXPECT_EQ(nixl::trace::toStringView(unknown), "nixl.attribute");
    EXPECT_EQ(std::string_view(nixl::trace::toStringView(unknown).data()), "nixl.attribute");
}

// A single beginSpan() must fan out to every enabled backend, and the Span's
// destructor must end all of them. This is the core requirement.
TEST(Tracing, FanOutToAllBackends) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    EXPECT_FALSE(tracer->empty());

    {
        auto span = tracer->beginSpan("op", nixl::trace::Kind::CommSend);
        EXPECT_TRUE(span.active());
        span.addAttribute(nixl::trace::AttributeKey::RemoteAgent, std::string_view{"val"});
    } // span ends here

    EXPECT_EQ(a.spansBegun, 1);
    EXPECT_EQ(b.spansBegun, 1);
    EXPECT_EQ(a.spansEnded, 1);
    EXPECT_EQ(b.spansEnded, 1);

    ASSERT_EQ(a.strAttrs.size(), 1u);
    ASSERT_EQ(b.strAttrs.size(), 1u);
    EXPECT_EQ(a.strAttrs[0].first, "remote_agent");
    EXPECT_EQ(a.strAttrs[0].second, "val");
    EXPECT_EQ(b.strAttrs[0].first, "remote_agent");
    EXPECT_EQ(b.strAttrs[0].second, "val");
}

TEST(Tracing, ContextSpanFansOutToAllBackends) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto context = testContext(0x24);

    {
        auto span = tracer->beginContextSpan(
            "op", nixl::trace::Kind::CommSend, {context, nixl::trace::RequestId{0x2424}});
        EXPECT_TRUE(span.active());
        span.addAttribute(nixl::trace::AttributeKey::RemoteAgent, std::string_view{"val"});
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->spansBegun, 0);
        ASSERT_EQ(log->contextSpans.size(), 1u);
        EXPECT_EQ(log->contextSpans[0].name, "op");
        EXPECT_EQ(log->contextSpans[0].kind, nixl::trace::Kind::CommSend);
        EXPECT_EQ(log->contextSpans[0].context, context);
        EXPECT_EQ(log->contextSpans[0].requestId, 0x2424u);
        EXPECT_EQ(log->spansEnded, 1);
        ASSERT_EQ(log->strAttrs.size(), 1u);
        EXPECT_EQ(log->strAttrs[0].first, "remote_agent");
        EXPECT_EQ(log->strAttrs[0].second, "val");
    }
}

// Typed attribute overloads must reach every backend with the correct type.
TEST(Tracing, TypedAttributesPropagate) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);

    {
        auto span = tracer->beginSpan("op");
        span.addAttribute(nixl::trace::AttributeKey::Bytes, std::int64_t{4096});
        span.addAttribute(nixl::trace::AttributeKey::NixlStartUs, 0.5);
        span.addAttribute(nixl::trace::AttributeKey::RemoteAgent, std::string_view{"agent_1"});
    }

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->intAttrs.size(), 1u);
        EXPECT_EQ(log->intAttrs[0].first, "bytes");
        EXPECT_EQ(log->intAttrs[0].second, 4096);
        ASSERT_EQ(log->dblAttrs.size(), 1u);
        EXPECT_EQ(log->dblAttrs[0].first, "nixl.start_us");
        EXPECT_DOUBLE_EQ(log->dblAttrs[0].second, 0.5);
        ASSERT_EQ(log->strAttrs.size(), 1u);
        EXPECT_EQ(log->strAttrs[0].second, "agent_1");
    }
}

TEST(Tracing, AttributeListReachesEveryBackendInOrder) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const nixl::trace::Attribute attrs[] = {
        {nixl::trace::AttributeKey::Bytes, 4096},
        {nixl::trace::AttributeKey::NixlStartUs, 0.5},
        {nixl::trace::AttributeKey::RemoteAgent, std::string_view{"agent_1"}},
        {nixl::trace::AttributeKey::DescCount, 8},
    };

    {
        auto span = tracer->beginSpan("op");
        span.addAttributes(attrs);
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->intAttrs,
                  (std::vector<std::pair<std::string, std::int64_t>>{{"bytes", 4096},
                                                                     {"desc_count", 8}}));
        ASSERT_EQ(log->dblAttrs.size(), 1u);
        EXPECT_EQ(log->dblAttrs[0].first, "nixl.start_us");
        EXPECT_EQ(log->strAttrs,
                  (std::vector<std::pair<std::string, std::string>>{{"remote_agent", "agent_1"}}));
    }
}

TEST(Tracing, AttributeHoldsItsValueAsARecordDoes) {
    const nixl::trace::Attribute count{nixl::trace::AttributeKey::DescCount, 8};
    const nixl::trace::Attribute ratio{nixl::trace::AttributeKey::NixlStartUs, 0.5f};
    const std::string agent = "agent_1";
    const nixl::trace::Attribute peer{nixl::trace::AttributeKey::RemoteAgent, agent};

    EXPECT_EQ(std::get<std::int64_t>(count.value), 8);
    EXPECT_DOUBLE_EQ(std::get<double>(ratio.value), 0.5);
    EXPECT_EQ(std::get<std::string_view>(peer.value), "agent_1");
}

TEST(Tracing, AttrMacroPassesPlainValues) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const std::string peer = "agent_1";

    {
        auto span = tracer->beginSpan("op");
        NIXL_TRACE_ATTR(span, nixl::trace::AttributeKey::NixlStatus, NIXL_ERR_BACKEND);
        NIXL_TRACE_ATTR(span, nixl::trace::AttributeKey::Bytes, std::size_t{4096});
        NIXL_TRACE_ATTR(span, nixl::trace::AttributeKey::NixlXferExecution, std::uint64_t{1} << 40);
        NIXL_TRACE_ATTR(span, nixl::trace::AttributeKey::DescCount, true);
        NIXL_TRACE_ATTR(span, nixl::trace::AttributeKey::RemoteAgent, peer);
        NIXL_TRACE_ATTR(span, nixl::trace::AttributeKey::NixlRemoteAgent, "write");
        NIXL_TRACE_ATTR(span, nixl::trace::AttributeKey::NixlStartUs, 0.5);
    }

    const std::vector<std::pair<std::string, std::int64_t>> ints{
        {"nixl.status", NIXL_ERR_BACKEND},
        {"bytes", 4096},
        {"nixl.xfer_execution", std::int64_t{1} << 40},
        {"desc_count", 1}};
    const std::vector<std::pair<std::string, std::string>> strings{{"remote_agent", "agent_1"},
                                                                   {"nixl.remote_agent", "write"}};
    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->intAttrs, ints);
        EXPECT_EQ(log->strAttrs, strings);
        ASSERT_EQ(log->dblAttrs.size(), 1u);
        EXPECT_EQ(log->dblAttrs[0].first, "nixl.start_us");
        EXPECT_DOUBLE_EQ(log->dblAttrs[0].second, 0.5);
    }
}

TEST(Tracing, SpanMacroRecordsItsInlineAttributes) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const std::string peer = "agent_1";

    {
        NIXL_TRACE_SCOPE(span,
                         tracer.get(),
                         nixl::trace::Record::RegisterMem,
                         {nixl::trace::AttributeKey::NixlStatus, NIXL_ERR_BACKEND},
                         {nixl::trace::AttributeKey::Bytes, std::size_t{4096}},
                         {nixl::trace::AttributeKey::DescCount, true},
                         {nixl::trace::AttributeKey::RemoteAgent, peer},
                         {nixl::trace::AttributeKey::NixlRemoteAgent, "write"},
                         {nixl::trace::AttributeKey::NixlStartUs, 0.5});
        EXPECT_TRUE(span.active());
    }

    const std::vector<std::pair<std::string, std::int64_t>> ints{
        {"nixl.status", NIXL_ERR_BACKEND}, {"bytes", 4096}, {"desc_count", 1}};
    const std::vector<std::pair<std::string, std::string>> strings{{"remote_agent", "agent_1"},
                                                                   {"nixl.remote_agent", "write"}};
    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->spanNames, std::vector<std::string>{"nixl::registerMem"});
        EXPECT_EQ(log->intAttrs, ints);
        EXPECT_EQ(log->strAttrs, strings);
        ASSERT_EQ(log->dblAttrs.size(), 1u);
        EXPECT_EQ(log->dblAttrs[0].first, "nixl.start_us");
        EXPECT_DOUBLE_EQ(log->dblAttrs[0].second, 0.5);
    }
}

TEST(Tracing, SpanMacroSkipsItsAttributesWithoutASpan) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    int evaluations = 0;

    {
        NIXL_TRACE_SCOPE(none,
                         static_cast<nixl::trace::Tracer *>(nullptr),
                         nixl::trace::Record::RegisterMem,
                         {nixl::trace::AttributeKey::Bytes, ++evaluations});
        EXPECT_FALSE(none.active());
        nixl::trace::Traceable unsampled{tracer.get(),
                                         {testContext(0x41, 0x00), nixl::trace::RequestId{0x41}}};
        nixl::trace::Traceable untraced{nullptr, {testContext(0x43), nixl::trace::RequestId{0x43}}};
        for (nixl::trace::Traceable *traced : {&unsampled, &untraced}) {
            NIXL_TRACE_POST_SPAN(post,
                                 *traced,
                                 nullptr,
                                 nixl::trace::Record::PostXferReqWrite,
                                 {nixl::trace::AttributeKey::Bytes, ++evaluations});
            NIXL_TRACE_REQUEST_SPAN(release,
                                    *traced,
                                    nixl::trace::Record::ReleaseXferReq,
                                    {nixl::trace::AttributeKey::Bytes, ++evaluations});
            EXPECT_FALSE(post.active());
            EXPECT_FALSE(release.active());
            EXPECT_EQ(currentIdentity(*tracer), nullptr);
        }
    }

    EXPECT_EQ(evaluations, 0);
    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->spansBegun, 0);
        EXPECT_TRUE(log->contextSpans.empty());
        EXPECT_TRUE(log->pushedContexts.empty());
    }
}

TEST(Tracing, FinishMacroRecordsTheStatusAndReturnsIt) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);

    {
        NIXL_TRACE_SCOPE(span, tracer.get(), nixl::trace::Record::RegisterMem);
        EXPECT_EQ(NIXL_TRACE_FINISH(span, NIXL_ERR_BACKEND), NIXL_ERR_BACKEND);
        nixl::trace::Span inactive;
        EXPECT_EQ(NIXL_TRACE_FINISH(inactive, NIXL_ERR_NOT_FOUND), NIXL_ERR_NOT_FOUND);
    }

    const std::vector<std::pair<std::string, std::int64_t>> status{
        {"nixl.status", NIXL_ERR_BACKEND}};
    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->intAttrs, status);
    }
}

TEST(Tracing, EndMacroRecordsTheStatusAndEndsTheSpanAtOnce) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);

    {
        NIXL_TRACE_SCOPE(span, tracer.get(), nixl::trace::Record::RegisterMem);
        EXPECT_EQ(NIXL_TRACE_END(span, NIXL_ERR_BACKEND), NIXL_ERR_BACKEND);
        EXPECT_FALSE(span.active());
        for (const CallLog *log : {&a, &b}) {
            EXPECT_EQ(log->spansEnded, 1);
        }
    }

    const std::vector<std::pair<std::string, std::int64_t>> status{
        {"nixl.status", NIXL_ERR_BACKEND}};
    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->spansEnded, 1);
        EXPECT_EQ(log->intAttrs, status);
    }
}

TEST(Tracing, RequestSpanMacrosRecordInTheirOwnScope) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b, 0.0, zeroFirstRandomBytes);
    zero_draws_left = 0;
    const nixl::trace::TraceContext request = testContext(0x42);
    auto post_context = request;
    post_context.spanId->fill(0x5a);
    post_context.parentSpanId = request.spanId;

    {
        nixl::trace::Traceable traced{tracer.get(), {request, nixl::trace::RequestId{0x42}}};
        {
            NIXL_TRACE_POST_SPAN(post,
                                 traced,
                                 nullptr,
                                 nixl::trace::Record::PostXferReqWrite,
                                 {nixl::trace::AttributeKey::Bytes, 8});
            EXPECT_TRUE(post.active());
            EXPECT_EQ(traced.postIdentity().context, post_context);
            for (const CallLog *log : {&a, &b}) {
                EXPECT_EQ(log->activeContexts,
                          std::vector<nixl::trace::TraceContext>{post_context});
            }
        }
        {
            NIXL_TRACE_REQUEST_SPAN(release, traced, nixl::trace::Record::ReleaseXferReq);
            EXPECT_TRUE(release.active());
            for (const CallLog *log : {&a, &b}) {
                EXPECT_EQ(log->activeContexts, std::vector<nixl::trace::TraceContext>{request});
            }
        }
        EXPECT_EQ(currentIdentity(*tracer), nullptr);
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts,
                  (std::vector<nixl::trace::TraceContext>{post_context, request}));
        EXPECT_EQ(log->pushedRequestIds, (std::vector<std::uint64_t>{0x42, 0x42}));
        EXPECT_TRUE(log->activeContexts.empty());
        ASSERT_EQ(log->contextSpans.size(), 1u);
        EXPECT_EQ(log->contextSpans[0].name, "nixl::postXferReq.write");
        EXPECT_EQ(log->contextSpans[0].kind, nixl::trace::Kind::CommSend);
        EXPECT_EQ(log->contextSpans[0].context, post_context);
        EXPECT_EQ(log->contextSpans[0].requestId, 0x42u);
        EXPECT_EQ(log->spanNames, std::vector<std::string>{"nixl::releaseXferReq"});
        EXPECT_EQ(log->intAttrs,
                  (std::vector<std::pair<std::string, std::int64_t>>{{"bytes", 8},
                                                                     {"nixl.xfer_request_id", 0x42},
                                                                     {"nixl.xfer_execution", 1},
                                                                     {"nixl.xfer_request_id", 0x42},
                                                                     {"nixl.xfer_execution", 1}}));
    }
}

TEST(Tracing, ScopeUnderMacroRecordsTheSuppliedContextsChild) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b, 0.0, zeroFirstRandomBytes);
    zero_draws_left = 0;
    const auto application = testContext(0x61);
    const nixl_blob_t supplied = nixl::trace::formatTraceparent(application);
    auto child = application;
    child.spanId->fill(0x5a);
    child.parentSpanId = application.spanId;

    {
        nixl_opt_args_t options;
        options.traceContext = supplied;
        NIXL_TRACE_SCOPE_UNDER(notif,
                               tracer.get(),
                               &options,
                               nixl::trace::Record::GenNotif,
                               {nixl::trace::AttributeKey::Bytes, 1});
        EXPECT_TRUE(notif.active());
        ASSERT_NE(currentIdentity(*tracer), nullptr);
        EXPECT_EQ(currentIdentity(*tracer)->context, child);
        EXPECT_EQ(*currentIdentity(*tracer)->requestId, 0u);
    }

    EXPECT_EQ(currentIdentity(*tracer), nullptr);
    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, std::vector<nixl::trace::TraceContext>{child});
        EXPECT_EQ(log->spansBegun, 0);
        ASSERT_EQ(log->contextSpans.size(), 1u);
        EXPECT_EQ(log->contextSpans[0].name, "nixl::genNotif");
        EXPECT_EQ(log->contextSpans[0].context, child);
        EXPECT_EQ(log->contextSpans[0].requestId, 0u);
        EXPECT_EQ(log->intAttrs, (std::vector<std::pair<std::string, std::int64_t>>{{"bytes", 1}}));
    }
}

TEST(Tracing, ScopeUnderMacroIsAPlainSpanWithoutAUsableContext) {
    gtest::LogMessageCounter ignored("ignoring a supplied trace context");
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    nixl_opt_args_t unsampled;
    unsampled.traceContext = nixl::trace::formatTraceparent(testContext(0x62, 0x00));
    nixl_opt_args_t neither;
    neither.traceContext = "neither";
    int evaluations = 0;

    {
        NIXL_TRACE_SCOPE_UNDER(plain,
                               tracer.get(),
                               nullptr,
                               nixl::trace::Record::GenNotif,
                               {nixl::trace::AttributeKey::Bytes, 1});
        EXPECT_TRUE(plain.active());
        EXPECT_EQ(currentIdentity(*tracer), nullptr);
    }
    {
        NIXL_TRACE_SCOPE_UNDER(unparsed,
                               tracer.get(),
                               &neither,
                               nixl::trace::Record::GenNotif,
                               {nixl::trace::AttributeKey::Bytes, 2});
        EXPECT_TRUE(unparsed.active());
    }
    {
        NIXL_TRACE_SCOPE_UNDER(skipped,
                               tracer.get(),
                               &unsampled,
                               nixl::trace::Record::GenNotif,
                               {nixl::trace::AttributeKey::Bytes, ++evaluations});
        EXPECT_FALSE(skipped.active());
    }
    {
        NIXL_TRACE_SCOPE_UNDER(untraced,
                               static_cast<nixl::trace::Tracer *>(nullptr),
                               &unsampled,
                               nixl::trace::Record::GenNotif,
                               {nixl::trace::AttributeKey::Bytes, ++evaluations});
        EXPECT_FALSE(untraced.active());
    }

    EXPECT_EQ(evaluations, 0);
    EXPECT_EQ(ignored.count(), 1u);
    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->spanNames, (std::vector<std::string>{"nixl::genNotif", "nixl::genNotif"}));
        EXPECT_TRUE(log->contextSpans.empty());
        EXPECT_TRUE(log->pushedContexts.empty());
        EXPECT_EQ(log->intAttrs,
                  (std::vector<std::pair<std::string, std::int64_t>>{{"bytes", 1}, {"bytes", 2}}));
    }
}

TEST(Tracing, ScopeUnderMacroEvaluatesItsTracerAndOptionsOnce) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    nixl_opt_args_t sampled;
    sampled.traceContext = nixl::trace::formatTraceparent(testContext(0x63));
    int tracers = 0;
    int options = 0;
    const auto tracerOf = [&tracers](nixl::trace::Tracer *chosen) {
        ++tracers;
        return chosen;
    };
    const auto optionsOf = [&options, &sampled]() -> const nixl_opt_args_t * {
        ++options;
        return &sampled;
    };

    {
        NIXL_TRACE_SCOPE_UNDER(
            traced, tracerOf(tracer.get()), optionsOf(), nixl::trace::Record::GenNotif);
        EXPECT_NE(currentIdentity(*tracer), nullptr);
        EXPECT_TRUE(traced.active());
    }
    EXPECT_EQ(tracers, 1);
    EXPECT_EQ(options, 1);

    {
        NIXL_TRACE_SCOPE_UNDER(
            untraced, tracerOf(nullptr), optionsOf(), nixl::trace::Record::GenNotif);
        EXPECT_EQ(currentIdentity(*tracer), nullptr);
    }
    EXPECT_EQ(tracers, 2);
    EXPECT_EQ(options, 1);
}

TEST(Tracing, AttrMacroSkipsItsValueOnAnInactiveSpan) {
    int evaluations = 0;
    nixl::trace::Span span;

    NIXL_TRACE_ATTR(span, nixl::trace::AttributeKey::Bytes, ++evaluations);

    EXPECT_FALSE(span.active());
    EXPECT_EQ(evaluations, 0);
}

TEST(Tracing, EmittedRecordsFanOutToAllBackends) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto context = testContext(0x21, 0x02);
    const nixl::trace::Attribute attrs[] = {
        {nixl::trace::AttributeKey::RemoteAgent, std::string_view{"agent_1"}},
        {nixl::trace::AttributeKey::Bytes, std::int64_t{4096}},
        {nixl::trace::AttributeKey::NixlStartUs, 0.5}};
    const attr_list_t expected{{"remote_agent", std::string("agent_1")},
                               {"bytes", std::int64_t{4096}},
                               {"nixl.start_us", 0.5}};

    tracer->emitSpan("span",
                     nixl::trace::Kind::CommSend,
                     {context, nixl::trace::RequestId{0x2121}},
                     10,
                     25,
                     attrs);
    tracer->emitEvent(
        "event", nixl::trace::Kind::CommRecv, {context, nixl::trace::RequestId{0x2222}}, 42, attrs);

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedSpans.size(), 1u);
        EXPECT_EQ(log->emittedSpans[0].name, "span");
        EXPECT_EQ(log->emittedSpans[0].kind, nixl::trace::Kind::CommSend);
        EXPECT_EQ(log->emittedSpans[0].context, context);
        EXPECT_EQ(log->emittedSpans[0].requestId, 0x2121u);
        EXPECT_EQ(log->emittedSpans[0].times, (std::vector<nixlTime::us_t>{10, 25}));
        EXPECT_EQ(log->emittedSpans[0].attrs, expected);
        ASSERT_EQ(log->emittedEvents.size(), 1u);
        EXPECT_EQ(log->emittedEvents[0].name, "event");
        EXPECT_EQ(log->emittedEvents[0].kind, nixl::trace::Kind::CommRecv);
        EXPECT_EQ(log->emittedEvents[0].context, context);
        EXPECT_EQ(log->emittedEvents[0].requestId, 0x2222u);
        EXPECT_EQ(log->emittedEvents[0].times, std::vector<nixlTime::us_t>{42});
        EXPECT_EQ(log->emittedEvents[0].attrs, expected);
    }
}

TEST(Tracing, TracerKeepsAtMostMaxTraceBackends) {
    std::vector<CallLog> logs(nixl::trace::maxTraceBackends + 1);
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    for (CallLog &log : logs) {
        backends.push_back(std::make_unique<MockBackend>("mock", &log));
    }
    nixl::trace::CompositeTracer tracer{std::move(backends), 1.0};

    EXPECT_EQ(tracer.backends().size(), nixl::trace::maxTraceBackends);
    { auto span = tracer.beginSpan("op", nixl::trace::Kind::CommSend); }
    tracer.emitEvent("event",
                     nixl::trace::Kind::CommRecv,
                     {testContext(0x21), nixl::trace::RequestId{0x2121}},
                     42,
                     {});

    for (std::size_t i = 0; i < logs.size(); ++i) {
        const std::size_t expected = i < nixl::trace::maxTraceBackends ? 1 : 0;
        EXPECT_EQ(static_cast<std::size_t>(logs[i].spansBegun), expected) << "backend " << i;
        EXPECT_EQ(logs[i].emittedEvents.size(), expected) << "backend " << i;
    }
}

// A tracer with no backends is empty, and the spans it begins are inactive.
TEST(Tracing, EmptyTracerIsInert) {
    nixl::trace::CompositeTracer tracer{std::vector<std::unique_ptr<nixl::trace::TraceBackend>>{},
                                        0.0};
    EXPECT_TRUE(tracer.empty());

    auto span = tracer.beginSpan("op", nixl::trace::Kind::CommRecv);
    EXPECT_FALSE(span.active());
    const auto context_span = tracer.beginContextSpan(
        "c", nixl::trace::Kind::CommSend, {testContext(0x23), nixl::trace::RequestId{0x2323}});
    EXPECT_FALSE(context_span.active());
}

// A default-constructed Span (what the macros bind when the tracer is null) is
// inert.
TEST(Tracing, DefaultSpanIsInert) {
    nixl::trace::Span span;
    EXPECT_FALSE(span.active());
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

// loadTraceBackends ignores empty entries and loads nothing when no backend
// resolves to a loadable plugin, so the agent builds no tracer.
TEST(Tracing, LoadTraceBackendsSkipsAnUnknownBackend) {
    {
        // No plugin exists for this backend, so it loads nothing, with a warning.
        const gtest::LogIgnoreGuard ignore("requested but plugin");
        EXPECT_TRUE(nixl::trace::loadTraceBackends({kUnloadableBackend}, "agent").empty());
    }

    EXPECT_TRUE(nixl::trace::loadTraceBackends({}, "agent").empty());
}

// The no-backend runtime path: a backend is requested but no plugin provides
// it, so no backend loads and the agent has no tracer. NVTX's functional coverage
// lives in the e2e transfer tests, which load the real plugin.
TEST(Tracing, RequestedBackendWithoutPluginIsInert) {
    const gtest::LogIgnoreGuard ignore("requested but plugin");
    EXPECT_TRUE(nixl::trace::loadTraceBackends({kUnloadableBackend}, "stub_agent").empty());
}

// Cross-thread correlation: push/pop must fan out to every backend, so a
// context set at the request boundary reaches all active backends.
TEST(Tracing, ContextFansOutToAllBackends) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto context = testContext(0x11);

    tracer->pushContext({context, nixl::trace::RequestId{0x1111}});
    tracer->popContext();

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, std::vector{context});
        EXPECT_EQ(log->pushedRequestIds, std::vector<std::uint64_t>{0x1111});
        EXPECT_EQ(log->poppedContexts, std::vector{context});
    }
}

// Scope is the RAII guard used at call sites: it pushes on
// construction and pops on destruction, across every backend.
TEST(Tracing, ScopeBalancesPushPop) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto context = testContext(0x12);

    {
        const nixl::trace::Scope scope(tracer.get(), {context, nixl::trace::RequestId{0x1212}});
        const auto span = tracer->beginSpan("op");
        for (const CallLog *log : {&a, &b}) {
            EXPECT_EQ(log->activeContexts, std::vector{context});
            EXPECT_TRUE(log->poppedContexts.empty());
        }
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, std::vector{context});
        EXPECT_EQ(log->pushedRequestIds, std::vector<std::uint64_t>{0x1212});
        EXPECT_EQ(log->poppedContexts, std::vector{context});
        EXPECT_TRUE(log->activeContexts.empty());
    }
}

TEST(Tracing, ScopeSkipsAnUnsampledContext) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto unsampled = testContext(0x13, 0x02);
    ASSERT_TRUE(unsampled.valid());
    ASSERT_FALSE(unsampled.sampled());

    { const nixl::trace::Scope scope(tracer.get(), {unsampled, nixl::trace::RequestId{0x1313}}); }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_TRUE(log->pushedContexts.empty());
        EXPECT_TRUE(log->poppedContexts.empty());
    }
}

TEST(Tracing, ScopePushesOnlyASampledContext) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto context = testContext(0x18);
    const nixl::trace::TraceContext no_context;

    { const nixl::trace::Scope scope(tracer.get(), {context, nixl::trace::RequestId{0}}); }
    { const nixl::trace::Scope scope(tracer.get(), {no_context, nixl::trace::RequestId{0x1818}}); }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, std::vector{context});
        EXPECT_EQ(log->pushedRequestIds, std::vector<std::uint64_t>{0});
        EXPECT_EQ(log->poppedContexts, std::vector{context});
    }
}

// A null tracer (no active backend) must make Scope a safe no-op.
TEST(Tracing, ScopeNullTracerIsInert) {
    nixl::trace::Tracer *tracer = nullptr;
    { const nixl::trace::Scope scope(tracer, {testContext(0x14), nixl::trace::RequestId{0x1414}}); }
    SUCCEED();
}

TEST(Tracing, ScopeWithoutContextOrRequestIdIsInert) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    auto no_trace_id = testContext(0x15);
    no_trace_id.traceId->fill(0);
    auto no_span_id = testContext(0x15);
    no_span_id.spanId->fill(0);

    for (const auto &context : {nixl::trace::TraceContext{}, no_trace_id, no_span_id}) {
        const nixl::trace::Scope scope(tracer.get(), {context, nixl::trace::RequestId{0}});
        const auto span = tracer->beginSpan("op");
    }

    EXPECT_EQ(a.spansBegun, 3);
    for (const CallLog *log : {&a, &b}) {
        EXPECT_TRUE(log->pushedContexts.empty());
        EXPECT_TRUE(log->poppedContexts.empty());
    }
}

TEST(Tracing, NestedScopesPopInReverseOrder) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const auto outer = testContext(0x16);
    const auto inner = testContext(0x17);

    {
        const nixl::trace::Scope outer_scope(tracer.get(), {outer, nixl::trace::RequestId{0x1616}});
        {
            const nixl::trace::Scope inner_scope(tracer.get(),
                                                 {inner, nixl::trace::RequestId{0x1717}});
            EXPECT_EQ(a.activeContexts, (std::vector{outer, inner}));
        }
        EXPECT_EQ(a.activeContexts, std::vector{outer});
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->pushedContexts, (std::vector{outer, inner}));
        EXPECT_EQ(log->pushedRequestIds, (std::vector<std::uint64_t>{0x1616, 0x1717}));
        EXPECT_EQ(log->poppedContexts, (std::vector{inner, outer}));
        EXPECT_TRUE(log->activeContexts.empty());
    }
}

TEST(Tracing, ScopeIsCurrentUntilItEnds) {
    CallLog a, b;
    auto tracer = makeMockTracer(a, b);
    const nixl::trace::TraceIdentity outer{testContext(0x19), nixl::trace::RequestId{0x1919}, 1};
    const nixl::trace::TraceIdentity inner{testContext(0x1a), nixl::trace::RequestId{0x1a1a}, 2};

    const auto sink = tracer->makeSink("test", {});

    EXPECT_EQ(sink->currentIdentity(), nullptr);
    {
        const nixl::trace::Scope outer_scope(tracer.get(), outer);
        ASSERT_NE(sink->currentIdentity(), nullptr);
        EXPECT_EQ(*sink->currentIdentity()->requestId, 0x1919u);
        {
            const nixl::trace::Scope inner_scope(tracer.get(), inner);
            ASSERT_NE(sink->currentIdentity(), nullptr);
            EXPECT_EQ(*sink->currentIdentity()->requestId, 0x1a1au);
            EXPECT_EQ(sink->currentIdentity()->execution, 2u);
            {
                const nixl::trace::Scope untraced(nullptr, outer);
                EXPECT_EQ(*sink->currentIdentity()->requestId, 0x1a1au);
            }
        }
        ASSERT_NE(sink->currentIdentity(), nullptr);
        EXPECT_EQ(sink->currentIdentity()->context, outer.context);
        std::thread([&sink] { EXPECT_EQ(sink->currentIdentity(), nullptr); }).join();
    }
    EXPECT_EQ(sink->currentIdentity(), nullptr);
}

TEST(Tracing, RequestDrawsItsIdentityFromItsTracer) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);

    const nixl::trace::CallScope call(tracer.get(), nullptr);
    const nixlXferReqH request(
        "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});

    const auto identity = request.traceable().requestIdentity();
    EXPECT_TRUE(identity.context.valid());
    EXPECT_TRUE(identity.context.sampled());
    EXPECT_NE(*identity.requestId, 0u);
    EXPECT_EQ(identity.execution, 0u);
}

TEST(Tracing, RequestContextsAreDistinctAndStable) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    const nixl::trace::CallScope call(tracer.get(), nullptr);
    const nixlXferReqH first(
        "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});
    const nixlXferReqH second(
        "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});
    const auto first_context = first.traceable().requestIdentity().context;

    EXPECT_NE(first_context, second.traceable().requestIdentity().context);
    EXPECT_EQ(first.traceable().requestIdentity().context, first_context);
}

TEST(Tracing, RequestDefaultContextHasInertCorrelation) {
    const nixlXferReqH request(
        "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});

    EXPECT_FALSE(request.traceable().requestIdentity().context.valid());
}

static_assert(std::is_constructible_v<nixl::trace::CallScope,
                                      nixl::trace::Tracer *,
                                      const nixl_opt_args_t *>);
static_assert(!std::is_constructible_v<nixl::trace::CallScope, nixl::trace::Tracer *, nixl_blob_t>);

TEST(Tracing, RequestOutsideACallScopeIsUntraced) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    {
        const nixl::trace::CallScope call(tracer.get(), nullptr);
        const nixlXferReqH inside(
            "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});
        EXPECT_EQ(inside.traceable().tracer(), tracer.get());
    }

    const nixlXferReqH request(
        "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});

    EXPECT_EQ(request.traceable().tracer(), nullptr);
    EXPECT_FALSE(request.traceable().requestIdentity().context.valid());
}

TEST(Tracing, CallScopesNest) {
    CallLog a, b, c, d;
    const auto outer_tracer = makeMockTracer(a, b, 1.0);
    const auto inner_tracer = makeMockTracer(c, d, 1.0);
    const nixl::trace::CallScope outer(outer_tracer.get(), nullptr);
    {
        const nixl::trace::CallScope inner(inner_tracer.get(), nullptr);
        const nixl::trace::CallScope untraced(nullptr, nullptr);
        const nixlXferReqH request(
            "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});
        EXPECT_EQ(request.traceable().tracer(), inner_tracer.get());
    }

    const nixlXferReqH request(
        "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});
    EXPECT_EQ(request.traceable().tracer(), outer_tracer.get());
}

TEST(Tracing, RequestJoinsTheCallScopesContext) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0.0);
    const auto parent = testContext(0x3f, 0x01);
    nixl_opt_args_t options;
    options.traceContext = nixl::trace::formatTraceparent(parent);

    const nixl::trace::CallScope call(tracer.get(), &options);
    const nixlXferReqH request(
        "remote", NIXL_WRITE, DRAM_SEG, DRAM_SEG, 0, nixl_remote_section_weak_t{});

    const auto identity = request.traceable().requestIdentity();
    EXPECT_EQ(identity.context.traceId, parent.traceId);
    EXPECT_TRUE(identity.context.sampled());
}

TEST(Tracing, SampledRequestTakesTheCallsBaggageAndItsPostsInheritIt) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    nixl_opt_args_t options;
    options.traceBaggage = "tenant=a,job=7";
    const nixl::trace::CallScope call(tracer.get(), &options);
    nixl::trace::Traceable traced;

    EXPECT_EQ(traced.requestBaggage(), "tenant=a,job=7");
    static_cast<void>(traced.post());
    EXPECT_EQ(traced.postBaggage(), "tenant=a,job=7");
    static_cast<void>(traced.post(nixl::trace::noSuppliedContext, "step=2"));
    EXPECT_EQ(traced.postBaggage(), "step=2");
    static_cast<void>(traced.post());
    EXPECT_EQ(traced.postBaggage(), "tenant=a,job=7");
}

TEST(Tracing, UnsampledRequestKeepsNoBaggage) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0.0);
    nixl_opt_args_t options;
    options.traceBaggage = "tenant=a";
    const nixl::trace::CallScope call(tracer.get(), &options);
    nixl::trace::Traceable traced;

    EXPECT_TRUE(traced.requestBaggage().empty());
    static_cast<void>(traced.post(nixl::trace::noSuppliedContext, "step=2"));
    EXPECT_TRUE(traced.postBaggage().empty());
}

TEST(Tracing, BaggageNixlCannotCarryIsIgnoredOnce) {
    gtest::LogMessageCounter ignored("ignoring a supplied baggage");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    nixl_opt_args_t options;
    options.traceBaggage = std::string(nixl::trace::maxBaggageSize + 1, 'k');
    const nixl::trace::CallScope call(tracer.get(), &options);
    nixl::trace::Traceable traced;

    EXPECT_TRUE(traced.requestBaggage().empty());
    static_cast<void>(traced.post(nixl::trace::noSuppliedContext, "tab\there"));
    EXPECT_TRUE(traced.postBaggage().empty());
    const std::string largest(nixl::trace::maxBaggageSize, 'k');
    static_cast<void>(traced.post(nixl::trace::noSuppliedContext, largest));
    EXPECT_EQ(traced.postBaggage(), largest);
    EXPECT_EQ(ignored.count(), 1u);
}

TEST(Tracing, RequestPostAndNotificationSpansRecordTheirBaggage) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    nixl_opt_args_t options;
    options.traceBaggage = "tenant=a";
    const nixl::trace::CallScope call(tracer.get(), &options);
    nixl::trace::Traceable traced;
    nixl_opt_args_t post_args;
    post_args.traceBaggage = "step=2";
    nixl_opt_args_t notif_options;
    notif_options.traceContext = nixl::trace::formatTraceparent(testContext(0x64));
    notif_options.traceBaggage = "notif=1";

    {
        NIXL_TRACE_REQUEST_SPAN(request, traced, nixl::trace::Record::CreateXferReq);
        EXPECT_TRUE(request.active());
    }
    {
        NIXL_TRACE_POST_SPAN(post, traced, &post_args, nixl::trace::Record::PostXferReqWrite);
        EXPECT_TRUE(post.active());
    }
    {
        NIXL_TRACE_SCOPE_UNDER(notif, tracer.get(), &notif_options, nixl::trace::Record::GenNotif);
        EXPECT_TRUE(notif.active());
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->strAttrs,
                  (std::vector<std::pair<std::string, std::string>>{{"nixl.baggage", "tenant=a"},
                                                                    {"nixl.baggage", "step=2"},
                                                                    {"nixl.baggage", "notif=1"}}));
    }
}

TEST(Tracing, ScopesMakeTheirBaggageCurrentForTheBackend) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    const auto sink = tracer->makeSink("test", {});
    nixl_opt_args_t options;
    options.traceBaggage = "tenant=a";
    const nixl::trace::CallScope call(tracer.get(), &options);
    nixl::trace::Traceable traced;

    {
        NIXL_TRACE_POST_SPAN(post, traced, nullptr, nixl::trace::Record::PostXferReqWrite);
        EXPECT_EQ(sink->currentBaggage(), "tenant=a");
        EXPECT_EQ(nixlBackendPostTrace{sink.get()}.baggage(), "tenant=a");
    }
    EXPECT_TRUE(sink->currentBaggage().empty());
    EXPECT_TRUE(nixlBackendPostTrace{sink.get()}.baggage().empty());
}

TEST(Tracing, ActiveTracerGeneratesValidDistinctContexts) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);

    const auto first = nixl::trace::generateRequestIdentity(tracer.get()).context;
    const auto second = nixl::trace::generateRequestIdentity(tracer.get()).context;

    EXPECT_TRUE(first.valid());
    EXPECT_TRUE(second.valid());
    EXPECT_NE(first.spanId, second.spanId);
}

// The tracer carries the sampling ratio to the generation call sites, which only
// have the tracer pointer.
TEST(Tracing, TracerSampleRatioReachesGeneratedContexts) {
    CallLog a, b;
    const auto unsampled_tracer = makeMockTracer(a, b);
    EXPECT_EQ(unsampled_tracer->sampleRatio(), 0.0);
    const auto unsampled = nixl::trace::generateRequestIdentity(unsampled_tracer.get());
    EXPECT_FALSE(unsampled.context.valid());
    EXPECT_EQ(*unsampled.requestId, 0u);

    CallLog c, d;
    const auto sampled_tracer = makeMockTracer(c, d, /*sample_ratio=*/1.0);
    EXPECT_EQ(sampled_tracer->sampleRatio(), 1.0);

    const auto context = nixl::trace::generateRequestIdentity(sampled_tracer.get()).context;
    EXPECT_TRUE(context.valid());
    EXPECT_TRUE(context.sampled());
}

TEST(Tracing, ContextGenerationFailureIsUntracedAndLoggedOncePerTracer) {
    gtest::LogMessageCounter failures("could not generate a trace context");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, failRandomBytes);

    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(nixl::trace::generateRequestIdentity(tracer.get()).context,
                  nixl::trace::TraceContext{});
    }
    EXPECT_EQ(failures.count(), 1u);

    CallLog c, d;
    const auto other_tracer = makeMockTracer(c, d, 1.0, failRandomBytes);
    EXPECT_EQ(nixl::trace::generateRequestIdentity(other_tracer.get()).context,
              nixl::trace::TraceContext{});
    EXPECT_EQ(failures.count(), 2u);
}

TEST(Tracing, RequestIdentityDrawsItsIdsAgainWhileZero) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, zeroFirstRandomBytes);
    zero_draws_left = 2;

    const nixl::trace::TraceIdentity identity = nixl::trace::generateRequestIdentity(tracer.get());

    std::array<std::uint8_t, 16> trace_id{};
    trace_id.fill(0x5a);
    std::array<std::uint8_t, 8> span_id{};
    span_id.fill(0x5a);
    EXPECT_EQ(*identity.context.traceId, trace_id);
    EXPECT_EQ(*identity.context.spanId, span_id);
    EXPECT_EQ(*identity.requestId, 0x5a5a5a5a5a5a5a5aULL);
}

TEST(Tracing, RequestIdentityTakesItsIdsFromOneDraw) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, countedSequenceRandomBytes);
    random_draws = 0;

    const nixl::trace::TraceIdentity identity = nixl::trace::generateRequestIdentity(tracer.get());

    EXPECT_EQ(random_draws, 1u);
    std::array<std::uint8_t, 32> drawn{};
    std::iota(drawn.begin(), drawn.end(), std::uint8_t{1});
    std::uint64_t request_id = 0;
    std::memcpy(&request_id, drawn.data() + 24, sizeof(request_id));
    EXPECT_TRUE(std::ranges::equal(*identity.context.traceId, std::span(drawn).first(16)));
    EXPECT_TRUE(std::ranges::equal(*identity.context.spanId, std::span(drawn).subspan(16, 8)));
    EXPECT_EQ(*identity.requestId,
              request_id & static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
    EXPECT_EQ(identity.execution, 0u);
    EXPECT_TRUE(identity.context.sampled());
}

TEST(Tracing, RequestIdentityDrawsOnlyAZeroIdAgain) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, zeroSpanIdOnceRandomBytes);
    random_draws = 0;

    const nixl::trace::TraceIdentity identity = nixl::trace::generateRequestIdentity(tracer.get());

    std::array<std::uint8_t, 16> trace_id{};
    trace_id.fill(0x11);
    std::array<std::uint8_t, 8> span_id{};
    span_id.fill(0x5a);
    EXPECT_EQ(random_draws, 2u);
    EXPECT_EQ(*identity.context.traceId, trace_id);
    EXPECT_EQ(*identity.context.spanId, span_id);
    EXPECT_EQ(*identity.requestId, 0x2222222222222222ULL);
    EXPECT_TRUE(identity.context.sampled());
}

TEST(Tracing, RequestIdentityIsNoneWithoutTracerOrSource) {
    gtest::LogMessageCounter failures("could not generate a trace context");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, failRandomBytes);

    for (const nixl::trace::Tracer *source :
         std::initializer_list<const nixl::trace::Tracer *>{nullptr, tracer.get(), tracer.get()}) {
        const nixl::trace::TraceIdentity identity = nixl::trace::generateRequestIdentity(source);
        EXPECT_EQ(identity.context, nixl::trace::TraceContext{});
        EXPECT_EQ(*identity.requestId, 0u);
        EXPECT_EQ(identity.execution, 0u);
    }
    EXPECT_EQ(failures.count(), 1u);
}

TEST(Tracing, PostContextDrawsOnlyItsSpanIdAgainWhileZero) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, zeroFirstRandomBytes);
    zero_draws_left = 1;
    const auto request = testContext(0x21, 0x03);
    auto expected = request;
    expected.spanId->fill(0x5a);
    expected.parentSpanId = request.spanId;

    EXPECT_EQ(nixl::trace::generatePostContext(request, tracer.get()), expected);
}

TEST(Tracing, PostContextIsUnsampledWhenItsSourceFails) {
    gtest::LogMessageCounter failures("could not generate a trace context");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, failRandomBytes);
    const auto request = testContext(0x22, 0x03);
    auto unsampled = request;
    unsampled.flags = 0x02;

    EXPECT_EQ(nixl::trace::generatePostContext(request, nullptr), request);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(nixl::trace::generatePostContext(request, tracer.get()), unsampled);
    }
    EXPECT_EQ(failures.count(), 1u);
}

TEST(Tracing, TraceablePostIsUntracedWhenItsSpanIdCannotBeDrawn) {
    gtest::LogMessageCounter failures("could not generate a trace context");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, failRandomBytes);
    const auto request = testContext(0x58);
    const auto application = testContext(0x59);
    nixl::trace::Traceable traced{tracer.get(), {request, nixl::trace::RequestId{0x58}}};
    auto unsampled_request = request;
    unsampled_request.flags = 0x00;
    auto unsampled_application = application;
    unsampled_application.flags = 0x00;

    EXPECT_EQ(traced.post().context, unsampled_request);
    EXPECT_EQ(traced.post(nixl::trace::formatTraceparent(application)).context,
              unsampled_application);
    EXPECT_EQ(failures.count(), 1u);
    {
        NIXL_TRACE_POST_SPAN(post, traced, nullptr, nixl::trace::Record::PostXferReqWrite);
        EXPECT_FALSE(post.active());
    }
    for (const CallLog *log : {&a, &b}) {
        EXPECT_TRUE(log->contextSpans.empty());
        EXPECT_TRUE(log->pushedContexts.empty());
    }
}

TEST(Tracing, RequestIdFitsAnInt64) {
    const nixl::trace::CompositeTracer tracer{
        std::vector<std::unique_ptr<nixl::trace::TraceBackend>>{},
        1.0,
        [](std::span<std::uint8_t> bytes) { std::ranges::fill(bytes, std::uint8_t{0xff}); }};

    EXPECT_EQ(*nixl::trace::generateRequestIdentity(&tracer).requestId,
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
}

TEST(Tracing, RequestIdentityJoinsASuppliedTraceparent) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0.0, zeroFirstRandomBytes);
    zero_draws_left = 0;
    const auto parent = testContext(0x41, 0x01);

    const nixl::trace::TraceIdentity identity =
        nixl::trace::generateRequestIdentity(tracer.get(), nixl::trace::formatTraceparent(parent));

    std::array<std::uint8_t, 8> span_id{};
    span_id.fill(0x5a);
    EXPECT_EQ(identity.context.traceId, parent.traceId);
    EXPECT_EQ(*identity.context.spanId, span_id);
    EXPECT_EQ(identity.context.flags, 0x01);
    EXPECT_EQ(identity.context.parentSpanId, parent.spanId);
    EXPECT_EQ(*identity.requestId, 0x5a5a5a5a5a5a5a5aULL);
    EXPECT_EQ(identity.execution, 0u);
}

TEST(Tracing, RequestIdentityJoinsASuppliedRecord) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    const auto parent = testContext(0x42, 0x03);
    std::array<std::uint8_t, nixl::trace::traceContextWireSize> record{};
    ASSERT_TRUE(nixl::trace::encodeTraceContext(parent, record).has_value());

    const nixl::trace::TraceIdentity identity = nixl::trace::generateRequestIdentity(
        tracer.get(), {reinterpret_cast<const char *>(record.data()), record.size()});

    EXPECT_EQ(identity.context.traceId, parent.traceId);
    EXPECT_NE(identity.context.spanId, parent.spanId);
    EXPECT_EQ(identity.context.flags, 0x03);
    EXPECT_TRUE(identity.context.sampled());
    EXPECT_EQ(identity.context.parentSpanId, parent.spanId);
    EXPECT_NE(*identity.requestId, 0u);
}

TEST(Tracing, RequestIdentityKeepsOnlyTheFlagsItDefines) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);

    const nixl::trace::TraceIdentity identity = nixl::trace::generateRequestIdentity(
        tracer.get(), "00-43434343434343434343434343434343-4343434343434343-ff");

    EXPECT_EQ(identity.context.traceId, testContext(0x43).traceId);
    EXPECT_EQ(identity.context.flags, 0x03);
}

TEST(Tracing, UnparsableParentIsIgnoredAndLoggedOncePerTracer) {
    gtest::LogMessageCounter ignored("ignoring a supplied trace context");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    std::string later_version(nixl::trace::traceContextWireSize, '\x43');
    later_version.front() = '\x02';

    EXPECT_EQ(*nixl::trace::generateRequestIdentity(nullptr, "neither").requestId, 0u);
    EXPECT_EQ(ignored.count(), 0u);
    for (const nixl_blob_t &parent :
         {nixl_blob_t("neither"),
          nixl_blob_t("00-00000000000000000000000000000000-4444444444444444-01"),
          later_version}) {
        const nixl::trace::TraceIdentity identity =
            nixl::trace::generateRequestIdentity(tracer.get(), parent);
        EXPECT_TRUE(identity.context.valid());
        EXPECT_TRUE(identity.context.sampled());
        EXPECT_EQ(*identity.context.parentSpanId, (std::array<std::uint8_t, 8>{}));
        EXPECT_NE(*identity.requestId, 0u);
    }
    EXPECT_EQ(ignored.count(), 1u);

    CallLog c, d;
    const auto other_tracer = makeMockTracer(c, d, 1.0);
    EXPECT_TRUE(
        nixl::trace::generateRequestIdentity(other_tracer.get(), "neither").context.valid());
    EXPECT_EQ(ignored.count(), 2u);
}

TEST(Tracing, TraceableDrawsEachPostASpanIdUnderItsRequest) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, zeroFirstRandomBytes);
    zero_draws_left = 0;
    const auto sampled = testContext(0x31, 0x03);
    const auto unsampled = testContext(0x32, 0x02);
    nixl::trace::Traceable sampled_trace{tracer.get(), {sampled, nixl::trace::RequestId{0x31}}};
    nixl::trace::Traceable unsampled_trace{tracer.get(), {unsampled, nixl::trace::RequestId{0x32}}};
    auto drawn = sampled;
    drawn.spanId->fill(0x5a);
    drawn.parentSpanId = sampled.spanId;

    EXPECT_EQ(sampled_trace.post().context, drawn);
    EXPECT_EQ(unsampled_trace.post().context, unsampled);
    EXPECT_EQ(sampled_trace.post().context, drawn);
    EXPECT_EQ(unsampled_trace.post().context, unsampled);
}

TEST(Tracing, TraceablePostTakesTheNextExecution) {
    const auto context = testContext(0x33);
    nixl::trace::Traceable traced{nullptr, {context, nixl::trace::RequestId{0x33}}};
    nixl::trace::Traceable untraced{nullptr, {}};

    EXPECT_EQ(traced.requestIdentity().context, context);
    EXPECT_EQ(traced.requestIdentity().execution, 0u);
    EXPECT_EQ(traced.post().execution, 1u);
    EXPECT_EQ(traced.post().execution, 2u);
    EXPECT_EQ(untraced.post().execution, 0u);

    EXPECT_EQ(traced.postIdentity().execution, 2u);
    EXPECT_EQ(untraced.postIdentity().execution, 0u);
}

TEST(Tracing, TraceablePostIdentityIsItsLatestPosts) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0, zeroFirstRandomBytes);
    zero_draws_left = 0;
    const auto context = testContext(0x35, 0x03);
    nixl::trace::Traceable traced{tracer.get(), {context, nixl::trace::RequestId{0x35}}};
    auto drawn = context;
    drawn.spanId->fill(0x5a);
    drawn.parentSpanId = context.spanId;

    const auto prepped = traced.requestIdentity();
    const auto first = traced.post();
    const auto second = traced.post();
    const auto posted = traced.postIdentity();

    EXPECT_EQ(prepped.context, context);
    EXPECT_EQ(*prepped.requestId, 0x35u);
    EXPECT_EQ(prepped.execution, 0u);
    EXPECT_EQ(first.context, drawn);
    EXPECT_EQ(*first.requestId, 0x35u);
    EXPECT_EQ(first.execution, 1u);
    EXPECT_EQ(second.context, drawn);
    EXPECT_EQ(second.execution, 2u);
    EXPECT_EQ(posted.context, drawn);
    EXPECT_EQ(*posted.requestId, 0x35u);
    EXPECT_EQ(posted.execution, 2u);
}

TEST(Tracing, TraceablePostJoinsASuppliedContextForThatPostOnly) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0.0, zeroFirstRandomBytes);
    zero_draws_left = 0;
    const auto request = testContext(0x51);
    const auto application = testContext(0x52);
    nixl::trace::Traceable traced{tracer.get(), {request, nixl::trace::RequestId{0x51}}};
    auto under_application = application;
    under_application.spanId->fill(0x5a);
    under_application.parentSpanId = application.spanId;
    auto under_request = request;
    under_request.spanId->fill(0x5a);
    under_request.parentSpanId = request.spanId;

    EXPECT_EQ(traced.post(nixl::trace::formatTraceparent(application)).context, under_application);
    EXPECT_EQ(traced.post().context, under_request);
    EXPECT_EQ(traced.postIdentity().execution, 2u);
}

TEST(Tracing, TraceablePostTakesASuppliedContextsSampledFlag) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0.0, zeroFirstRandomBytes);
    zero_draws_left = 0;
    const auto unsampled_request = testContext(0x53, 0x02);
    const auto sampled_application = testContext(0x55, 0x01);
    const auto unsampled_application = testContext(0x56, 0x00);
    nixl::trace::Traceable unsampled{tracer.get(),
                                     {unsampled_request, nixl::trace::RequestId{0x53}}};
    nixl::trace::Traceable sampled{tracer.get(),
                                   {testContext(0x54, 0x01), nixl::trace::RequestId{0x54}}};

    const auto joined = unsampled.post(nixl::trace::formatTraceparent(sampled_application)).context;
    EXPECT_TRUE(joined.sampled());
    EXPECT_EQ(joined.traceId, sampled_application.traceId);
    EXPECT_EQ(joined.parentSpanId, sampled_application.spanId);
    EXPECT_EQ(unsampled.post().context, unsampled_request);

    EXPECT_EQ(sampled.post(nixl::trace::formatTraceparent(unsampled_application)).context,
              unsampled_application);
    EXPECT_TRUE(sampled.post().context.sampled());
}

TEST(Tracing, TraceablePostIgnoresAnUnparsableContext) {
    gtest::LogMessageCounter ignored("ignoring a supplied trace context");
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 0.0, zeroFirstRandomBytes);
    zero_draws_left = 0;
    const auto request = testContext(0x57);
    nixl::trace::Traceable traced{tracer.get(), {request, nixl::trace::RequestId{0x57}}};
    auto under_request = request;
    under_request.spanId->fill(0x5a);
    under_request.parentSpanId = request.spanId;

    EXPECT_EQ(traced.post("neither").context, under_request);
    EXPECT_EQ(traced.post("neither").context, under_request);
    EXPECT_EQ(ignored.count(), 1u);
}

static_assert(sizeof(nixl::trace::AttributeKey) == 2);
static_assert(sizeof(nixl::trace::Attribute) == 32);
static_assert(nixl::trace::toStringView(nixl::trace::AttributeKey::RemoteAgent) == "remote_agent");
static_assert(nixl::trace::toStringView(nixl::trace::AttributeKey::NixlStatus) == "nixl.status");
static_assert(sizeof(nixl::trace::Name) == 16);
static_assert(nixl::trace::Name(nixl::trace::Record::Submit).record() ==
              nixl::trace::Record::Submit);
static_assert(nixl::trace::Name(nixl::trace::Record::Submit).view() == "nixl::submit");
static_assert(!nixl::trace::Name("nixl::submit").record().has_value());
static_assert(nixl::trace::Name("rail0.write").view() == "rail0.write");
static_assert(nixl::trace::kindOf(nixl::trace::Record::RegisterMem) == nixl::trace::Kind::MemoryW);
static_assert(nixl::trace::kindOf(nixl::trace::Record::PrepMemView) == nixl::trace::Kind::MemoryR);
static_assert(nixl::trace::kindOf(nixl::trace::Record::PostXferReqWrite) ==
              nixl::trace::Kind::CommSend);
static_assert(nixl::trace::kindOf(nixl::trace::Record::PostXferReqRead) ==
              nixl::trace::Kind::CommRecv);

TEST(Tracing, EveryRecordHasItsOwnName) {
    std::set<std::string_view> names;
    for (std::size_t i = 0; i < nixl::trace::recordCount; ++i) {
        const std::string_view name =
            nixl::trace::toStringView(static_cast<nixl::trace::Record>(i));
        EXPECT_TRUE(name.starts_with("nixl::")) << name;
        names.insert(name);
    }
    EXPECT_EQ(names.size(), nixl::trace::recordCount);
}

TEST(Tracing, TraceableRecordsItsEventsInItsLatestPost) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::Traceable traced{tracer.get(), {testContext(0x37), nixl::trace::RequestId{0x37}}};
    const auto post = traced.post().context;

    traced.trace(nixl::trace::Record::XferComplete);
    traced.trace(nixl::trace::Record::XferError,
                 {nixl::trace::AttributeKey::NixlStatus, NIXL_ERR_BACKEND});

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), 2u);
        EXPECT_EQ(log->emittedEvents[0].name, "nixl::xfer.complete");
        EXPECT_EQ(log->emittedEvents[0].kind, nixl::trace::Kind::Metadata);
        EXPECT_EQ(log->emittedEvents[0].context, post);
        EXPECT_EQ(log->emittedEvents[0].requestId, 0x37u);
        EXPECT_EQ(log->emittedEvents[0].attrs,
                  (attr_list_t{{"nixl.xfer_request_id", std::int64_t{0x37}},
                               {"nixl.xfer_execution", std::int64_t{1}}}));
        EXPECT_EQ(log->emittedEvents[1].name, "nixl::xfer.error");
        EXPECT_EQ(log->emittedEvents[1].attrs,
                  (attr_list_t{{"nixl.xfer_request_id", std::int64_t{0x37}},
                               {"nixl.xfer_execution", std::int64_t{1}},
                               {"nixl.status", std::int64_t{NIXL_ERR_BACKEND}}}));
    }
}

TEST(Tracing, TraceableRecordsItsRequestSpanWhenItEnds) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    const auto request = testContext(0x3b);
    const nixlTime::us_t before = nixlTime::getUs();
    {
        nixl::trace::Traceable traced{tracer.get(), {request, nixl::trace::RequestId{0x3b}}};
        EXPECT_EQ(traced.post().execution, 1u);
        EXPECT_EQ(traced.post().execution, 2u);
        EXPECT_TRUE(a.emittedContextSpans.empty());
        EXPECT_TRUE(b.emittedContextSpans.empty());
    }
    const nixlTime::us_t after = nixlTime::getUs();

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedContextSpans.size(), 1u);
        const EmittedRecord &span = log->emittedContextSpans[0];
        EXPECT_EQ(span.name, "nixl::xferReq");
        EXPECT_EQ(span.kind, nixl::trace::Kind::Generic);
        EXPECT_EQ(span.context, request);
        EXPECT_EQ(span.requestId, 0x3bu);
        ASSERT_EQ(span.times.size(), 2u);
        EXPECT_LE(before, span.times[0]);
        EXPECT_LE(span.times[0], span.times[1]);
        EXPECT_LE(span.times[1], after);
        EXPECT_EQ(span.attrs,
                  (attr_list_t{{"nixl.xfer_request_id", std::int64_t{0x3b}},
                               {"nixl.xfer_execution", std::int64_t{2}}}));
        EXPECT_TRUE(log->emittedEvents.empty());
        EXPECT_TRUE(log->emittedSpans.empty());
    }
}

TEST(Tracing, TraceableRecordsNothingUnsampledOrWithoutATracer) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    {
        const nixl::trace::Traceable unsampled{
            tracer.get(), {testContext(0x39, 0x02), nixl::trace::RequestId{0x39}}};
        const nixl::trace::Traceable untraced{nullptr,
                                              {testContext(0x3a), nixl::trace::RequestId{0x3a}}};

        unsampled.trace(nixl::trace::Record::XferComplete);
        untraced.trace(nixl::trace::Record::XferError,
                       {nixl::trace::AttributeKey::NixlStatus, NIXL_ERR_BACKEND});
    }

    for (const CallLog *log : {&a, &b}) {
        EXPECT_TRUE(log->emittedEvents.empty());
        EXPECT_TRUE(log->emittedContextSpans.empty());
    }
}

TEST(Tracing, TraceableEndsDespiteABackendThatThrows) {
    const auto tracer = makeThrowingTracer();
    const nixl::trace::TraceIdentity request{testContext(0x3c), nixl::trace::RequestId{0x3c}};
    EXPECT_NO_THROW(nixl::trace::Traceable(tracer.get(), request));
}

TEST(Tracing, TraceInfoWithoutATracerIsNotFound) {
    const nixl::trace::Traceable traceable(nullptr,
                                           {testContext(0x3e), nixl::trace::RequestId{0x3e}});
    nixl_xfer_trace_info_t info;
    EXPECT_EQ(nixl::trace::traceInfo(traceable, info), NIXL_ERR_NOT_FOUND);
}

TEST(Tracing, TraceableRecordsAnEventWithSeveralAttributes) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    const nixl::trace::Traceable traced(tracer.get(),
                                        {testContext(0x47), nixl::trace::RequestId{0x4747}});
    const nixl::trace::Attribute attrs[] = {
        {nixl::trace::AttributeKey::NixlStatus, NIXL_ERR_REMOTE_DISCONNECT},
        {nixl::trace::AttributeKey::NixlBackendStatusCode, std::int64_t{-25}},
        {nixl::trace::AttributeKey::NixlBackendStatusName, std::string_view{"reset"}}};

    traced.trace(nixl::trace::Record::XferError, attrs);

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), 1u);
        EXPECT_EQ(log->emittedEvents[0].name, "nixl::xfer.error");
        EXPECT_EQ(log->emittedEvents[0].attrs,
                  (attr_list_t{{"nixl.xfer_request_id", std::int64_t{0x4747}},
                               {"nixl.xfer_execution", std::int64_t{0}},
                               {"nixl.status", std::int64_t{NIXL_ERR_REMOTE_DISCONNECT}},
                               {"nixl.backend_status_code", std::int64_t{-25}},
                               {"nixl.backend_status_name", std::string("reset")}}));
    }
}

TEST(Tracing, TraceableRecordsOnlyWhenSampledWithATracer) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    EXPECT_TRUE(
        nixl::trace::Traceable(tracer.get(), {testContext(0x48), nixl::trace::RequestId{0x4848}})
            .recording());
    EXPECT_FALSE(nixl::trace::Traceable(tracer.get(),
                                        {testContext(0x49, 0x00), nixl::trace::RequestId{0x4949}})
                     .recording());
    EXPECT_FALSE(
        nixl::trace::Traceable(nullptr, {testContext(0x4a), nixl::trace::RequestId{0x4a4a}})
            .recording());
}

// A phase recorded through the plugin-facing sink reaches every enabled trace
// backend, tagged with the recording backend and the plugin's timestamp.
TEST(TracePhaseSink, RecordedPhaseReachesEveryBackend) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};

    sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, 1234, {}, {}, {});

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
        sink.recordPhase(phase, 1, {}, {}, {});
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

    sink.recordPhase(nixl_trace_phase_t::OTHER, 1, {}, {}, "post_write");
    sink.recordPhase(nixl_trace_phase_t::OTHER, 2, {}, {}, {});

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

    sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, 5, {}, {}, "rail0");

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

    sink.recordPhase(nixl_trace_phase_t::SUBMIT, 5, {}, {}, {});

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

    const nixlBackendTraceAttr attrs[] = {{nixl::trace::AttributeKey::RemoteAgent, "0"},
                                          {nixl::trace::AttributeKey::NixlRemoteAgent, "write"}};
    sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, 7, {}, attrs, {});

    ASSERT_EQ(a.emittedEvents.size(), 1u);
    EXPECT_EQ(a.emittedEvents[0].attrs,
              (attr_list_t{{"nixl.backend", std::string("LIBFABRIC")},
                           {"nixl.phase.timestamp_us", std::int64_t{7}},
                           {"remote_agent", std::string("0")},
                           {"nixl.remote_agent", std::string("write")}}));
}

TEST(TracePhaseSink, ValidContextIsTheRecordsOwn) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    constexpr std::string_view traceparent =
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    const auto context = nixl::trace::parseTraceparent(traceparent);
    ASSERT_TRUE(context.has_value());
    const nixl::trace::TraceContext zeroed;

    sink.recordPhase(nixl_trace_phase_t::NOTIF_RECEIVED, 1, {}, {}, {});
    sink.recordPhase(nixl_trace_phase_t::NOTIF_RECEIVED, 2, {zeroed}, {}, {});
    sink.recordPhase(nixl_trace_phase_t::NOTIF_RECEIVED, 3, {*context}, {}, {});

    for (const CallLog *log : {&a, &b}) {
        EXPECT_TRUE(log->pushedContexts.empty());
        ASSERT_EQ(log->emittedEvents.size(), 3u);
        EXPECT_EQ(log->emittedEvents[0].context, nixl::trace::TraceContext{});
        EXPECT_EQ(log->emittedEvents[1].context, zeroed);
        EXPECT_EQ(log->emittedEvents[2].context, *context);
        EXPECT_EQ(log->emittedEvents[0].attrs.size(), 2u);
        EXPECT_EQ(log->emittedEvents[1].attrs.size(), 2u);
        EXPECT_EQ(log->emittedEvents[2].attrs.size(), 2u);
    }
}

TEST(TracePhaseSink, RequestIdAndExecutionAreRecordedWhenNonZero) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    constexpr std::uint64_t request_id = 0x7edcba9876543210ULL;

    sink.recordPhase(
        nixl_trace_phase_t::SUBMIT, 1, {{}, nixl::trace::RequestId{request_id}, 2}, {}, {});
    sink.recordPhase(
        nixl_trace_phase_t::SUBMIT, 2, {{}, nixl::trace::RequestId{request_id}, 0}, {}, {});
    sink.recordPhase(nixl_trace_phase_t::SUBMIT, 3, {}, {}, {});

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), 3u);
        EXPECT_EQ(log->emittedEvents[0].requestId, request_id);
        EXPECT_EQ(log->emittedEvents[1].requestId, request_id);
        EXPECT_EQ(log->emittedEvents[2].requestId, 0u);
        EXPECT_EQ(log->emittedEvents[0].attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", std::int64_t{1}},
                               {"nixl.xfer_request_id", static_cast<std::int64_t>(request_id)},
                               {"nixl.xfer_execution", std::int64_t{2}}}));
        EXPECT_EQ(log->emittedEvents[1].attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", std::int64_t{2}},
                               {"nixl.xfer_request_id", static_cast<std::int64_t>(request_id)}}));
        EXPECT_EQ(log->emittedEvents[2].attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", std::int64_t{3}}}));
    }
}

TEST(TracePhaseSink, PhaseUnderAnIdentityRecordsItsFields) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    constexpr std::string_view traceparent =
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    const auto context = nixl::trace::parseTraceparent(traceparent);
    ASSERT_TRUE(context.has_value());
    constexpr std::uint64_t request_id = 0x7edcba9876543210ULL;
    const nixlBackendTraceAttr attrs[] = {{nixl::trace::AttributeKey::NixlRemoteAgent, "peer"}};

    sink.recordPhase(nixl_trace_phase_t::NOTIF_RECEIVED,
                     7,
                     {*context, nixl::trace::RequestId{request_id}, 2},
                     attrs,
                     "rail0");
    sink.recordPhase(nixl_trace_phase_t::SUBMIT, 8, {}, {}, {});

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), 2u);
        EXPECT_EQ(log->emittedEvents[0].context, *context);
        EXPECT_EQ(log->emittedEvents[0].requestId, request_id);
        EXPECT_EQ(log->emittedEvents[0].times, std::vector<nixlTime::us_t>{7});
        EXPECT_EQ(log->emittedEvents[0].attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", std::int64_t{7}},
                               {"nixl.xfer_request_id", static_cast<std::int64_t>(request_id)},
                               {"nixl.xfer_execution", std::int64_t{2}},
                               {"nixl.phase.label", std::string("rail0")},
                               {"nixl.remote_agent", std::string("peer")}}));
        EXPECT_EQ(log->emittedEvents[1].context, nixl::trace::TraceContext{});
        EXPECT_EQ(log->emittedEvents[1].requestId, 0u);
        EXPECT_EQ(log->emittedEvents[1].attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", std::int64_t{8}}}));
    }
}

TEST(TracePhaseSink, PhaseWithoutATimeIsStampedWhenRecorded) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    const nixl::trace::TraceIdentity identity{testContext(0x38), nixl::trace::RequestId{0x38}, 1};
    const nixlBackendTraceAttr attrs[] = {{nixl::trace::AttributeKey::RemoteAgent, "0"}};

    const nixlTime::us_t before = nixlTime::getUs();
    sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, identity, attrs, "rail0");
    const nixlTime::us_t after = nixlTime::getUs();

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), 1u);
        const EmittedRecord &event = log->emittedEvents[0];
        ASSERT_EQ(event.times.size(), 1u);
        const nixlTime::us_t at = event.times[0];
        EXPECT_GE(at, before);
        EXPECT_LE(at, after);
        EXPECT_EQ(event.context, identity.context);
        EXPECT_EQ(event.requestId, *identity.requestId);
        EXPECT_EQ(event.attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", static_cast<std::int64_t>(at)},
                               {"nixl.xfer_request_id", std::int64_t{0x38}},
                               {"nixl.xfer_execution", std::int64_t{1}},
                               {"nixl.phase.label", std::string("rail0")},
                               {"remote_agent", std::string("0")}}));
    }
}

TEST(Tracing, TracerNeedsTimestampsUnlessEveryBackendOwnsThem) {
    CallLog log;
    const auto needs = [&](std::initializer_list<bool> owns) {
        std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
        for (const bool owned : owns) {
            if (owned) {
                backends.push_back(std::make_unique<SelfTimedBackend>("self", &log, 0));
            } else {
                backends.push_back(std::make_unique<MockBackend>("mock", &log));
            }
        }
        return nixl::trace::CompositeTracer{std::move(backends), 0.0}.needsTimestamps();
    };

    EXPECT_TRUE(needs({false}));
    EXPECT_TRUE(needs({true, false}));
    EXPECT_TRUE(needs({false, true}));
    EXPECT_FALSE(needs({true}));
    EXPECT_FALSE(needs({true, true}));
    EXPECT_FALSE(needs({}));
}

TEST(TracePhaseSink, BackendsThatOwnTimestampsGetNoClockReading) {
    CallLog a, b;
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<SelfTimedBackend>("a", &a, 0));
    backends.push_back(std::make_unique<SelfTimedBackend>("b", &b, 0));
    nixl::trace::CompositeTracer tracer{std::move(backends), 0.0};
    nixl::trace::TracerPhaseSink sink{tracer, "UCX"};
    const nixl::trace::TraceIdentity identity{testContext(0x3c), nixl::trace::RequestId{0x3c}, 1};
    const nixlBackendPostTrace post{&sink, identity};

    EXPECT_FALSE(sink.needsTimestamps());
    EXPECT_EQ(post.now(), 0u);
    post.record(nixl_trace_phase_t::SUBMIT);

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), 1u);
        EXPECT_EQ(log->emittedEvents[0].times, std::vector<nixlTime::us_t>{0});
        EXPECT_EQ(log->emittedEvents[0].attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.xfer_request_id", std::int64_t{0x3c}},
                               {"nixl.xfer_execution", std::int64_t{1}}}));
    }
}

TEST(Tracing, RecordReadsNoClockForBackendsThatOwnTimestamps) {
    CallLog a, b;
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<SelfTimedBackend>("a", &a, 0));
    backends.push_back(std::make_unique<SelfTimedBackend>("b", &b, 0));
    nixl::trace::CompositeTracer tracer{std::move(backends), 0.0};
    const nixl::trace::Traceable traced{&tracer, {testContext(0x3d), nixl::trace::RequestId{0x3d}}};

    traced.trace(nixl::trace::Record::XferComplete);

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), 1u);
        EXPECT_EQ(log->emittedEvents[0].name, "nixl::xfer.complete");
        EXPECT_EQ(log->emittedEvents[0].times, std::vector<nixlTime::us_t>{0});
    }
}

TEST(TracePhaseSink, PostTraceRecordsOnlyWithASinkAndASampledPost) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    const nixl::trace::TraceIdentity sampled{testContext(0x39), nixl::trace::RequestId{0x39}, 1};
    const nixl::trace::TraceIdentity unsampled{
        testContext(0x3a, 0x02), nixl::trace::RequestId{0x3a}, 1};
    std::vector<nixlBackendPostTrace> inactive{nixlBackendPostTrace{},
                                               nixlBackendPostTrace{&sink},
                                               {nullptr, sampled},
                                               {&sink, unsampled}};
    {
        const nixl::trace::Scope scope(tracer.get(), sampled);
        inactive.emplace_back(nullptr, sampled);
        inactive.push_back(nixlBackendPostTrace{nullptr});
        EXPECT_TRUE(nixlBackendPostTrace{&sink}.active());
    }
    {
        const nixl::trace::Scope scope(tracer.get(), unsampled);
        inactive.push_back(nixlBackendPostTrace{&sink});
    }
    for (const auto &trace : inactive) {
        EXPECT_FALSE(trace.active());
        trace.record(nixl_trace_phase_t::SUBMIT);
        trace.recordAt(nixl_trace_phase_t::WIRE_COMPLETED, 1);
    }

    EXPECT_TRUE(a.emittedEvents.empty());
    EXPECT_TRUE(b.emittedEvents.empty());
    EXPECT_TRUE((nixlBackendPostTrace{&sink, sampled}.active()));
}

TEST(TracePhaseSink, PostTraceAsksItsSinkForTimestampsOnce) {
    TimestampQuestionSink sink;
    const nixlBackendPostTrace post{&sink, {testContext(0x3d), nixl::trace::RequestId{0x3d}, 1}};
    const nixlTime::us_t before = nixlTime::getUs();

    EXPECT_GE(post.now(), before);
    post.record(nixl_trace_phase_t::SUBMIT);
    post.record(nixl_trace_phase_t::WIRE_SUBMITTED);

    EXPECT_EQ(sink.questions, 1);
    ASSERT_EQ(sink.timestamps.size(), 2u);
    EXPECT_GE(sink.timestamps[0], before);
    EXPECT_GE(sink.timestamps[1], sink.timestamps[0]);
}

TEST(TracePhaseSink, PostTraceRecordsItsPostsIdentity) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    const nixl::trace::TraceIdentity identity{testContext(0x3b), nixl::trace::RequestId{0x3b}, 2};
    const nixlBackendPostTrace posted = [&] {
        const nixl::trace::Scope scope(tracer.get(), identity);
        return nixlBackendPostTrace{&sink};
    }();
    const nixlBackendPostTrace received{&sink, identity};
    const nixlBackendTraceAttr rail[] = {{nixl::trace::AttributeKey::RemoteAgent, "0"}};
    const nixlBackendTraceAttr peer[] = {{nixl::trace::AttributeKey::NixlRemoteAgent, "peer"}};

    const nixlTime::us_t before = nixlTime::getUs();
    posted.record(nixl_trace_phase_t::SUBMIT, rail);
    const nixlTime::us_t after = nixlTime::getUs();
    received.recordAt(nixl_trace_phase_t::NOTIF_RECEIVED, 42, peer);

    EXPECT_TRUE(sink.needsTimestamps());
    EXPECT_GE(posted.now(), after);

    EXPECT_EQ(posted.identity().context, identity.context);
    EXPECT_EQ(posted.identity().requestId, identity.requestId);
    EXPECT_EQ(posted.identity().execution, identity.execution);
    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedEvents.size(), 2u);
        const EmittedRecord &submit = log->emittedEvents[0];
        ASSERT_EQ(submit.times.size(), 1u);
        const nixlTime::us_t at = submit.times[0];
        EXPECT_EQ(submit.name, "nixl::submit");
        EXPECT_GE(at, before);
        EXPECT_LE(at, after);
        EXPECT_EQ(submit.context, identity.context);
        EXPECT_EQ(submit.requestId, *identity.requestId);
        EXPECT_EQ(submit.attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", static_cast<std::int64_t>(at)},
                               {"nixl.xfer_request_id", std::int64_t{0x3b}},
                               {"nixl.xfer_execution", std::int64_t{2}},
                               {"remote_agent", std::string("0")}}));
        const EmittedRecord &receipt = log->emittedEvents[1];
        EXPECT_EQ(receipt.name, "nixl::notif.received");
        EXPECT_EQ(receipt.times, std::vector<nixlTime::us_t>{42});
        EXPECT_EQ(receipt.context, identity.context);
        EXPECT_EQ(receipt.attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.phase.timestamp_us", std::int64_t{42}},
                               {"nixl.xfer_request_id", std::int64_t{0x3b}},
                               {"nixl.xfer_execution", std::int64_t{2}},
                               {"nixl.remote_agent", std::string("peer")}}));
    }
}

TEST(TracePhaseSink, DrainAsksTheSinkOnlyWhilePhasesArePending) {
    std::atomic<std::uint64_t> pending{0};
    DrainCountingSink sink{&pending};

    nixl::trace::recordDeferredPhases(&sink);
    EXPECT_EQ(sink.drains, 0);

    pending = 1;
    nixl::trace::recordDeferredPhases(&sink);
    EXPECT_EQ(sink.drains, 1);
}

TEST(TracePhaseSink, DeferredPhaseIsPendingUntilRecorded) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b, 1.0);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    sink.recordDeferredPhases();
    const std::uint64_t before = nixl::trace::pendingDeferredPhases.load();
    nixlBackendPostPhases phases;
    {
        const nixl::trace::Scope scope(tracer.get(),
                                       {testContext(0x3e), nixl::trace::RequestId{0x3e}, 1});
        phases.begin(&sink, NIXL_WRITE, nixl_trace_completion_t::CALLBACK);
    }

    phases.callbackCompleted(NIXL_SUCCESS);
    EXPECT_EQ(nixl::trace::pendingDeferredPhases.load(), before + 1);
    EXPECT_TRUE(sink.mayHaveDeferredPhases());

    phases.drain();
    EXPECT_EQ(nixl::trace::pendingDeferredPhases.load(), before);
    for (const CallLog *log : {&a, &b}) {
        EXPECT_TRUE(std::ranges::any_of(log->emittedEvents, [](const EmittedRecord &event) {
            return event.name == "nixl::wire.completed";
        }));
    }
}

TEST(TracePhaseSink, CallbackCompletionRecordsAFailuresStatus) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    const nixl::trace::TraceIdentity failed{testContext(0x3c), nixl::trace::RequestId{0x3c}, 1};
    const nixl::trace::TraceIdentity completed{testContext(0x3d), nixl::trace::RequestId{0x3d}, 1};
    for (const auto &[identity, status] :
         {std::pair{failed, NIXL_ERR_REMOTE_DISCONNECT}, std::pair{completed, NIXL_SUCCESS}}) {
        nixlBackendPostPhases phases;
        {
            const nixl::trace::Scope scope(tracer.get(), identity);
            phases.begin(&sink, NIXL_WRITE, nixl_trace_completion_t::CALLBACK);
        }
        phases.callbackCompleted(status);
        phases.drain();
    }

    for (const CallLog *log : {&a, &b}) {
        std::vector<const EmittedRecord *> completions;
        for (const EmittedRecord &event : log->emittedEvents) {
            if (event.name == "nixl::wire.completed") {
                completions.push_back(&event);
            }
        }
        ASSERT_EQ(completions.size(), 2u);
        EXPECT_EQ(completions[0]->context, failed.context);
        const attr_t failure{"nixl.status", std::int64_t{NIXL_ERR_REMOTE_DISCONNECT}};
        EXPECT_NE(std::ranges::find(completions[0]->attrs, failure), completions[0]->attrs.end());
        EXPECT_EQ(completions[1]->context, completed.context);
        EXPECT_TRUE(std::ranges::none_of(
            completions[1]->attrs, [](const attr_t &attr) { return attr.first == "nixl.status"; }));
        ASSERT_EQ(log->emittedSpans.size(), 2u);
        ASSERT_EQ(completions[0]->times.size(), 1u);
        EXPECT_EQ(log->emittedSpans[0].times.back(), completions[0]->times.front());
    }
}

TEST(TracePhaseSink, CompletionRecordsAFailuresStatus) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    const nixl::trace::TraceIdentity failed{testContext(0x3f), nixl::trace::RequestId{0x3f}, 1};
    const nixl::trace::TraceIdentity completed{testContext(0x40), nixl::trace::RequestId{0x40}, 1};
    for (const auto &[identity, status] :
         {std::pair{failed, NIXL_ERR_BACKEND}, std::pair{completed, NIXL_SUCCESS}}) {
        nixlBackendPostPhases phases;
        {
            const nixl::trace::Scope scope(tracer.get(), identity);
            phases.begin(&sink, NIXL_WRITE, nixl_trace_completion_t::LAST_PART);
        }
        phases.completed(status);
    }

    for (const CallLog *log : {&a, &b}) {
        std::vector<const EmittedRecord *> completions;
        for (const EmittedRecord &event : log->emittedEvents) {
            if (event.name == "nixl::wire.completed") {
                completions.push_back(&event);
            }
        }
        ASSERT_EQ(completions.size(), 2u);
        EXPECT_EQ(completions[0]->context, failed.context);
        const attr_t failure{"nixl.status", std::int64_t{NIXL_ERR_BACKEND}};
        EXPECT_NE(std::ranges::find(completions[0]->attrs, failure), completions[0]->attrs.end());
        EXPECT_EQ(completions[1]->context, completed.context);
        EXPECT_TRUE(std::ranges::none_of(
            completions[1]->attrs, [](const attr_t &attr) { return attr.first == "nixl.status"; }));
        ASSERT_EQ(log->emittedSpans.size(), 2u);
        ASSERT_EQ(completions[0]->times.size(), 1u);
        EXPECT_EQ(log->emittedSpans[0].times.back(), completions[0]->times.front());
    }
}

// A backend may decline a span; Tracer drops the null, leaving the Span
// inactive even though the tracer has a backend. Nothing must be recorded and
// no null span may be dereferenced.
TEST(Tracing, InactiveSpanRecordsNothing) {
    CallLog a;
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    backends.push_back(std::make_unique<MockBackend>("a", &a, /*null_spans=*/true));
    nixl::trace::CompositeTracer tracer{std::move(backends), 0.0};
    auto span = tracer.beginSpan("op");
    auto context_span = tracer.beginContextSpan(
        "op", nixl::trace::Kind::Generic, {testContext(0x25), nixl::trace::RequestId{0x2525}});

    span.addAttribute(nixl::trace::AttributeKey::RemoteAgent, std::string_view{"0"});
    span.addAttribute(nixl::trace::AttributeKey::Bytes, std::int64_t{1});
    context_span.addAttribute(nixl::trace::AttributeKey::RemoteAgent, std::string_view{"0"});
    context_span.addAttribute(nixl::trace::AttributeKey::Bytes, std::int64_t{1});

    EXPECT_FALSE(tracer.empty());
    EXPECT_FALSE(span.active());
    EXPECT_FALSE(context_span.active());
    EXPECT_EQ(a.spansBegun, 1);
    EXPECT_EQ(a.contextSpans.size(), 1u);
    EXPECT_EQ(a.spansEnded, 0);
    EXPECT_TRUE(a.strAttrs.empty());
    EXPECT_TRUE(a.intAttrs.empty());
}

// recordPhase() and recordSpan() are noexcept: a backend that throws must not
// propagate, and the drop must be reported once per sink however many phases
// and spans are dropped.
TEST(TracePhaseSink, ThrowingBackendIsReportedOnce) {
    const auto tracer = makeThrowingTracer();
    gtest::LogMessageCounter drops("Dropping a trace phase or span");
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};

    EXPECT_NO_THROW(sink.recordPhase(nixl_trace_phase_t::SUBMIT, 1, {}, {}, {}));
    EXPECT_NO_THROW(sink.recordPhase(nixl_trace_phase_t::WIRE_SUBMITTED, 2, {}, {}, {}));
    EXPECT_NO_THROW(
        sink.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT, NIXL_WRITE, 3, 4, {}, {}, {}));

    EXPECT_EQ(drops.count(), 1u);
}

TEST(TracePhaseSink, RecordedSpanReachesEveryBackend) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    constexpr std::string_view traceparent =
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    const auto context = nixl::trace::parseTraceparent(traceparent);
    ASSERT_TRUE(context.has_value());
    constexpr std::uint64_t request_id = 0x7edcba9876543210ULL;
    const nixlBackendTraceAttr attrs[] = {{nixl::trace::AttributeKey::RemoteAgent, "0"}};

    sink.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT,
                    NIXL_WRITE,
                    1000,
                    1250,
                    nixl::trace::TraceIdentity{*context, nixl::trace::RequestId{request_id}, 3},
                    attrs,
                    {});

    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->spansBegun, 0);
        EXPECT_TRUE(log->pushedContexts.empty());
        EXPECT_TRUE(log->emittedEvents.empty());
        ASSERT_EQ(log->emittedSpans.size(), 1u);
        const EmittedRecord &span = log->emittedSpans[0];
        EXPECT_EQ(span.name, "nixl::wire.inflight.write");
        EXPECT_EQ(span.times, (std::vector<nixlTime::us_t>{1000, 1250}));
        EXPECT_EQ(span.context, *context);
        EXPECT_EQ(span.requestId, request_id);
        EXPECT_EQ(span.attrs,
                  (attr_list_t{{"nixl.backend", std::string("UCX")},
                               {"nixl.xfer_request_id", static_cast<std::int64_t>(request_id)},
                               {"nixl.xfer_execution", std::int64_t{3}},
                               {"remote_agent", std::string("0")}}));
    }
}

TEST(TracePhaseSink, SpanNamedByARecordHasItsKindAndALabeledOneItsOperations) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};

    sink.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT, NIXL_WRITE, 1, 2, {}, {}, {});
    sink.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT, NIXL_READ, 3, 4, {}, {}, {});
    sink.recordSpan(nixl_trace_span_t::OTHER, NIXL_WRITE, 5, 6, {}, {}, {});
    sink.recordSpan(nixl_trace_span_t::OTHER, NIXL_READ, 7, 8, {}, {}, "rail0.read");

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedSpans.size(), 4u);
        EXPECT_EQ(log->emittedSpans[0].name, "nixl::wire.inflight.write");
        EXPECT_EQ(log->emittedSpans[0].kind,
                  nixl::trace::kindOf(nixl::trace::Record::WireInflightWrite));
        EXPECT_EQ(log->emittedSpans[0].kind, nixl::trace::Kind::CommSend);
        EXPECT_EQ(log->emittedSpans[1].name, "nixl::wire.inflight.read");
        EXPECT_EQ(log->emittedSpans[1].kind,
                  nixl::trace::kindOf(nixl::trace::Record::WireInflightRead));
        EXPECT_EQ(log->emittedSpans[1].kind, nixl::trace::Kind::CommRecv);
        EXPECT_EQ(log->emittedSpans[2].name, "nixl::span");
        EXPECT_EQ(log->emittedSpans[2].kind, nixl::trace::kindOf(nixl::trace::Record::Span));
        EXPECT_EQ(log->emittedSpans[3].name, "rail0.read");
        EXPECT_EQ(log->emittedSpans[3].kind, nixl::trace::Kind::CommRecv);
    }
}

TEST(TracePhaseSink, SpanVocabularyIsDistinct) {
    constexpr nixl_trace_phase_t kPhases[] = {nixl_trace_phase_t::SUBMIT,
                                              nixl_trace_phase_t::WIRE_SUBMITTED,
                                              nixl_trace_phase_t::WIRE_COMPLETED,
                                              nixl_trace_phase_t::NOTIF_SENT,
                                              nixl_trace_phase_t::NOTIF_RECEIVED,
                                              nixl_trace_phase_t::REMOTE_OBSERVED,
                                              nixl_trace_phase_t::OTHER};
    constexpr nixl_trace_span_t kSpans[] = {nixl_trace_span_t::WIRE_INFLIGHT,
                                            nixl_trace_span_t::OTHER};
    std::set<std::string_view> names;
    for (const auto phase : kPhases) {
        names.insert(toStringView(phase));
    }
    for (const auto span : kSpans) {
        for (const auto operation : {NIXL_WRITE, NIXL_READ}) {
            names.insert(toStringView(span, operation));
        }
    }
    EXPECT_EQ(names.size(), std::size(kPhases) + 3);

    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "LIBFABRIC"};

    sink.recordSpan(nixl_trace_span_t::OTHER, NIXL_WRITE, 1, 2, {}, {}, "rail0.write");
    sink.recordSpan(nixl_trace_span_t::OTHER, NIXL_WRITE, 3, 4, {}, {}, {});
    sink.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT, NIXL_WRITE, 5, 6, {}, {}, "rail0");

    ASSERT_EQ(a.emittedSpans.size(), 3u);
    EXPECT_EQ(a.emittedSpans[0].name, "rail0.write");
    EXPECT_EQ(a.emittedSpans[0].attrs,
              (attr_list_t{{"nixl.backend", std::string("LIBFABRIC")},
                           {"nixl.span.label", std::string("rail0.write")}}));
    EXPECT_EQ(a.emittedSpans[1].name, "nixl::span");
    EXPECT_EQ(a.emittedSpans[1].attrs, (attr_list_t{{"nixl.backend", std::string("LIBFABRIC")}}));
    EXPECT_EQ(a.emittedSpans[2].name, "nixl::wire.inflight.write");
    EXPECT_EQ(a.emittedSpans[2].attrs,
              (attr_list_t{{"nixl.backend", std::string("LIBFABRIC")},
                           {"nixl.span.label", std::string("rail0")}}));
}

TEST(TracePhaseSink, SpanWithoutContextOrIdRecordsNeither) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    const nixl::trace::TraceContext zeroed;

    sink.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT, NIXL_WRITE, 1, 2, {}, {}, {});
    sink.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT,
                    NIXL_READ,
                    3,
                    4,
                    nixl::trace::TraceIdentity{zeroed},
                    {},
                    {});

    for (const CallLog *log : {&a, &b}) {
        ASSERT_EQ(log->emittedSpans.size(), 2u);
        for (const EmittedRecord &span : log->emittedSpans) {
            EXPECT_FALSE(span.context.valid());
            EXPECT_EQ(span.requestId, 0u);
            EXPECT_EQ(span.attrs, (attr_list_t{{"nixl.backend", std::string("UCX")}}));
        }
    }
}

TEST(Tracing, AllocationCounterCountsOnlyItsOwnThread) {
    const gtest::AllocationCounter allocations;
    void *const value = ::operator new(sizeof(std::uint64_t));
    EXPECT_EQ(allocations.count(), 1u);

    std::atomic<bool> start = false;
    std::size_t worker_count = 0;
    std::size_t worker_total = 0;
    std::thread worker([&start, &worker_count, &worker_total] {
        while (!start) {
            std::this_thread::yield();
        }
        const gtest::AllocationCounter worker_allocations;
        {
            const gtest::AllocationCounter nested_allocations;
            void *const other = ::operator new(sizeof(std::uint64_t));
            worker_count = nested_allocations.count();
            ::operator delete(other);
        }
        void *const after = ::operator new(sizeof(std::uint64_t));
        worker_total = worker_allocations.count();
        ::operator delete(after);
    });
    const std::size_t with_thread = allocations.count();
    start = true;
    worker.join();
    EXPECT_EQ(allocations.count(), with_thread);
    EXPECT_EQ(worker_count, 1u);
    EXPECT_EQ(worker_total, 2u);
    ::operator delete(value);
}

TEST(TracePhaseSink, RecordingAllocatesNothing) {
    std::size_t attrs_seen = 0;
    const auto tracer = makeDiscardingTracer(attrs_seen);
    nixl::trace::TracerPhaseSink sink{*tracer, "UCX"};
    const nixl::trace::TraceIdentity identity{
        testContext(0x11), nixl::trace::RequestId{0xfedcba9876543210ULL}, 3};
    const nixlBackendTraceAttr attrs[] = {{nixl::trace::AttributeKey::RemoteAgent, "0"}};

    const gtest::AllocationCounter allocations;
    sink.recordPhase(nixl_trace_phase_t::SUBMIT, 1, identity, attrs, {});
    sink.recordPhase(nixl_trace_phase_t::WIRE_COMPLETED, identity);
    sink.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT, NIXL_WRITE, 1, 2, identity, attrs, {});
    EXPECT_EQ(allocations.count(), 0u);
    EXPECT_EQ(attrs_seen, 5u + 4u + 4u);
}

TEST(Tracing, SpanWithoutBackendSpansAllocatesNothing) {
    std::size_t attrs_seen = 0;
    const auto tracer = makeDiscardingTracer(attrs_seen);
    const gtest::AllocationCounter allocations;
    {
        const nixl::trace::Span span = tracer->beginSpan("nixl::createXferReq");
        EXPECT_FALSE(span.active());
        const nixl::trace::Span context_span =
            tracer->beginContextSpan("nixl::postXferReq.write",
                                     nixl::trace::Kind::CommSend,
                                     {testContext(0x22), nixl::trace::RequestId{7}});
        EXPECT_FALSE(context_span.active());
    }
    EXPECT_EQ(allocations.count(), 0u);
}

TEST(Tracing, MovedSpanKeepsItsBackendSpans) {
    CallLog a, b;
    const auto tracer = makeMockTracer(a, b);
    {
        nixl::trace::Span first = tracer->beginSpan("nixl::createXferReq");
        nixl::trace::Span second = std::move(first);
        EXPECT_FALSE(first.active());
        ASSERT_TRUE(second.active());
        second.addAttribute(nixl::trace::AttributeKey::DescCount, std::int64_t{8});
        EXPECT_EQ(a.spansEnded, 0);
    }
    for (const CallLog *log : {&a, &b}) {
        EXPECT_EQ(log->spansBegun, 1);
        EXPECT_EQ(log->spansEnded, 1);
        EXPECT_EQ(log->intAttrs,
                  (std::vector<std::pair<std::string, std::int64_t>>{{"desc_count", 8}}));
    }
}
