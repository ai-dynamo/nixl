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
#include "tracing/trace_plugin.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

struct pushedScope {
    nixl::trace::TraceContext context;
    std::uint64_t requestId = 0;
};

[[nodiscard]] std::vector<pushedScope> &
contextStack() noexcept {
    static thread_local std::vector<pushedScope> stack;
    return stack;
}

void
addAttributes(nixl::trace::SpanBackend &record, std::span<const nixl::trace::Attribute> attrs) {
    for (const nixl::trace::Attribute &attr : attrs) {
        std::visit([&](const auto value) { record.addAttribute(attr.key, value); }, attr.value);
    }
}

class recordingSpan final : public nixl::trace::SpanBackend {
public:
    recordingSpan(int fd, std::string line) : fd_(fd), line_(std::move(line)) {}

    ~recordingSpan() override {
        line_.push_back('\n');
        static_cast<void>(::write(fd_, line_.data(), line_.size()));
    }

    void
    addAttribute(std::string_view key, std::string_view value) override {
        line_.append(" ").append(key).append("=").append(value);
    }

    void
    addAttribute(std::string_view key, std::int64_t value) override {
        addAttribute(key, std::to_string(value));
    }

    void
    addAttribute(std::string_view, double) override {}

    void
    addCtrlDep(nixl::trace::SpanId) override {}

    void
    addDataDep(nixl::trace::SpanId) override {}

    [[nodiscard]] nixl::trace::SpanId
    id() const noexcept override {
        return {};
    }

private:
    const int fd_;
    std::string line_;
};

class recordingBackend final : public nixl::trace::TraceBackend {
public:
    explicit recordingBackend(const nixlTraceBackendInitParams &init_params)
        : agent_(init_params.agentName),
          fd_(openTraceFile()) {}

    ~recordingBackend() override {
        ::close(fd_);
    }

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(std::string_view name, nixl::trace::Kind) override {
        const auto &stack = contextStack();
        const pushedScope top = stack.empty() ? pushedScope{} : stack.back();
        return std::make_unique<recordingSpan>(fd_, lineFor(name, top.context, top.requestId));
    }

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginContextSpan(std::string_view name,
                     nixl::trace::Kind,
                     const nixl::trace::TraceContext &context,
                     std::uint64_t request_id) override {
        return std::make_unique<recordingSpan>(fd_, lineFor(name, context, request_id, "span"));
    }

    void
    mark(std::string_view name, nixl::trace::Kind kind) override {
        static_cast<void>(beginSpan(name, kind));
    }

    void
    emitSpan(std::string_view name,
             nixl::trace::Kind,
             const nixl::trace::TraceContext &context,
             std::uint64_t request_id,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const nixl::trace::Attribute> attrs) override {
        recordingSpan span(fd_,
                           lineFor(name, context, request_id) + " start_us=" +
                               std::to_string(start) + " end_us=" + std::to_string(end));
        addAttributes(span, attrs);
    }

    void
    emitEvent(std::string_view name,
              nixl::trace::Kind,
              const nixl::trace::TraceContext &context,
              std::uint64_t request_id,
              nixlTime::us_t at,
              std::span<const nixl::trace::Attribute> attrs) override {
        recordingSpan event(fd_,
                            lineFor(name, context, request_id) + " at_us=" + std::to_string(at));
        addAttributes(event, attrs);
    }

    void
    pushContext(const nixl::trace::TraceContext &context, std::uint64_t request_id) override {
        contextStack().push_back({context, request_id});
    }

    void
    popContext() override {
        if (!contextStack().empty()) {
            contextStack().pop_back();
        }
    }

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return "recorder";
    }

private:
    [[nodiscard]] std::string
    lineFor(std::string_view name,
            const nixl::trace::TraceContext &context,
            std::uint64_t request_id,
            std::string_view context_key = "context") const {
        std::string line = agent_ + " " + std::string(name);
        if (request_id != 0) {
            line += " correlation=" + std::to_string(request_id);
        }
        if (context.valid()) {
            line += " " + std::string(context_key) + "=" + nixl::trace::formatTraceparent(context);
        }
        return line;
    }

    [[nodiscard]] static int
    openTraceFile() {
        const char *path = std::getenv("NIXL_TEST_TRACE_FILE");
        const int fd =
            (path == nullptr) ? -1 : ::open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (fd < 0) {
            throw std::runtime_error("cannot open NIXL_TEST_TRACE_FILE");
        }
        return fd;
    }

    const std::string agent_;
    const int fd_;
};

} // namespace

extern "C" NIXL_TRACE_PLUGIN_EXPORT nixlTracePlugin *
nixl_trace_plugin_init() {
    return nixlTracePluginCreator<recordingBackend>::create(
        nixl_trace_plugin_api_version::V2, "recorder", "0.1.0");
}

extern "C" NIXL_TRACE_PLUGIN_EXPORT void
nixl_trace_plugin_fini() {}
