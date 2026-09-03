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

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "tracing/trace_plugin.h"

namespace {

class publicPluginBackend : public nixl::trace::TraceBackend {
public:
    explicit publicPluginBackend(const nixlTraceBackendInitParams &init_params)
        : agentName_(init_params.agentName) {}

    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(std::string_view, nixl::trace::Kind) override {
        return nullptr;
    }

    void
    mark(std::string_view, nixl::trace::Kind) override {}

    void
    pushCorrelationId(std::uint64_t) override {}

    void
    popCorrelationId() override {}

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return agentName_;
    }

private:
    std::string agentName_;
};

class throwingPublicPluginBackend final : public publicPluginBackend {
public:
    explicit throwingPublicPluginBackend(const nixlTraceBackendInitParams &init_params)
        : publicPluginBackend(init_params) {
        throw std::runtime_error("expected test failure");
    }
};

} // namespace

TEST(TracingPublicPlugin, CreatesBackendFromInstalledContract) {
    using creator_t = nixlTracePluginCreator<publicPluginBackend>;
    auto *plugin = creator_t::create(nixl_trace_plugin_api_version::V1, "external", "1.0.0");

    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(plugin->api_version, nixl_trace_plugin_api_version::V1);
    EXPECT_EQ(plugin->getName(), "external");
    EXPECT_EQ(plugin->getVersion(), "1.0.0");

    auto backend = plugin->create_backend({.agentName = "agent_0"});
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->name(), "agent_0");
}

TEST(TracingPublicPlugin, ConvertsConstructorFailureToNull) {
    using creator_t = nixlTracePluginCreator<throwingPublicPluginBackend>;
    auto *plugin = creator_t::create(nixl_trace_plugin_api_version::V1, "throwing", "1.0.0");

    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(plugin->create_backend({.agentName = "agent_0"}), nullptr);
}
