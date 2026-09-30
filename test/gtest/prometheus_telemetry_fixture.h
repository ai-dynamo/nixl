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
#ifndef NIXL_TEST_GTEST_PROMETHEUS_TELEMETRY_FIXTURE_H
#define NIXL_TEST_GTEST_PROMETHEUS_TELEMETRY_FIXTURE_H

#include "common.h"
#include "plugin_manager.h"

#include <cstdint>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

// Shared fixture for the native Prometheus telemetry exporter tests, which live
// in both telemetry_prometheus_test.cpp and telemetry_histogram_test.cpp. Because
// both files drive this single fixture class, gtest groups them into one test
// suite and runs SetUpTestSuite once per iteration. The registration is kept to
// once per process because registering the same directory twice (a second suite,
// or --gtest_repeat re-entering this hook) trips the plugin manager's "already
// registered" warning, which the gtest main treats as a failure.
class prometheusTelemetryTest : public ::testing::Test {
protected:
    static void
    SetUpTestSuite() {
        [[maybe_unused]] static const bool registered = [] {
            // Guarded: the plugin manager logs an ERROR for a directory that is
            // not there, and this build does not always have the plugin.
            const std::string build_plugin_dir =
                std::string(BUILD_DIR) + "/src/plugins/telemetry/prometheus";
            if (std::filesystem::is_directory(build_plugin_dir)) {
                nixlPluginManager::getInstance().addPluginDirectory(build_plugin_dir);
            }
            return true;
        }();
    }

    void
    SetUp() override {
        port_ = gtest::PortAllocator::next_tcp_port();
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_LOCAL", "y");
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_PORT", std::to_string(port_));
    }

    void
    TearDown() override {
        env_.popVar();
        env_.popVar();
    }

    gtest::ScopedEnv env_;
    uint16_t port_ = 0;
};

#endif // NIXL_TEST_GTEST_PROMETHEUS_TELEMETRY_FIXTURE_H
