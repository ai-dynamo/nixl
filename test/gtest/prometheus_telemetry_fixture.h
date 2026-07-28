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
#include <string>

#include <gtest/gtest.h>

// Shared fixture for the native Prometheus telemetry exporter tests, which live
// in both telemetry_prometheus_test.cpp and telemetry_histogram_test.cpp. Because
// both files drive this single fixture class, gtest groups them into one test
// suite and runs SetUpTestSuite exactly once, so the plugin directory is
// registered a single time for the whole process (registering it per-test, or
// twice from two suites, trips the plugin manager's "already registered" warning
// which the gtest main treats as a failure).
class prometheusTelemetryTest : public ::testing::Test {
protected:
    static void
    SetUpTestSuite() {
        nixlPluginManager::getInstance().addPluginDirectory(std::string(BUILD_DIR) +
                                                            "/src/plugins/telemetry/prometheus");
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
