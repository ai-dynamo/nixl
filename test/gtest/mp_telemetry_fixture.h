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
#ifndef NIXL_TEST_GTEST_MP_TELEMETRY_FIXTURE_H
#define NIXL_TEST_GTEST_MP_TELEMETRY_FIXTURE_H

#include "common.h"
#include "plugin_manager.h"
#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

#include <gtest/gtest.h>

[[nodiscard]] inline std::size_t
idx(nixl_telemetry_event_type_t t) {
    return static_cast<std::size_t>(t);
}

[[nodiscard]] inline nixlTelemetryExporterInitParams
initParams(const std::string &agent) {
    return nixlTelemetryExporterInitParams{agent, 4096};
}

// A telemetry directory of this test's own, mode 0700: the exporter asks
// operators for that, and without it a permissive umask makes it warn about the
// directory, which the gtest main counts as a failure.
class mpTempDirTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = std::filesystem::path(::testing::TempDir()) /
            ("nixl_mp_" + std::to_string(::getpid()) + "_" + info->test_suite_name() + "_" +
             info->name());
        std::filesystem::create_directories(dir_);
        std::filesystem::permissions(
            dir_, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
    }

    void
    TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::filesystem::path dir_;
};

// The multi-process exporter pointed at that directory and a port of its own.
// Driven both directly (telemetry_mp_exporter_test.cpp) and through the core
// (telemetry_mp_core_test.cpp): both files name this single fixture class, so
// gtest groups them into one suite and runs SetUpTestSuite once per iteration.
class MpExporterTest : public mpTempDirTest {
protected:
    // A build tree has no <libnixl.so dir>/plugins, so LoadsThroughPluginManager
    // finds the plugin only if the build path is registered. Registered once per
    // process (--gtest_repeat re-enters this hook), and only when it exists:
    // re-registering, or registering a missing directory, logs a warning/error
    // that the gtest main counts as a failure. When it is absent (a binary run
    // from an install tree) NIXL_PLUGIN_DIR is what supplies the plugin.
    static void
    SetUpTestSuite() {
        [[maybe_unused]] static const bool registered = [] {
            const std::string build_plugin_dir =
                std::string(BUILD_DIR) + "/src/plugins/telemetry/prometheus_mp";
            if (std::filesystem::is_directory(build_plugin_dir)) {
                nixlPluginManager::getInstance().addPluginDirectory(build_plugin_dir);
            }
            return true;
        }();
    }

    void
    SetUp() override {
        mpTempDirTest::SetUp();
        port_ = gtest::PortAllocator::next_tcp_port();
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_LOCAL", "y");
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_PORT", std::to_string(port_));
        env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", dir_.string());
    }

    [[nodiscard]] std::filesystem::path
    singleStoreFile() const {
        std::error_code ec;
        for (const auto &entry : std::filesystem::directory_iterator(dir_, ec)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("nixl.", 0) == 0 && name.size() > 5 &&
                name.substr(name.size() - 5) == ".mmap") {
                return entry.path();
            }
        }
        return {};
    }

    gtest::ScopedEnv env_;
    uint16_t port_ = 0;
};

#endif // NIXL_TEST_GTEST_MP_TELEMETRY_FIXTURE_H
