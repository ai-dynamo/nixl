/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "prometheus_mp_exporter.h"
#include "mp_store.h"

#include "common.h"

#include <gtest/gtest.h>

#include <prometheus/exposer.h>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

using nixl::telemetry::mp::readStoreSnapshot;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto RX_BYTES = nixl_telemetry_event_type_t::AGENT_RX_BYTES;

[[nodiscard]] std::size_t
idx(nixl_telemetry_event_type_t t) {
    return static_cast<std::size_t>(t);
}

[[nodiscard]] nixlTelemetryExporterInitParams
initParams(const std::string &agent) {
    return nixlTelemetryExporterInitParams{agent, 4096};
}

class MpExporterTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = std::filesystem::path(::testing::TempDir()) /
            ("nixl_mp_exporter_" + std::to_string(::getpid()) + "_" + info->name());
        std::filesystem::create_directories(dir_);
        port_ = gtest::PortAllocator::next_tcp_port();
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_LOCAL", "y");
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_PORT", std::to_string(port_));
        env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", dir_.string());
    }

    void
    TearDown() override {
        env_.popVar();
        env_.popVar();
        env_.popVar();
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
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
    std::filesystem::path dir_;
};

TEST_F(MpExporterTest, OwnerBindsAndRecordsToStore) {
    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-owner"));
    EXPECT_TRUE(exporter.isExporter());

    const auto file = singleStoreFile();
    ASSERT_FALSE(file.empty());

    exporter.exportEvent({TX_BYTES, 1234});

    const auto snap = readStoreSnapshot(file);
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->agentName, "agent-owner");
    EXPECT_EQ(snap->counters[idx(TX_BYTES)], 1234u);
    EXPECT_EQ(snap->gauges[idx(TX_BYTES)], 1234u);
}

TEST_F(MpExporterTest, WriterModeWhenPortTaken) {
    // Occupy the port first so the exporter loses the bind race.
    prometheus::Exposer blocker("127.0.0.1:" + std::to_string(port_));

    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-writer"));
    EXPECT_FALSE(exporter.isExporter());

    const auto file = singleStoreFile();
    ASSERT_FALSE(file.empty());

    exporter.exportEvent({RX_BYTES, 77});

    const auto snap = readStoreSnapshot(file);
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->agentName, "agent-writer");
    EXPECT_EQ(snap->counters[idx(RX_BYTES)], 77u);
}

TEST(MpExporterStandaloneTest, MissingMultiprocDirThrows) {
    ::unsetenv("NIXL_TELEMETRY_MULTIPROC_DIR");
    EXPECT_THROW(
        { nixlTelemetryPrometheusMpExporter exporter(nixlTelemetryExporterInitParams{"a", 4096}); },
        std::runtime_error);
}

} // namespace
