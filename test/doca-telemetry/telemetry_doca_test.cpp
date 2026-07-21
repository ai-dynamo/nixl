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

#include "scrape_util.h"

#include <doca_telemetry_exporter.h>
#include <doca_error.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include <gtest/gtest.h>

using nixl::doca_test::loopbackConnection;
using nixl::doca_test::scrapeUntilValue;

class docaTelemetryTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        port_ = loopbackConnection::findFreePort();
        ASSERT_NE(port_, 0) << "failed to allocate a free TCP port";
    }

    uint16_t port_ = 0;
};

// Raw DOCA Metrics API proof, mirroring the DOCA prometheus_example sample:
// stand up schema/source/metrics, accumulate a counter via add_counter_increment,
// flush explicitly, and confirm the DOCA Prometheus endpoint serves the
// cumulative value. Establishes that the DOCA telemetry exporter works in-process
// (the gRPC duplicate-flag clash is resolved in current DOCA).
TEST_F(docaTelemetryTest, RawDocaApiServesAccumulatingCounter) {
    // PROMETHEUS_ENDPOINT must be set before schema_init (per the DOCA sample).
    const std::string endpoint = "http://127.0.0.1:" + std::to_string(port_);
    ASSERT_EQ(::setenv("PROMETHEUS_ENDPOINT", endpoint.c_str(), 1), 0);

    doca_telemetry_exporter_schema *schema = nullptr;
    ASSERT_EQ(doca_telemetry_exporter_schema_init("nixl_doca_raw_test", &schema), DOCA_SUCCESS);
    ASSERT_EQ(doca_telemetry_exporter_schema_start(schema), DOCA_SUCCESS);

    doca_telemetry_exporter_source *source = nullptr;
    ASSERT_EQ(doca_telemetry_exporter_source_create(schema, &source), DOCA_SUCCESS);
    doca_telemetry_exporter_source_set_id(source, "raw_test_source");
    doca_telemetry_exporter_source_set_tag(source, "");
    ASSERT_EQ(doca_telemetry_exporter_source_start(source), DOCA_SUCCESS);

    ASSERT_EQ(doca_telemetry_exporter_metrics_create_context(source), DOCA_SUCCESS);
    doca_telemetry_exporter_metrics_set_flush_interval_ms(source, 1000);

    doca_telemetry_exporter_label_set_id_t label_set = 0;
    const char *label_names[] = {"type"};
    ASSERT_EQ(doca_telemetry_exporter_metrics_add_label_names(source, label_names, 1, &label_set),
              DOCA_SUCCESS);

    // Increment the same counter three times -> cumulative 3.
    const char *label_values[] = {"counter"};
    for (int i = 0; i < 3; ++i) {
        uint64_t ts = 0;
        ASSERT_EQ(doca_telemetry_exporter_get_timestamp(&ts), DOCA_SUCCESS);
        ASSERT_EQ(doca_telemetry_exporter_metrics_add_counter_increment(
                      source, ts, "raw_ops_total", 1, label_set, label_values),
                  DOCA_SUCCESS);
    }
    ASSERT_EQ(doca_telemetry_exporter_metrics_flush(source), DOCA_SUCCESS);

    const nixl::doca_test::labelSet labels{{"type", "counter"}};
    const auto metrics =
        scrapeUntilValue(port_, "raw_ops_total", 3.0, std::chrono::seconds(12), labels);
    const std::optional<double> value = metrics.latestValue("raw_ops_total", labels);
    ASSERT_TRUE(value.has_value())
        << "raw_ops_total{type=counter} not served at the DOCA Prometheus endpoint";
    EXPECT_EQ(*value, 3.0) << "add_counter_increment x3 (by 1) must yield a cumulative 3";

    doca_telemetry_exporter_metrics_destroy_context(source);
    doca_telemetry_exporter_source_destroy(source);
    doca_telemetry_exporter_schema_destroy(schema);
}

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
