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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_SCRAPE_ENDPOINT_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_SCRAPE_ENDPOINT_H

#include "mp_collector.h"
#include "owner_election.h"

#include <prometheus/exposer.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace nixl::telemetry::mp {

/**
 * @class scrapeEndpoint
 * @brief The one scrape endpoint of a telemetry directory, from the point of
 *        view of a process that may or may not be the one serving it.
 *
 * Holds everything that being the owner consists of -- the election, the HTTP
 * exposer and the collector behind it -- so that a process which is not the
 * owner holds none of it and costs nothing. claim() decides at startup;
 * reclaim() is how a non-owner keeps checking, since the kernel frees the
 * election when the owner dies and someone has to take the endpoint over.
 *
 * The outcomes are reported rather than logged: what a lost election means to
 * the user depends on which process it is and what it was configured with,
 * which is the exporter's business, not the mechanism's.
 */
class scrapeEndpoint {
public:
    enum class status {
        SERVING, ///< This process won the election and is serving the endpoint.
        SIBLING_OWNS, ///< Another process of this directory holds the election.
        PORT_TAKEN, ///< Won, but something outside the run holds the port.
    };

    scrapeEndpoint(std::filesystem::path dir,
                   std::string bind_address,
                   std::chrono::nanoseconds stale_ttl);

    /**
     * @brief Runs the election once and serves the endpoint if it wins.
     * @throw std::exception if the exposer fails for any reason other than the
     *        port being taken -- a genuine error, unlike a bind collision.
     */
    status
    claim();

    /**
     * @brief Re-runs the election of a non-owner, at most once per retry
     *        interval, and serves if it now wins.
     * @param now_ns A recent monotonicNs() reading, so the throttle costs the
     *        caller no clock of its own.
     *
     * Quiet and non-throwing: it runs on the export path, where a failure to
     * take over is the status quo rather than news.
     */
    void
    reclaim(uint64_t now_ns) noexcept;

    [[nodiscard]] bool
    serving() const noexcept {
        return static_cast<bool>(exposer_);
    }

    /// The endpoint the current owner published, empty if there is none.
    [[nodiscard]] std::string
    ownerEndpoint() const {
        return election_.publishedEndpoint();
    }

    /// The endpoint this process serves, or would serve if it won.
    [[nodiscard]] const std::string &
    bindAddress() const noexcept {
        return bindAddress_;
    }

private:
    // Binds and starts aggregating; false when the port is held elsewhere, in
    // which case the election is conceded rather than held for the whole run.
    [[nodiscard]] bool
    serve();

    std::filesystem::path dir_;
    std::string bindAddress_;
    std::chrono::nanoseconds staleTtl_;
    uint64_t lastAttemptNs_ = 0;
    uint64_t retryIntervalNs_;

    // Declared so destruction is exposer_ -> collector_ -> election_: stop
    // serving before dropping the collector it weak-references, and free the
    // port before releasing the election -- a rank that wins it in between
    // would otherwise fail to bind the port still held here.
    ownerElection election_;
    std::shared_ptr<nixlMultiprocessCollector> collector_;
    std::shared_ptr<prometheus::Exposer> exposer_;
};

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_SCRAPE_ENDPOINT_H
