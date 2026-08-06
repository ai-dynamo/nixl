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
#include "scrape_endpoint.h"

#include "common/nixl_log.h"

#include <exception>
#include <utility>

namespace nixl::telemetry::mp {

namespace {

    // How often a non-owner re-runs the election to notice that the owner died.
    // The election is one non-blocking flock, but it sits on the export path, so
    // it is throttled; the endpoint is down for this plus the scrape interval,
    // unless the rank that takes over cannot bind -- the port is held from
    // outside the run -- which puts that rank on backoff instead. A dead
    // owner does not hold it: civetweb sets SO_REUSEADDR, so its connections
    // left in TIME_WAIT do not block the taker.
    // A non-owner starts at 0, so its first exported event re-checks immediately
    // -- an owner that died between this process's election and its first event
    // is caught at once.
    constexpr std::chrono::nanoseconds retryInterval = std::chrono::milliseconds(200);

    // Once a re-election has won and still failed to bind, the port is held from
    // outside the run rather than by a rank of it, which is a condition that
    // lasts. Retrying an Exposer five times a second against it is the one costly
    // case, so back off; a transient conflict then clears in seconds instead of
    // milliseconds.
    constexpr std::chrono::nanoseconds backoff = std::chrono::seconds(5);

    // civetweb reports a failed port bind with this exact text (as used by the
    // single-process prometheus exporter). Only this case is treated as a benign
    // bind collision; any other Exposer failure is a genuine error.
    constexpr char bindFailureMarker[] = "Failed to setup server ports";

} // namespace

scrapeEndpoint::scrapeEndpoint(std::filesystem::path dir,
                               std::string bind_address,
                               std::chrono::nanoseconds stale_ttl)
    : dir_(std::move(dir)),
      bindAddress_(std::move(bind_address)),
      staleTtl_(stale_ttl),
      retryInterval_(retryInterval) {}

scrapeEndpoint::status
scrapeEndpoint::claim() {
    election_.emplace(dir_, bindAddress_);
    if (!election_->won()) {
        election_.reset();
        return status::SIBLING_OWNS;
    }
    return serve() ? status::SERVING : status::PORT_TAKEN;
}

void
scrapeEndpoint::reclaim(std::chrono::nanoseconds now) noexcept {
    // Re-electing while serving would give up the lock this process holds:
    // emplace destroys the election in place before running the new one.
    if (serving()) {
        return;
    }
    if (now - lastAttempt_ < retryInterval_) {
        return;
    }
    lastAttempt_ = now;
    try {
        // Quiet: an unusable lock file is the condition claim() already warned
        // about once, and repeating it every retry would bury the log.
        election_.emplace(dir_, bindAddress_, false);
        if (!election_->won()) {
            election_.reset();
            return;
        }
        if (!serve()) {
            election_.reset();
            retryInterval_ = backoff;
            return;
        }
        NIXL_INFO << "prometheus_mp: no process owns telemetry dir " << dir_.string()
                  << " any more; this one now serves " << bindAddress_;
    }
    catch (const std::exception &e) {
        // Telemetry must not break the export path this runs on. Winning the
        // election and then failing to serve for any other reason must still give
        // the election back, or this process holds it while serving nothing and
        // no other rank can take over.
        if (!serving()) {
            election_.reset();
        }
        NIXL_DEBUG << "prometheus_mp: cannot take over " << bindAddress_ << ": " << e.what();
    }
    catch (...) {
        if (!serving()) {
            election_.reset();
        }
        NIXL_DEBUG << "prometheus_mp: cannot take over " << bindAddress_;
    }
}

bool
scrapeEndpoint::serve() {
    try {
        auto exposer = std::make_shared<prometheus::Exposer>(bindAddress_);
        auto collector = std::make_shared<nixlMultiprocessCollector>(dir_, staleTtl_);
        exposer->RegisterCollectable(collector);
        collector_ = std::move(collector);
        exposer_ = std::move(exposer);
        return true;
    }
    catch (const std::exception &e) {
        if (std::string(e.what()).find(bindFailureMarker) == std::string::npos) {
            throw;
        }
        NIXL_DEBUG << "prometheus_mp: cannot bind " << bindAddress_ << ": " << e.what();
        // Concede: holding an election this process cannot act on would keep
        // every other rank from trying the port for the rest of the run.
        election_.reset();
        return false;
    }
}

} // namespace nixl::telemetry::mp
