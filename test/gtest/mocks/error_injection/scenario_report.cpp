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

#include "mocks/error_injection/scenario_report.h"

#include <iostream>
#include <string>

namespace mocks::error_injection {

namespace {

    /* Fits NIXL_ERR_REMOTE_DISCONNECT, the widest status name at 26 characters. */
    constexpr size_t column_width = 30;
    constexpr size_t column_gap = 4;
    /* "transformed to NIXL_ERR_REPOST_ACTIVE", only used to size the header rule. */
    constexpr size_t longest_verdict = 37;

    /*
     * Pads to a fixed column, but never below column_gap trailing spaces: std::setw
     * drops the separator entirely once a value grows past the field width, which
     * silently runs neighboring cells together.
     */
    std::string
    column(const std::string &value) {
        const size_t padding =
            column_width > value.size() + column_gap ? column_width - value.size() : column_gap;
        return value + std::string(padding, ' ');
    }

    /*
     * Describes what the agent did with the injected status. Driven by the
     * scenario's declared behavior, not by comparing the returned status to the
     * injected one, so that a site collapsing NIXL_ERR_BACKEND to NIXL_ERR_BACKEND
     * is still reported as a collapse.
     */
    std::string
    verdictText(const scenario &s, const observation &obs) {
        const std::string observed_status = nixlEnumStrings::statusStr(obs.status);

        if (obs.status != s.expected) {
            return "FAIL: expected " + std::string(nixlEnumStrings::statusStr(s.expected));
        }
        if (!obs.failure.empty()) {
            return "FAIL: " + obs.failure;
        }

        switch (s.behavior) {
        case behavior_t::COMPLETED:
            return "-";
        case behavior_t::PASS_THROUGH:
            return "pass-through";
        case behavior_t::PASS_THROUGH_RECOVERABLE:
            return "pass-through + remote retained";
        case behavior_t::COLLAPSED:
            return "collapsed to " + observed_status;
        case behavior_t::TRANSFORMED:
            return "transformed to " + observed_status;
        case behavior_t::IGNORED:
            return "ignored";
        case behavior_t::PROGRESSED:
            return "progressed to " + observed_status;
        case behavior_t::AGENT_GENERATED:
            return "agent-generated";
        }
        return "(unknown)";
    }

} // namespace

bool
passed(const scenario &s, const observation &obs) {
    return obs.status == s.expected && obs.failure.empty();
}

void
printHeader(bool csv) {
    if (csv) {
        std::cout << "sb site,injected,agent stopped at,agent returned,verdict\n";
        return;
    }

    std::cout << "Injecting southbound errors into " << mock_backend_name << "\n\n"
              << column("sb site") << column("injected") << column("agent stopped at")
              << column("agent returned") << "verdict\n"
              << std::string(4 * column_width + longest_verdict, '-') << "\n";
}

void
printRow(const scenario &s, const observation &obs, bool csv) {
    const std::string injected_status =
        s.site == injection_site_t::NONE ? "-" : nixlEnumStrings::statusStr(s.injected);
    const std::string observed_status = nixlEnumStrings::statusStr(obs.status);
    const std::string verdict = verdictText(s, obs);

    if (csv) {
        std::cout << siteName(s.site) << "," << injected_status << "," << obs.stage << ","
                  << observed_status << "," << verdict << "\n";
    } else {
        std::cout << column(siteName(s.site)) << column(injected_status) << column(obs.stage)
                  << column(observed_status) << verdict << "\n";
    }
}

void
printSummary(size_t selected, size_t failures, bool csv) {
    if (!csv) {
        std::cout << "\n" << selected << " scenarios, " << failures << " failures\n";
    }
}

} // namespace mocks::error_injection
