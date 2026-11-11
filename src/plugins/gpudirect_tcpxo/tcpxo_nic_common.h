/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_NIC_COMMON_H
#define NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_NIC_COMMON_H

#include <ifaddrs.h>

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "tcpxo_common.h"

namespace tcpxo {

struct NetIf {
    std::string prefix;
    int port = -1;
};

std::vector<NetIf> ExtractNetIfsFromString(absl::string_view);

inline bool
NetIfNamesMatch(absl::string_view string, absl::string_view ref, bool match_exact) {
    if (match_exact) {
        return string == ref;
    }
    return absl::StartsWith(string, ref);
}

inline bool
PortsMatch(int port1, int port2) {
    if (port1 == -1 || port2 == -1) {
        return true;
    }
    return port1 == port2;
}

bool
IsNetIfInList(absl::string_view, int, absl::Span<const NetIf>, bool);

/* Format a string representation of a (struct sockaddr *) socket address using
 * ::getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
std::string
SocketAddrToString(const struct sockaddr *);

inline std::string
SocketAddrToString(const SocketAddress *saddr) {
    if (saddr == nullptr) {
        return "";
    }
    return SocketAddrToString(&saddr->sa);
}

// Similar to SocketAddrToString, but only formats IP addresses, without the ports.
std::string
SocketIpToString(const struct sockaddr *);

int
FindInterfacesFromString(absl::string_view,
                         std::vector<std::string> &,
                         std::vector<SocketAddress> &,
                         int);

bool
SubnetsMatch(const struct ifaddrs &, const SocketAddress *);

int
FindInterfacesMatchingRemoteSubnet(std::vector<std::string> &,
                                   std::vector<SocketAddress> &,
                                   const SocketAddress *);

absl::Status
GetSocketAddrFromString(SocketAddress *, absl::string_view);

int
DiscoverBestInterfaces(std::vector<std::string> &,
                       std::vector<SocketAddress> &,
                       int,
                       absl::string_view,
                       absl::string_view);

} // namespace tcpxo

#endif // NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_NIC_COMMON_H
