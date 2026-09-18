/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#include "tcpxo_nic_common.h"

#include <arpa/inet.h>
#include <cstring>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "common/nixl_log.h"

namespace tcpxo {

std::vector<NetIf>
ExtractNetIfsFromString(absl::string_view string) {
    std::vector<NetIf> if_list;
    if (string.empty()) {
        return if_list;
    }

    for (absl::string_view segment : absl::StrSplit(string, ',', absl::SkipEmpty())) {
        size_t colon_pos = segment.find(':');
        if (colon_pos != absl::string_view::npos) {
            NetIf net_if;
            net_if.prefix = std::string(segment.substr(0, colon_pos));
            int port;
            if (absl::SimpleAtoi(segment.substr(colon_pos + 1), &port)) {
                net_if.port = port;
            }
            if_list.push_back(std::move(net_if));
        } else {
            if_list.push_back({std::string(segment), -1});
        }
    }
    return if_list;
}

bool
IsNetIfInList(absl::string_view string,
              int port,
              absl::Span<const NetIf> if_list,
              bool match_exact) {
    // Make an exception for the case where no user list is defined
    if (if_list.empty()) {
        return true;
    }
    for (const auto &net_if : if_list) {
        if (NetIfNamesMatch(string, net_if.prefix, match_exact) && PortsMatch(port, net_if.port)) {
            return true;
        }
    }
    return false;
}

std::string
SocketAddrToString(const struct sockaddr *saddr) {
    if (saddr == nullptr) {
        return "";
    }
    if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) {
        return "";
    }
    std::array<char, NI_MAXHOST> host;
    std::array<char, NI_MAXSERV> service;
    int rv = ::getnameinfo(saddr,
                           sizeof(SocketAddress),
                           host.data(),
                           NI_MAXHOST,
                           service.data(),
                           NI_MAXSERV,
                           NI_NUMERICHOST | NI_NUMERICSERV);
    if (rv != 0) {
        return "";
    }
    return absl::StrFormat("%s<%s>", host.data(), service.data());
}

std::string
SocketIpToString(const struct sockaddr *saddr) {
    if (saddr == nullptr) {
        return "";
    }
    std::array<char, NI_MAXHOST> host;
    int rv = ::getnameinfo(
        saddr, sizeof(SocketAddress), host.data(), NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
    if (rv != 0) {
        return "";
    }
    return std::string(host.data());
}

int
FindInterfacesFromString(absl::string_view prefix_list,
                         std::vector<std::string> &names,
                         std::vector<SocketAddress> &addrs,
                         int sock_family) {
    bool search_not = !prefix_list.empty() && prefix_list[0] == '^';
    if (search_not) {
        prefix_list.remove_prefix(1);
    }
    bool search_exact = !prefix_list.empty() && prefix_list[0] == '=';
    if (search_exact) {
        prefix_list.remove_prefix(1);
    }
    std::vector<NetIf> user_ifs = ExtractNetIfsFromString(prefix_list);

    int num_found = 0;
    int max_ifs = kMaxNetIfs;
    struct ifaddrs *interfaces, *interface;
    if (::getifaddrs(&interfaces) != 0) {
        return 0;
    }
    for (interface = interfaces; interface && num_found < max_ifs;
         interface = interface->ifa_next) {
        if (interface->ifa_addr == nullptr) {
            continue;
        }

        /* We only support IPv4 & IPv6 */
        int family = interface->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6) {
            continue;
        }

        NIXL_INFO << absl::StrFormat(
            "Found interface %s:%s", interface->ifa_name, SocketAddrToString(interface->ifa_addr));

        /* Allow the caller to force the socket family type */
        if (sock_family != -1 && family != sock_family) {
            continue;
        }

        /* We also need to skip IPv6 loopback and link-local interfaces */
        if (family == AF_INET6) {
            struct sockaddr_in6 *sa = (struct sockaddr_in6 *)(interface->ifa_addr);
            if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) {
                continue;
            }
            if (IN6_IS_ADDR_LINKLOCAL(&sa->sin6_addr)) {
                continue;
            }
        }

        // check against user specified interfaces
        if (!(IsNetIfInList(interface->ifa_name, -1, user_ifs, search_exact) ^ search_not)) {
            continue;
        }

        // Check that this interface has not already been saved
        // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
        if (std::find(names.begin(), names.end(), interface->ifa_name) == names.end()) {
            // Store the interface name
            names[num_found] = interface->ifa_name;
            // Store the IP address
            int sa_len = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
            memcpy(&addrs[num_found], interface->ifa_addr, sa_len);
            num_found++;
        }
    }

    ::freeifaddrs(interfaces);
    return num_found;
}

bool
SubnetsMatch(const struct ifaddrs &local_if, const SocketAddress *remote) {
    /* Check family first */
    int family = local_if.ifa_addr->sa_family;
    if (family != remote->sa.sa_family) {
        return false;
    }

    if (family == AF_INET) {
        struct sockaddr_in *local_addr = (struct sockaddr_in *)(local_if.ifa_addr);
        struct sockaddr_in *mask = (struct sockaddr_in *)(local_if.ifa_netmask);
        const struct sockaddr_in &remote_addr = remote->sin;
        struct in_addr local_subnet, remote_subnet;
        local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
        remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
        return local_subnet.s_addr == remote_subnet.s_addr;
    } else if (family == AF_INET6) {
        struct sockaddr_in6 *local_addr = (struct sockaddr_in6 *)(local_if.ifa_addr);
        struct sockaddr_in6 *mask = (struct sockaddr_in6 *)(local_if.ifa_netmask);
        const struct sockaddr_in6 &remote_addr = remote->sin6;
        const struct in6_addr &local_in6 = local_addr->sin6_addr;
        const struct in6_addr &mask_in6 = mask->sin6_addr;
        const struct in6_addr &remote_in6 = remote_addr.sin6_addr;
        bool same = true;
        for (int c = 0; c < 16; c++) { // Network byte order is big-endian
            if ((local_in6.s6_addr[c] & mask_in6.s6_addr[c]) !=
                (remote_in6.s6_addr[c] & mask_in6.s6_addr[c])) {
                same = false;
                break;
            }
        }
        // At last, we need to compare scope id
        // Two Link-type addresses can have the same subnet address even though they
        // are not in the same scope For Global type, this field is 0, so a
        // comparison wouldn't matter
        same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
        return same;
    } else {
        NIXL_WARN << "Net : Unsupported address family type";
        return false;
    }
}

int
FindInterfacesMatchingRemoteSubnet(std::vector<std::string> &if_names,
                                   std::vector<SocketAddress> &local_addrs,
                                   const SocketAddress *remote_addr) {
    int num_found = 0;
    int max_ifs = kMaxNetIfs;
    struct ifaddrs *interfaces, *interface;
    if (::getifaddrs(&interfaces) != 0) {
        return 0;
    }
    for (interface = interfaces; interface && num_found < max_ifs;
         interface = interface->ifa_next) {
        if (interface->ifa_addr == nullptr) {
            continue;
        }

        /* We only support IPv4 & IPv6 */
        int family = interface->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6) {
            continue;
        }

        // check against user specified interfaces
        if (!SubnetsMatch(*interface, remote_addr)) {
            continue;
        }

        // Store the local IP address
        int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
        memcpy(&local_addrs[num_found], interface->ifa_addr, salen);

        // Store the interface name
        if_names[num_found] = interface->ifa_name;

        NIXL_INFO << absl::StrFormat(
            "NET : Found interface %s:%s in the same subnet as remote address %s",
            interface->ifa_name,
            SocketAddrToString(&local_addrs[num_found].sa),
            SocketAddrToString(&remote_addr->sa));
        num_found++;
    }

    if (num_found == 0) {
        NIXL_WARN << absl::StrFormat(
            "Net : No interface num_found in the same subnet as remote address %s",
            SocketAddrToString(&remote_addr->sa));
    }
    ::freeifaddrs(interfaces);
    return num_found;
}

absl::Status
GetSocketAddrFromString(SocketAddress *ua, absl::string_view ip_port_pair) {
    if (ip_port_pair.size() <= 1) {
        absl::string_view warn_msg("Net : string is null or too short");
        NIXL_WARN << warn_msg;
        return absl::InvalidArgumentError(warn_msg);
    }

    bool ipv6 = ip_port_pair[0] == '[';
    /* Construct the sockaddress structure */
    if (!ipv6) {
        std::vector<NetIf> ni_list = ExtractNetIfsFromString(ip_port_pair);
        // parse <ip_or_hostname>:<port> string, expect one pair
        if (ni_list.empty()) {
            absl::string_view warn_msg("Net : No valid <IPv4_or_hostname>:<port> pair found");
            NIXL_WARN << warn_msg;
            return absl::InvalidArgumentError(warn_msg);
        }
        const auto &ni = ni_list[0];

        struct addrinfo hints, *p;
        int rv;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;

        if ((rv = ::getaddrinfo(ni.prefix.c_str(), nullptr, &hints, &p)) != 0) {
            auto warn_msg = absl::StrFormat(
                "Net : error encountered when getting address info : %s", ::gai_strerror(rv));
            NIXL_WARN << warn_msg;
            return absl::InvalidArgumentError(warn_msg);
        }

        // use the first
        if (p->ai_family == AF_INET) {
            struct sockaddr_in &sin = ua->sin;
            memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
            sin.sin_family = AF_INET; // IPv4
            sin.sin_port = htons(ni.port); // port
        } else if (p->ai_family == AF_INET6) {
            struct sockaddr_in6 &sin6 = ua->sin6;
            memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
            sin6.sin6_family = AF_INET6; // IPv6
            sin6.sin6_port = htons(ni.port); // port
            sin6.sin6_flowinfo = 0; // needed by IPv6, but possibly obsolete
            sin6.sin6_scope_id = 0; // should be global scope, set to 0
        } else {
            ::freeaddrinfo(p);
            absl::string_view warn_msg("Net : unsupported IP family");
            NIXL_WARN << warn_msg;
            return absl::InvalidArgumentError(warn_msg);
        }
        ::freeaddrinfo(p); // all done with this structure
    } else {
        size_t i, j = absl::string_view::npos;
        size_t len = ip_port_pair.size();
        for (i = 1; i < len; i++) {
            if (ip_port_pair[i] == '%') {
                j = i;
            }
            if (ip_port_pair[i] == ']') {
                break;
            }
        }
        if (i == len) {
            absl::string_view warn_msg("Net : No valid [IPv6]:port pair found");
            NIXL_WARN << warn_msg;
            return absl::InvalidArgumentError(warn_msg);
        }
        bool global_scope = (j == absl::string_view::npos);

        std::string ip_str(ip_port_pair.substr(1, global_scope ? i - 1 : j - 1));
        std::string port_str;
        if (i + 2 < len) {
            port_str = std::string(ip_port_pair.substr(i + 2));
        }
        int port = 0;
        if (!absl::SimpleAtoi(port_str, &port)) {
            absl::string_view warn_msg("Can not get valid port number");
            NIXL_WARN << warn_msg;
            return absl::InvalidArgumentError(warn_msg);
        }

        std::string if_name;
        if (!global_scope) {
            if_name = std::string(ip_port_pair.substr(j + 1, i - j - 1));
        }

        struct sockaddr_in6 &sin6 = ua->sin6;
        sin6.sin6_family = AF_INET6; // IPv6
        inet_pton(AF_INET6, ip_str.c_str(), &(sin6.sin6_addr)); // IP address
        sin6.sin6_port = htons(port); // port
        sin6.sin6_flowinfo = 0; // needed by IPv6, but possibly obsolete
        sin6.sin6_scope_id =
            global_scope ? 0 : ::if_nametoindex(if_name.c_str()); // 0 if global scope; intf index
                                                                  // if link scope
    }
    return absl::OkStatus();
}

int
DiscoverBestInterfaces(std::vector<std::string> &if_names,
                       std::vector<SocketAddress> &if_addrs,
                       int sock_family,
                       absl::string_view socket_ifname,
                       absl::string_view comm_id) {
    int num_ifs = 0;
    // User specified interface
    if (socket_ifname.size() > 1) {
        NIXL_INFO << absl::StrFormat("FASTRAK_SOCKET_IFNAME set by environment to %s",
                                     socket_ifname);
        num_ifs = FindInterfacesFromString(socket_ifname, if_names, if_addrs, sock_family);
    } else {
        // Try to automatically pick the right one
        // Start with IB
        num_ifs = FindInterfacesFromString("ib", if_names, if_addrs, sock_family);
        // else see if we can get some hint from COMM ID
        if (num_ifs == 0 && comm_id.size() > 1) {
            NIXL_INFO << absl::StrFormat("NIXL_COMM_ID set by environment to %s", comm_id);
            // Try to find interface that is in the same subnet as the IP in comm id
            SocketAddress id_addr;
            GetSocketAddrFromString(&id_addr, comm_id).IgnoreError();
            num_ifs = FindInterfacesMatchingRemoteSubnet(if_names, if_addrs, &id_addr);
        }
        // Then look for anything else (but not docker or lo)
        if (num_ifs == 0) {
            num_ifs = FindInterfacesFromString("^docker,lo", if_names, if_addrs, sock_family);
        }
        // Finally look for docker, then lo.
        if (num_ifs == 0) {
            num_ifs = FindInterfacesFromString("docker", if_names, if_addrs, sock_family);
        }
        if (num_ifs == 0) {
            num_ifs = FindInterfacesFromString("lo", if_names, if_addrs, sock_family);
        }
    }
    return num_ifs;
}

} // namespace tcpxo
