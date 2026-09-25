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

#include "dxs_endpoint.h"

#include <sys/socket.h>
#include <climits>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <google/protobuf/repeated_ptr_field.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "common/nixl_log.h"
#include "absl/log/log.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "buffer_mgmt_daemon/client/buffer_mgr_client.h"
#include "buffer_mgmt_daemon/pci_utils.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/dxs-client.h"
#include "dxs/client/oss/status_macros.h" // for ASSIGN_OR_RETURN, RETURN_IF_ERROR
#else
#include "rxdm_dxs_stub.h"
#endif
#include "nixl_cuda/cuda_common.h" // IWYU: pragma keep
#include "tcpxo_common.h"
#include "tcpxo_nic_common.h" // IWYU: pragma keep

namespace tcpxo {

namespace {

    std::string
    GetPciPath(absl::string_view dev_name) {
        auto device_path = absl::StrFormat("/sys/class/net/%s/device", dev_name);
        std::array<char, PATH_MAX> real_path;
        if (realpath(device_path.c_str(), real_path.data()) == nullptr) {
            return "";
        }
        return std::string(real_path.data());
    }

} // namespace

// static
absl::StatusOr<std::unique_ptr<DxsEndpoint>>
DxsEndpoint::Create(absl::string_view nic_dev_name,
                    absl::string_view nic_pci_path,
                    uint8_t fastrak_idx,
                    GpuDev &&gpu_dev,
                    absl::string_view ip_addr,
                    const DxsEndpointConfig &config) {
    auto endpoint = absl::WrapUnique(
        new DxsEndpoint(nic_dev_name, nic_pci_path, fastrak_idx, std::move(gpu_dev)));
    ASSIGN_OR_RETURN(endpoint->buffer_manager_client_,
                     tcpdirect::BufferManagerClient::Create(ip_addr));
    ASSIGN_OR_RETURN(endpoint->dxs_client_,
                     dxs::DxsClient::Create(std::string(ip_addr),
                                            dxs::kDefaultDxsAddr,
                                            dxs::kDefaultDxsPort,
                                            "0",
                                            config.use_llcm,
                                            config.llcm_device_directory,
                                            config.close_send_on_done));
    return endpoint;
}

absl::StatusOr<DxsConnection>
DxsEndpoint::Listen(uint64_t max_num_flows_per_dxs_conn, absl::Duration dxs_listen_timeout_ms) {
    std::vector<DxsFlow> dxs_flows(max_num_flows_per_dxs_conn);
    for (auto &dxs_flow : dxs_flows) {
        ASSIGN_OR_RETURN(std::unique_ptr<dxs::ListenSocketInterface> dxs_listen_sock,
                         dxs_client_->Listen());
        std::optional<absl::Status> ready = dxs_listen_sock->SocketReady();
        const auto start_time = absl::Now();
        while (!ready.has_value()) {
            if (absl::Now() - start_time > dxs_listen_timeout_ms) {
                return absl::DeadlineExceededError("Listen timed out for device");
            }
            ready = dxs_listen_sock->SocketReady();
        }
        RETURN_IF_ERROR(*std::move(ready));
        dxs_flow.listen_socket = std::move(dxs_listen_sock);
    }
    return DxsConnection(std::move(dxs_flows));
}

// static
absl::StatusOr<std::unique_ptr<DxsEndpointManager>>
DxsEndpointManager::InitializeNetIfs(bool loopback_only,
                                     absl::string_view ifname,
                                     absl::string_view sock_family_str,
                                     absl::string_view socket_ifname,
                                     absl::string_view comm_id,
                                     absl::string_view ctrl_dev,
                                     bool use_llcm,
                                     bool close_send_on_done,
                                     absl::string_view llcm_dev_directory) {
    std::vector<std::string> names(kMaxNetIfs);
    std::vector<SocketAddress> addrs(kMaxNetIfs);

#ifndef TCPXO_STUB_RXDM_DXS
    int sock_family = -1;
    int num_net_ifs_found = 0;
    if (loopback_only) {
        // Only use lo interface for loopback tests if env variable is set
        num_net_ifs_found = FindInterfacesFromString("lo", names, addrs, sock_family);
    } else if (!ifname.empty()) {
        NIXL_DEBUG << "FASTRAK_IFNAME: " << ifname;
        num_net_ifs_found = FindInterfacesFromString(ifname, names, addrs, sock_family);
    } else {
        // Allow user to force the INET socket family selection
        if (sock_family_str == "AF_INET") {
            sock_family = AF_INET;
        } else if (sock_family_str == "AF_INET6") {
            sock_family = AF_INET6;
        }
        num_net_ifs_found =
            DiscoverBestInterfaces(names, addrs, sock_family, socket_ifname, comm_id);
    }
    NIXL_DEBUG << "Found " << num_net_ifs_found << " network interfaces";

    if (num_net_ifs_found <= 0) {
        absl::string_view err_msg("No network interfaces found.");
        NIXL_ERROR << err_msg;
        return absl::FailedPreconditionError(err_msg);
    }

    std::vector<std::string> ctrl_dev_name(1);
    std::vector<SocketAddress> ctrl_socket_addr(1);
    int num_ctrl_dev_found =
        FindInterfacesFromString(ctrl_dev, ctrl_dev_name, ctrl_socket_addr, -1);
    if (num_ctrl_dev_found != 1) {
        auto err_msg = absl::StrFormat("No interfaces found for ctrl dev name %s", ctrl_dev);
        NIXL_ERROR << err_msg;
        return absl::FailedPreconditionError(err_msg);
    }
    ctrl_dev_name[0].resize(kMaxNetIfNameLen);

    ASSIGN_OR_RETURN(auto pci_and_dev_mapping, DiscoverPciAndDevMapping());
#else
    int num_net_ifs_found = 2;
    std::vector<std::string> ctrl_dev_name(1, std::string(ctrl_dev));
    std::vector<SocketAddress> ctrl_socket_addr(1);
    PciAndDevMapping pci_and_dev_mapping;
#endif

    auto endpoint_manager = std::make_unique<DxsEndpointManager>(
        num_net_ifs_found,
        std::move(ctrl_dev_name.front()),
        std::move(ctrl_socket_addr.front()),
        DxsEndpointConfig{
            .use_llcm = use_llcm,
            .close_send_on_done = close_send_on_done,
            .llcm_device_directory = std::string(llcm_dev_directory),
        },
        std::move(pci_and_dev_mapping));
    RETURN_IF_ERROR(endpoint_manager->InitializeAllEndpoints(addrs, names));
    return endpoint_manager;
}

absl::Status
DxsEndpointManager::InitializeAllEndpoints(absl::Span<const SocketAddress> addrs,
                                           absl::Span<const std::string> names) {
    // Get the GPU->NIC mapping from RxDM (based on NCCL Shim logic)
    auto nic_mapping_resp = tcpdirect::get_nic_mapping();
    if (!nic_mapping_resp.has_value()) {
        absl::string_view err_msg("Could not obtain NIC mappings from RxDM");
        NIXL_ERROR << err_msg;
        return absl::FailedPreconditionError(err_msg);
    }

#ifndef TCPXO_STUB_RXDM_DXS
    std::string log_line;
    size_t num_net_ifs_found =
        std::count_if(addrs.begin(), addrs.end(), [](const SocketAddress &addr) {
            return addr.sa.sa_family == AF_INET || addr.sa.sa_family == AF_INET6;
        });

    // Build IP -> NIC info map for easy lookup
    absl::flat_hash_map<std::string, std::pair<std::string, std::string>> ip_to_nic;
    for (size_t i = 0; i < num_net_ifs_found; ++i) {
        const std::string &dev_name = names[i];
        std::string pci_path = GetPciPath(dev_name);
        std::string ip_addr = SocketIpToString(&addrs[i].sa);
        ip_to_nic[ip_addr] = {pci_path, dev_name};
        if (!log_line.empty()) {
            absl::StrAppend(&log_line, ", ");
        }
        absl::StrAppend(&log_line,
                        absl::StrFormat("[%d]%s:%s", i, dev_name, SocketAddrToString(&addrs[i])));
    }

    // Initialize endpoints based on GPU->NIC mapping
    for (const auto &[gpu_pci, nic_info] : nic_mapping_resp->pci_nic_map()) {
        if (nic_info.closest_nic_ip().empty()) {
            const auto err_msg =
                absl::StrFormat("Missing closest NIC for GPU PCI address %s", gpu_pci);
            NIXL_ERROR << err_msg;
            return absl::FailedPreconditionError(err_msg);
        }
        const std::string ip_addr = nic_info.closest_nic_ip().Get(0);
        if (ip_to_nic.contains(ip_addr)) {
            const auto &[pci_path, dev_name] = ip_to_nic[ip_addr];

            auto gpu_dev_status = InitGpuDev(gpu_pci);
            if (!gpu_dev_status.ok()) {
                if (gpu_dev_status.status().code() == absl::StatusCode::kNotFound) {
                    NIXL_DEBUG << "Skipping GPU PCI address " << gpu_pci
                               << " as it is not accessible via CUDA: "
                               << gpu_dev_status.status().message();
                    continue;
                }
                NIXL_ERROR << "Failed to initialize GPU " << gpu_pci << ": "
                           << gpu_dev_status.status();
                return gpu_dev_status.status();
            }
            GpuDev gpu_dev = std::move(*gpu_dev_status);

            ASSIGN_OR_RETURN(uint8_t fastrak_idx, GetFastrakIdxFromPci(gpu_pci));
            ASSIGN_OR_RETURN(std::unique_ptr<DxsEndpoint> endpoint,
                             DxsEndpoint::Create(dev_name,
                                                 std::move(pci_path),
                                                 fastrak_idx,
                                                 std::move(gpu_dev),
                                                 ip_addr,
                                                 config_));
            dxs_addr_to_endpoint_[ip_addr] = endpoint.get();
            fastrak_idx_to_endpoint_[fastrak_idx] = endpoint.get();
            endpoints_.push_back(std::move(endpoint));
        } else {
            NIXL_WARN << "No network interface found for GPU " << gpu_pci
                      << " (closest IP address: " << ip_addr << ")";
        }
    }

    if (endpoints_.empty()) {
        const auto err_msg = "No GPUs are visible.";
        NIXL_ERROR << err_msg;
        return absl::FailedPreconditionError(err_msg);
    }

    NIXL_INFO << "Using " << log_line;
#else
    int num_net_ifs_found = 2;
    for (auto i = 0; i < num_net_ifs_found; ++i) {
        GpuDev gpu_dev;
        auto dev_name = absl::StrCat("mock", i);
        std::string pci_path = GetPciPath(dev_name);
        auto ip_addr = absl::StrCat("192.168.0.", i);

        pci_and_dev_mapping_.dev_id_to_fastrak_idx[i] = i;

        ASSIGN_OR_RETURN(
            std::unique_ptr<DxsEndpoint> endpoint,
            DxsEndpoint::Create(dev_name, pci_path, i, std::move(gpu_dev), ip_addr, config_));
        dxs_addr_to_endpoint_[ip_addr] = endpoint.get();
        fastrak_idx_to_endpoint_[i] = endpoint.get();
        endpoints_.push_back(std::move(endpoint));
    }
#endif
    return absl::OkStatus();
}

absl::StatusOr<PciAndDevMapping>
DxsEndpointManager::DiscoverPciAndDevMapping() {
    PciAndDevMapping result;
    std::vector<std::string> gpus;
    if (tcpdirect::list_vendor_devices(tcpdirect::kSysfsPciDevicesPath,
                                       gpus,
                                       tcpdirect::kNvidiaVendorId,
                                       tcpdirect::kH100DeviceId) < 0) {
        auto err_msg =
            absl::StrFormat("Failed to list H100 GPUs under [%s]", tcpdirect::kSysfsPciDevicesPath);
        NIXL_ERROR << err_msg;
        return absl::FailedPreconditionError(err_msg);
    }

    // Lexicographic order matches enumeration order.
    std::sort(gpus.begin(), gpus.end());
    // Remove duplicate entries
    gpus.erase(std::unique(gpus.begin(), gpus.end()), gpus.end());
    if (gpus.empty()) {
        auto err_msg = "DiscoverPciAndDevMapping: No GPUs found.";
        NIXL_ERROR << err_msg;
        return absl::FailedPreconditionError(err_msg);
    }

    NIXL_INFO << "DiscoverPciAndDevMapping: Found GPUs:";
    for (auto idx = 0u; idx < gpus.size(); idx++) {
        const auto bdf = absl::AsciiStrToLower(gpus[idx]);
        NIXL_INFO << absl::StrFormat("\tFasTrak IDX: [%d]. PCI addr: [%s]", idx, bdf);
        result.pci_addr_to_fastrak_idx.insert({bdf, static_cast<uint8_t>(idx)});
    }

#ifdef HAVE_CUDA
    ASSIGN_OR_RETURN(const auto num_devices, GetDeviceCount());
    for (int cuda_dev_id = 0; cuda_dev_id < num_devices; ++cuda_dev_id) {
        ASSIGN_OR_RETURN(const auto pci_bus_id, GetPciBusIdFromCudaDevId(cuda_dev_id));
        auto bdf = absl::AsciiStrToLower(pci_bus_id);
        // Log both to debug formatting mismatches
        NIXL_INFO << absl::StrFormat(
            "CUDA Dev ID: %d, PCI addr: %s (lowercased: %s)", cuda_dev_id, pci_bus_id, bdf);
        auto it = result.pci_addr_to_fastrak_idx.find(bdf);
        if (it == result.pci_addr_to_fastrak_idx.end()) {
            return absl::NotFoundError(
                absl::StrFormat("CUDA Dev ID %d PCI address %s not found in FasTrak PCI mapping.",
                                cuda_dev_id,
                                bdf));
        }
        result.dev_id_to_fastrak_idx.insert({cuda_dev_id, it->second});
    }
#endif
    return result;
}

absl::StatusOr<uint8_t>
DxsEndpointManager::GetFastrakIdxFromPci(absl::string_view pci_addr) {
    auto sanitized_pci_addr = absl::AsciiStrToLower(pci_addr);
    auto it = pci_and_dev_mapping_.pci_addr_to_fastrak_idx.find(sanitized_pci_addr);
    if (it == pci_and_dev_mapping_.pci_addr_to_fastrak_idx.end()) {
        auto err_msg =
            absl::StrFormat("No FasTrak index found for PCI addr %s", sanitized_pci_addr);
        NIXL_ERROR << err_msg;
        return absl::FailedPreconditionError(err_msg);
    }
    return it->second;
}

absl::StatusOr<DxsEndpoint * absl_nonnull>
DxsEndpointManager::GetEndpoint(absl::string_view nic_addr) {
    auto it = dxs_addr_to_endpoint_.find(std::string(nic_addr));
    if (it == dxs_addr_to_endpoint_.end()) {
        return absl::NotFoundError(
            absl::StrFormat("Endpoint with NIC addr %s not found", nic_addr));
    }
    return it->second;
}

absl::StatusOr<DxsEndpoint * absl_nonnull>
DxsEndpointManager::GetEndpoint(uint8_t fastrak_idx) {
    auto it = fastrak_idx_to_endpoint_.find(fastrak_idx);
    if (it == fastrak_idx_to_endpoint_.end()) {
        return absl::NotFoundError(
            absl::StrFormat("Endpoint with fastrak idx %d not found", fastrak_idx));
    }
    return it->second;
}

absl::StatusOr<uint8_t>
DxsEndpointManager::GetFastrakIdxFromDevId(int dev_id) const {
    auto it = pci_and_dev_mapping_.dev_id_to_fastrak_idx.find(dev_id);
    if (it == pci_and_dev_mapping_.dev_id_to_fastrak_idx.end()) {
        auto err_msg = absl::StrFormat("No FasTrak index found for device ID %d", dev_id);
        NIXL_ERROR << err_msg;
        return absl::NotFoundError(err_msg);
    }
    return it->second;
}

ConnectionTraceId
GenerateNextConnectionTraceId() {
    static std::atomic<uint32_t> next_id{0};

    auto id = next_id.fetch_add(1, std::memory_order_relaxed);
    while (id == static_cast<uint32_t>(ConnectionTraceId::kInvalid)) {
        id = next_id.fetch_add(1, std::memory_order_relaxed);
    }

    return static_cast<ConnectionTraceId>(id);
}

} // namespace tcpxo
