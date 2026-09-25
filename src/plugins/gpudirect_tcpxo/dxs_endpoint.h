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

#ifndef NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_DXS_ENDPOINT_H
#define NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_DXS_ENDPOINT_H

#include <cstdint>

#include <atomic>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "control_channel.pb.h"
#ifndef TCPXO_STUB_RXDM_DXS
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client.h"
#include "dxs/client/dxs-client-interface.h"
#else
#include "rxdm_dxs_stub.h"
#endif
#include "tcpxo_common.h"

namespace tcpxo {

// A flow is essentially a data stream between two DXS endpoints. We can send/recv data on multiple
// flows in parallel
struct DxsFlow {
    // The ListenSocketInterface specifies the address and port of a particular socket. We make a
    // socket for each flow. The reason is that connection stalls were encountered when using a
    // single socket for all flows (see the blame of NcclShim::Listen)
    absl_nonnull std::unique_ptr<dxs::ListenSocketInterface> listen_socket;
    // This is created after a successful accept, and thus will be null until then.
    std::unique_ptr<dxs::RecvSocketInterface> recv_socket{nullptr};
    // This is created after a successful connect, and thus will be null until then.
    std::unique_ptr<dxs::SendSocketInterface> send_socket{nullptr};
    uint64_t flow_num{0};
};

struct DxsEndpointConfig {
    bool use_llcm = true;
    bool close_send_on_done = false;
    std::string llcm_device_directory = std::string(dxs::kLlcmDeviceDirectory);
};

enum class ConnectionTraceId : uint32_t {
    kInvalid = 0,
};

ConnectionTraceId
GenerateNextConnectionTraceId();

/**
 * @brief All the flows for a particular connection.
 * @details
 * Each flow represents two real DXS connections. Each individual, real DXS connection is
 * uni-directional, so we make two real DXS connections for bi-directional communication.
 */
class DxsConnection {
public:
    explicit DxsConnection(std::vector<DxsFlow> flows)
        : flows_(std::move(flows)),
          last_flow_used_(std::make_unique<std::atomic<uint8_t>>(0)),
          connection_id_(GenerateNextConnectionTraceId()) {}

    inline uint8_t
    GetNextFlow() {
        return last_flow_used_->fetch_add(1) % flows_.size();
    }

    inline const std::vector<DxsFlow> &
    flows() const {
        return flows_;
    }

    inline std::vector<DxsFlow> &
    flows() {
        return flows_;
    }

    inline uint8_t
    last_flow_used() {
        return *last_flow_used_;
    }

    inline void
    set_last_flow_used(uint8_t last_flow_used) {
        last_flow_used_->store(last_flow_used);
    }

    inline ConnectionTraceId
    connection_id() {
        return connection_id_;
    }

private:
    std::vector<DxsFlow> flows_;
    // The flow last used to send/recv data
    // Wrapped with std::unique_ptr to make DxsConnection movable
    absl_nonnull std::unique_ptr<std::atomic<uint8_t>> last_flow_used_;
    // Used for logging purposes
    ConnectionTraceId connection_id_;
};

// A GPU<->NIC pairing, and all the resources needed therein. This class is responsible for
// establishing all flows between two endpoints.
class DxsEndpoint {
public:
    static absl::StatusOr<std::unique_ptr<DxsEndpoint>>
    Create(absl::string_view,
           absl::string_view,
           uint8_t,
           GpuDev &&,
           absl::string_view,
           const DxsEndpointConfig &);

    virtual ~DxsEndpoint() = default;

    virtual absl::StatusOr<DxsConnection>
    Listen(uint64_t max_num_flows_per_dxs_conn, absl::Duration dxs_listen_timeout_ms);

    absl::Status
    AcceptAndWait(DxsConnection &, const IPAddress &) {
        return absl::UnimplementedError("AcceptAndWait is unimplemented");
    }

    absl::Status
    ConnectAndWait(DxsConnection &, const DxsAddress &) {
        return absl::UnimplementedError("ConnectAndWait is unimplemented");
    }

    absl::Status
    Disconnect(DxsConnection &) {
        return absl::UnimplementedError("Disconnect is unimplemented");
    }

    tcpdirect::BufferManagerClientInterface *absl_nonnull
    GetBufferManagerClient() {
        return buffer_manager_client_.get();
    }

    inline uint8_t
    fastrak_idx() const {
        return fastrak_idx_;
    }

    dxs::DxsClientInterface *absl_nonnull
    dxs_client() {
        return dxs_client_.get();
    }

protected:
    DxsEndpoint(absl::string_view nic_dev_name,
                absl::string_view nic_pci_path,
                uint8_t fastrak_idx,
                GpuDev &&gpu_dev)
        : nic_dev_name_(nic_dev_name),
          nic_pci_path_(nic_pci_path),
          fastrak_idx_(fastrak_idx),
          gpu_dev_(std::move(gpu_dev)) {}

    const std::string nic_dev_name_;
    const std::string nic_pci_path_;
    // The order in which the GPUs are enumerated by the host. Expected to be consistent across all
    // FasTrak deployments: go/fastrak-handle-change-por
    const uint8_t fastrak_idx_;
    const GpuDev gpu_dev_;

    absl_nonnull std::unique_ptr<dxs::DxsClientInterface> dxs_client_;
    absl_nonnull std::unique_ptr<tcpdirect::BufferManagerClientInterface> buffer_manager_client_;
};

struct PciAndDevMapping {
    absl::flat_hash_map<std::string, uint8_t> pci_addr_to_fastrak_idx;
    absl::flat_hash_map<int, uint8_t> dev_id_to_fastrak_idx;
};

// This class maintains a mapping of all endpoints. This is the entrypoint for retrieving
// DxsEndpoints, which gets you to the RxDM and DXS interfaces.
class DxsEndpointManager {
public:
    static absl::StatusOr<std::unique_ptr<DxsEndpointManager>>
    InitializeNetIfs(bool,
                     absl::string_view,
                     absl::string_view,
                     absl::string_view,
                     absl::string_view,
                     absl::string_view,
                     bool,
                     bool,
                     absl::string_view);

    explicit DxsEndpointManager(int num_net_ifs_found,
                                absl::string_view ctrl_dev_name,
                                SocketAddress &&ctrl_dev_addr,
                                DxsEndpointConfig &&config,
                                PciAndDevMapping &&pci_and_dev_mapping)
        : pci_and_dev_mapping_(std::move(pci_and_dev_mapping)),
          num_net_ifs_found_(num_net_ifs_found),
          ctrl_dev_name_(ctrl_dev_name),
          ctrl_dev_addr_(std::move(ctrl_dev_addr)),
          config_(std::move(config)) {
        endpoints_.reserve(num_net_ifs_found);
    }

    absl::Status InitializeAllEndpoints(absl::Span<const SocketAddress>,
                                        absl::Span<const std::string>);

    const std::vector<std::unique_ptr<DxsEndpoint>> &
    endpoints() const {
        return endpoints_;
    }

    // Get the endpoint based on the NIC address. May need to augment/change to support
    // retrieval by GPU PCI Address
    absl::StatusOr<DxsEndpoint * absl_nonnull> GetEndpoint(absl::string_view);

    // Get the endpoint based on the FasTrak index of the GPU
    absl::StatusOr<DxsEndpoint * absl_nonnull> GetEndpoint(uint8_t);

    // Get the FasTrak index from the CUDA device index
    absl::StatusOr<uint8_t>
    GetFastrakIdxFromDevId(int) const;

    void
    InjectDevIdMappingForTest(int dev_id, uint8_t fastrak_idx) {
        pci_and_dev_mapping_.dev_id_to_fastrak_idx[dev_id] = fastrak_idx;
    }

private:
    static absl::StatusOr<PciAndDevMapping>
    DiscoverPciAndDevMapping();

    absl::StatusOr<uint8_t> GetFastrakIdxFromPci(absl::string_view);

    // All endpoints. Ordering here doesn't matter, unless we sort this list by FasTrak index
    std::vector<std::unique_ptr<DxsEndpoint> absl_nonnull> endpoints_;

    absl::flat_hash_map<std::string, DxsEndpoint * absl_nonnull> dxs_addr_to_endpoint_;
    absl::flat_hash_map<uint8_t, DxsEndpoint * absl_nonnull> fastrak_idx_to_endpoint_;
    PciAndDevMapping pci_and_dev_mapping_;

    const int num_net_ifs_found_;

    const std::string ctrl_dev_name_;
    const SocketAddress ctrl_dev_addr_;
    const DxsEndpointConfig config_;
};

inline std::ostream &
operator<<(std::ostream &os, const ConnectionTraceId &id) {
    std::ios::fmtflags f(os.flags());
    os << std::hex << static_cast<std::underlying_type<ConnectionTraceId>::type>(id);
    os.flags(f);
    return os;
}

} // namespace tcpxo

#endif // NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_DXS_ENDPOINT_H
