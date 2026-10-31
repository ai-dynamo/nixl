/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nixl_ep_v2/common/nixl_transport.cuh>

// The NIXL transport of the adaptive EP kernels, host side: one symmetric region per rank (workspace followed by the
// token buffer) registered with a NIXL agent over UCX; peers are addressed by their dense rank id in the declared
// maximum world. `connect_ranks` / `disconnect_ranks` stage memory views, `commit` probes every connected peer's NVLink
// pointer (`nixlGetPtr`), refuses when a same-domain peer has none or a rail peer has one, else swaps the views in and uploads the device context and the p2p table
namespace nixl_ep_v2::elastic::nixl {

// What a peer must know about us besides the agent metadata (the field names say it)
struct NIXLPeerInfo {
    uint64_t region_base = 0;
    int device_id = 0;
    int rank_id = -1;
    uint64_t num_region_bytes = 0;  // The allocation size (the exporter's, granularity-aligned)
};

struct NIXLAgentState;  // Holds the NIXL objects (defined in nixl.cpp so that only that file needs `nixl.h`)

class NIXLEPContext {
public:
    // Identity, fixed for the life of the buffer
    int rank_id;
    int num_rank_ids;
    int num_scaleup_ranks_per_domain;   // `max_num_scaleup_ranks`: rank ids `[d * n, (d + 1) * n)` form NVLink domain d
    int device_id;
    int num_channels;

    // The region: `[workspace][buffer]`, `num_region_bytes` long (the allocation may be a little larger)
    void* region_base = nullptr;
    int64_t num_region_bytes = 0;

    NIXLEPContext(const std::string& agent_name, const int& rank_id, const int& num_rank_ids,
                  const int& num_scaleup_ranks_per_domain, const int64_t& num_region_bytes, const int& num_channels);
    ~NIXLEPContext();

    // Release the views, deregister and free the region (idempotent)
    void finalize();

    // This agent's metadata blob and region info, for the peers (exchanged out of band by the caller)
    std::string get_local_metadata() const;
    NIXLPeerInfo get_local_peer_info() const;

    // Load the peers' metadata, wire the endpoints up and stage new views; ids distinct, not ours, not connected; every peer calls this with us too.
    // A throw is fatal to the process, not retried: a peer may already have consumed our hello and be connected at this epoch, and a retry would move our epoch past it
    void connect_ranks(const std::vector<int>& rank_ids, const std::vector<std::string>& remote_mds,
                       const std::vector<NIXLPeerInfo>& peer_infos);

    // Forget peers and stage new views; their metadata is invalidated at the next commit
    void disconnect_ranks(const std::vector<int>& rank_ids);

    bool is_connected(const int& rank_id) const;
    std::vector<int> get_connected_ranks() const;
    bool has_staged_views() const;

    // Make the staged views live (device sync, probe, refuse or swap, upload); never under capture or while a transport kernel may run
    // A connected same-domain peer without an NVLink pointer, or a rail peer with one: throw, with the active views, the device context and the p2p table untouched
    void commit(cudaStream_t stream);

    // The fixed device address of the context the kernels read
    const NIXLDeviceContext* get_device_context_ptr() const;

private:
    std::unique_ptr<NIXLAgentState> state;

    // Region allocation (VMM with fabric handle, else cudaMalloc)
    struct Region {
        CUdeviceptr ptr = 0;
        size_t num_bytes = 0;
        CUmemGenericAllocationHandle handle = 0;
        bool is_vmm = false;
    } region;

    // Peer registry
    std::vector<NIXLPeerInfo> peers;           // `[num_rank_ids]`, valid iff `connected`
    std::vector<int> connected;                // `[num_rank_ids]`, 0/1
    std::vector<std::string> remote_agent_names;
    std::vector<int> connect_epoch;            // `[num_rank_ids]`: how many times we connected to the peer (hello tag)
    std::vector<std::string> known_agent_names; // `[num_rank_ids]`: the agent last seen behind the id, kept across disconnects (a different agent = a new process behind the id: the epoch restarts)
    std::vector<int> pending_disconnects;      // Disconnected since the last commit (a reconnect must wait for it)
    std::vector<std::string> pending_invalidations;

    // Device tables
    NIXLDeviceContext ctx_host{};
    NIXLDeviceContext* ctx_device = nullptr;
    void** p2p_ptrs_device = nullptr;         // The table the kernels read, at a fixed address
    void** p2p_ptrs_probe_device = nullptr;   // The table a staged view yields, checked before it replaces the above
    int* connected_device = nullptr;          // Presence mask: never query nixlGetPtr on a null-agent entry

    void stage_views();
    void release_staged_views();
    void release_active_views();
    void activate_staged_views();
    bool finalized = false;
};

// Whether the device exports fabric-shareable memory handles (`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`); the Python
// topology evidence asks here instead of binding the driver itself
bool is_fabric_handle_supported(const int& device_id);

}  // namespace nixl_ep_v2::elastic::nixl
