// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nixl_ep_v2/common/nixl_transport.cuh>

#include "nixl_log.h"

namespace nixl::runtime::ep::elastic::topology {
struct EPTopology;
}

// The NIXL transport of the adaptive EP kernels, host side: one symmetric region per rank (workspace followed by the
// token buffer) registered with a NIXL agent over UCX; peers are addressed by their world rank in the declared
// maximum world. `connect_ranks` / `disconnect_ranks` stage memory views; `commit` validates the requested active
// endpoints' NVLink pointers and publishes staged views and the p2p table.
namespace nixl::runtime::ep::elastic::nixl {

// A failed transport mutation must not leave a captured graph free to replay on inconsistent state.
template <typename func_t>
static void mutate_transport_or_abort(const int& world_rank, const char* operation, const func_t& func) {
    try {
        func();
    } catch (const std::exception& e) {
        NIXL_ERROR << "NIXL EP world rank " << world_rank << ": " << operation
                   << " failed; recovery is unsupported; restart the process: " << e.what();
        std::abort();
    } catch (...) {
        NIXL_ERROR << "NIXL EP world rank " << world_rank << ": " << operation
                   << " failed; recovery is unsupported; restart the process";
        std::abort();
    }
}

// What a peer must know about us besides the agent metadata (the field names say it)
struct NIXLPeerInfo {
    uint64_t region_base = 0;
    int device_id = 0;
    int world_rank = -1;
    uint64_t num_region_bytes = 0;  // The exporter's registered length
};

struct NIXLAgentState;  // Holds the NIXL objects (defined in nixl.cpp so that only that file needs `nixl.h`)

class NIXLEPContext {
public:
    // Identity, fixed for the life of the buffer
    int world_rank;
    int num_world_ranks;
    int max_num_scaleup_ranks;          // Stride of the internal domain/position slots, independent of world-rank numbering
    int device_id;

    // The region: `[workspace][buffer]`, `num_region_bytes` long (the allocation may be a little larger)
    void* region_base() const { return reinterpret_cast<void*>(static_cast<uintptr_t>(region.ptr)); }
    int64_t num_region_bytes = 0;

    NIXLEPContext(const std::string& agent_name, const int& world_rank, const int& num_world_ranks,
                  const int& max_num_scaleup_ranks, const int64_t& num_region_bytes, const int& num_channels);
    ~NIXLEPContext();

    // Release the views, deregister and free the region (idempotent)
    void finalize();

    std::string get_local_agent_name() const;
    std::string get_local_metadata() const;
    NIXLPeerInfo get_local_peer_info() const;

    // Register peers, wire required endpoints and stage new views; world ranks distinct, not ours, not connected; every peer calls this with us too.
    // Failed preparation restores host state and leaves the active views untouched; the caller coordinates cancellation across peers.
    void connect_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_agent_names,
                       const std::vector<NIXLPeerInfo>& peer_infos, const std::vector<int>& world_to_slot,
                       const std::vector<std::string>& remote_metadata, const std::string& connection_id,
                       const std::vector<std::string>& remote_connection_ids);

    // Prepared host mapping; only Buffer's drained activation publishes its device copy.
    const std::vector<int>& get_world_to_slot() const { return world_to_slot; }

    // Retire peers, commit replacement views, then release their endpoints before returning.
    // Invalid requests leave state unchanged; failures after retirement begins abort.
    void disconnect_ranks(const std::vector<int>& world_ranks, const topology::EPTopology& next, cudaStream_t stream);

    bool is_connected(const int& wr) const;
    std::vector<int> get_connected_ranks() const;

    // Validate active endpoints, then publish staged views; call after draining, outside capture.
    // A pointer/domain mismatch leaves the active views, device context and p2p table untouched.
    void commit(const topology::EPTopology& next, cudaStream_t stream);

    // The fixed device address of the context the kernels read
    const NIXLDeviceContext* get_device_context_ptr() const;

private:
    // Region allocation (VMM with fabric handle, else cudaMalloc)
    struct Region {
        CUdeviceptr ptr = 0;
        size_t num_bytes = 0;
        CUmemGenericAllocationHandle handle = 0;
        bool is_vmm = false, mapped = false;

        Region() = default;
        Region(const Region&) = delete;
        Region& operator=(const Region&) = delete;
        ~Region() noexcept;
        void reset() noexcept;
    } region;

    struct CudaDeleter {
        void operator()(void* ptr) const noexcept;
    };

    // Logical peer registry: endpoints are needed only within our domain or at our position in another domain.
    std::vector<NIXLPeerInfo> peers;           // `[num_world_ranks]`, valid iff `connected`
    std::vector<int> connected;                // `[num_world_ranks]`, 0/1
    std::vector<int> world_to_slot;            // Prepared internal slot per world rank; -1 for unassigned ranks
    std::vector<int> endpoint_connected;       // Wired endpoints retained until peer retirement
    std::vector<std::string> known_agent_names; // `[num_world_ranks]`: every logical peer's agent

    // Mirror of the committed pointer table: membership-only changes need no GPU probe.
    std::vector<void*> p2p_ptrs_host;

    // Device tables
    NIXLDeviceContext ctx_host{};
    std::unique_ptr<NIXLDeviceContext, CudaDeleter> ctx_device;
    std::unique_ptr<void*, CudaDeleter> p2p_ptrs_device;        // The table the kernels read, at a fixed address
    std::unique_ptr<void*, CudaDeleter> p2p_ptrs_probe_device;  // The table a staged view yields, checked before it replaces the above
    std::unique_ptr<int, CudaDeleter> connected_device;         // Endpoint mask: never query nixlGetPtr on a null-agent entry

    // Destroy views and deregister memory before freeing any allocation, including when construction throws
    std::unique_ptr<NIXLAgentState> state;

    void stage_views();
    void connect_validated_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_agent_names,
                                 const std::vector<NIXLPeerInfo>& peer_infos, const std::vector<std::string>& remote_metadata,
                                 const std::string& connection_id, const std::vector<std::string>& remote_connection_ids,
                                 std::vector<size_t>& endpoint_indices);
    void require_preparation_usable() const;
    void validate_peer_pointers(const topology::EPTopology& next, const std::vector<void*>& pointers) const;
    void release_staged_views();
    void release_active_views();
    void activate_staged_views();
    bool preparation_failed = false;
};

// Whether the device exports fabric-shareable memory handles (`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`); the Python
// fabric evidence asks here instead of binding the driver itself
bool is_fabric_handle_supported(const int& device_id);

}  // namespace nixl::runtime::ep::elastic::nixl
