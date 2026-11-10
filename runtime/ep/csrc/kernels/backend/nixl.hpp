// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nixl_ep_v2/common/nixl_transport.cuh>

// The NIXL transport of the adaptive EP kernels, host side: one symmetric region per rank (workspace followed by the
// token buffer) registered with a NIXL agent over UCX; peers are addressed by their world rank in the declared
// maximum world. `connect_ranks` / `disconnect_ranks` stage memory views, `commit` probes every endpoint peer's NVLink
// pointer (`nixlGetPtr`), refuses when a same-domain peer has none or a rail peer has one, else swaps the views in and uploads the device context and the p2p table
namespace nixl_ep_v2::elastic::nixl {

// A failed transport mutation must not leave a captured graph free to replay on inconsistent state.
template <typename func_t>
static void mutate_transport_or_abort(const int& world_rank, const char* operation, const func_t& func) {
    try {
        func();
    } catch (const std::exception& e) {
        fprintf(stderr, "NIXL EP v2 world rank %d: %s failed; recovery is unsupported; restart the process: %s\n",
                world_rank, operation, e.what());
        fflush(stderr);
        std::abort();
    } catch (...) {
        fprintf(stderr, "NIXL EP v2 world rank %d: %s failed; recovery is unsupported; restart the process\n", world_rank, operation);
        fflush(stderr);
        std::abort();
    }
}

// What a peer must know about us besides the agent metadata (the field names say it)
struct NIXLPeerInfo {
    uint64_t region_base = 0;
    int device_id = 0;
    int world_rank = -1;
    uint64_t num_region_bytes = 0;  // The allocation size (the exporter's, granularity-aligned)
};

struct NIXLAgentState;  // Holds the NIXL objects (defined in nixl.cpp so that only that file needs `nixl.h`)

class NIXLEPContext {
public:
    // Identity, fixed for the life of the buffer
    int world_rank;
    int num_world_ranks;
    int max_num_scaleup_ranks;          // The domain size n: world ranks `[d * n, (d + 1) * n)` form NVLink domain d
    int device_id;

    // The region: `[workspace][buffer]`, `num_region_bytes` long (the allocation may be a little larger)
    void* region_base() const { return reinterpret_cast<void*>(static_cast<uintptr_t>(region.ptr)); }
    int64_t num_region_bytes = 0;

    NIXLEPContext(const std::string& agent_name, const int& world_rank, const int& num_world_ranks,
                  const int& max_num_scaleup_ranks, const int64_t& num_region_bytes, const int& num_channels);
    ~NIXLEPContext();

    // Release the views, deregister and free the region (idempotent)
    void finalize();

    // This agent's metadata blob and region info, for the peers (exchanged out of band by the caller)
    std::string get_local_metadata() const;
    NIXLPeerInfo get_local_peer_info() const;

    // Register peers, wire required endpoints and stage new views; world ranks distinct, not ours, not connected; every peer calls this with us too.
    // Validate the whole peer list before loading metadata; later failures abort because peer state may have changed.
    void connect_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_mds,
                       const std::vector<NIXLPeerInfo>& peer_infos);

    // Forget peers and stage new views; their metadata is invalidated at the next commit. Invalid requests leave state
    // unchanged; a memory-view preparation failure aborts the process (recovery is unsupported)
    void disconnect_ranks(const std::vector<int>& world_ranks);

    bool is_connected(const int& wr) const;
    std::vector<int> get_connected_ranks() const;
    bool has_staged_views() const;

    // Make the staged views live (device sync, probe, refuse or swap, upload); never under capture or while a transport kernel may run
    // A connected same-domain peer without an NVLink pointer, or a rail peer with one: throw, with the active views, the device context and the p2p table untouched
    void commit(cudaStream_t stream);

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
    std::vector<int> connect_epoch;            // `[num_world_ranks]`: endpoint connection epoch (hello tag)
    std::vector<std::string> known_agent_names; // `[num_world_ranks]`: endpoint peer's last agent; retained across disconnects to restart epochs for a new process
    std::vector<int> pending_disconnects;      // Disconnected since the last commit, which invalidates their agents' metadata (a reconnect must wait for it)

    // Device tables
    NIXLDeviceContext ctx_host{};
    std::unique_ptr<NIXLDeviceContext, CudaDeleter> ctx_device;
    std::unique_ptr<void*, CudaDeleter> p2p_ptrs_device;        // The table the kernels read, at a fixed address
    std::unique_ptr<void*, CudaDeleter> p2p_ptrs_probe_device;  // The table a staged view yields, checked before it replaces the above
    std::unique_ptr<int, CudaDeleter> connected_device;         // Endpoint mask: never query nixlGetPtr on a null-agent entry

    // Destroy views and deregister memory before freeing any allocation, including when construction throws
    std::unique_ptr<NIXLAgentState> state;

    void stage_views();
    void connect_validated_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_mds,
                                 const std::vector<NIXLPeerInfo>& peer_infos);
    void release_staged_views();
    void release_active_views();
    void activate_staged_views();
    bool finalized = false;
};

// Whether the device exports fabric-shareable memory handles (`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`); the Python
// fabric evidence asks here instead of binding the driver itself
bool is_fabric_handle_supported(const int& device_id);

}  // namespace nixl_ep_v2::elastic::nixl
