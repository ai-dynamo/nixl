// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

// The NIXL transport of the adaptive EP kernels, host side (see `nixl.hpp`)
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <unordered_set>

#include <cuda_runtime.h>  // before the nixl_ep_v2 headers: under g++ their `__device__` annotations are macros from here
#include <nixl.h>
#include <absl/log/absl_vlog_is_on.h>

#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/topology.cuh>

#include "nixl.hpp"
#include "nixl_device.cuh"
#include "nixl_log.h"
#include "../../utils/lazy_driver.hpp"
#include "../../utils/system.hpp"

namespace nixl::runtime::ep::elastic::nixl {

bool is_fabric_handle_supported(const int& device_id) {
    // The caller has initialised the CUDA runtime on this device (torch's current device)
    CUdevice device; int fabric = 0;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, device_id));
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    return fabric == 1;
}

#define NIXL_CHECK(cmd) do { \
    if (const auto nixl_status__ = (cmd); nixl_status__ != NIXL_SUCCESS) \
        throw EPException("NIXL", __FILE__, __LINE__, "status " + std::to_string(static_cast<int>(nixl_status__)) + " from `" #cmd "`"); \
} while (0)

struct NIXLAgentState {
    std::shared_ptr<nixlAgent> agent;
    std::string agent_name;
    nixlBackendH* backend = nullptr;
    nixl_opt_args_t extra_params;
    nixl_reg_dlist_t reg_descs{VRAM_SEG};

    // The local RDMA view is lifetime-fixed when needed; remote views follow staged membership.
    nixlMemViewH staged_remote = nullptr;
    nixlMemViewH active_local = nullptr, active_remote = nullptr;
    bool has_staged = false;
    bool registered = false;

    ~NIXLAgentState() noexcept;
};

static const std::string kHelloPrefix = "nixl_ep_v2_hello:";
static constexpr int kNumMaxUCXDeviceChannels = 64;  // UCX's worker address carries at most 64 device channels (`ucx_num_device_channels`)

// Kernel edges stay within a domain or join the same internal position across domains.
static bool needs_endpoint(const NIXLEPContext& ctx, const int& wr) {
    const int n = ctx.max_num_scaleup_ranks;
    const auto& slots = ctx.get_world_to_slot();
    return slots[wr] / n == slots[ctx.world_rank] / n or slots[wr] % n == slots[ctx.world_rank] % n;
}

// Cleanup must preserve an exception already unwinding, and still attempt the remaining releases
template <typename func_t>
static void cleanup(const char* what, const func_t& func) noexcept {
    try {
        func();
    } catch (const std::exception& e) {
        NIXL_ERROR << "NIXL EP cleanup (" << what << ") failed: " << e.what();
    } catch (...) {
        NIXL_ERROR << "NIXL EP cleanup (" << what << ") failed: unknown exception";
    }
}

NIXLEPContext::Region::~Region() noexcept {
    reset();
}

void NIXLEPContext::Region::reset() noexcept {
    if (is_vmm) {
        if (mapped)
            cleanup("unmap region", [&] { CUDA_DRIVER_CHECK(lazy_cuMemUnmap(ptr, num_bytes)); });
        if (handle != 0)
            cleanup("release region", [&] { CUDA_DRIVER_CHECK(lazy_cuMemRelease(handle)); });
        if (ptr != 0)
            cleanup("free region address", [&] { CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(ptr, num_bytes)); });
    } else if (ptr != 0) {
        cleanup("free region", [&] { CUDA_RUNTIME_CHECK(cudaFree(reinterpret_cast<void*>(ptr))); });
    }
    ptr = 0, num_bytes = 0, handle = 0, is_vmm = false, mapped = false;
}

void NIXLEPContext::CudaDeleter::operator()(void* ptr) const noexcept {
    if (ptr != nullptr)
        cleanup("free device table", [&] { CUDA_RUNTIME_CHECK(cudaFree(ptr)); });
}

template <typename dtype_t, typename deleter_t>
static void allocate_device(std::unique_ptr<dtype_t, deleter_t>& allocation, const size_t num_bytes) {
    void* ptr = nullptr;
    CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, num_bytes));
    allocation.reset(static_cast<dtype_t*>(ptr));
}

NIXLAgentState::~NIXLAgentState() noexcept {
    if (agent == nullptr)
        return;
    for (auto view: {staged_remote, active_local, active_remote})
        if (view != nullptr)
            cleanup("release memory view", [&] { agent->releaseMemView(view); });
    if (registered)
        cleanup("deregister region", [&] { NIXL_CHECK(agent->deregisterMem(reg_descs, &extra_params)); });
}

// Whether the agent holds a peer's metadata (`checkRemoteMD` with an empty list), and its invalidation (a no-op when absent)
static bool has_remote_md(nixlAgent& agent, const std::string& name) {
    nixl_xfer_dlist_t empty_descs(VRAM_SEG);
    return agent.checkRemoteMD(name, empty_descs) == NIXL_SUCCESS;
}

static nixl_status_t invalidate_remote_md(nixlAgent& agent, const std::string& name) {
    return has_remote_md(agent, name) ? agent.invalidateRemoteMD(name) : NIXL_SUCCESS;
}

// A refused peer request names both ends
[[noreturn]] static void refuse(const int& world_rank, const int& wr, const char* what) {
    throw EPException("NIXL", __FILE__, __LINE__, "world rank " + std::to_string(world_rank) + ", peer " + std::to_string(wr) + ": " + what);
}

NIXLEPContext::NIXLEPContext(const std::string& agent_name, const int& world_rank, const int& num_world_ranks,
                             const int& max_num_scaleup_ranks, const int64_t& num_region_bytes, const int& num_channels):
    world_rank(world_rank), num_world_ranks(num_world_ranks), max_num_scaleup_ranks(max_num_scaleup_ranks),
    num_region_bytes(num_region_bytes), peers(num_world_ranks), connected(num_world_ranks, 0),
    world_to_slot(num_world_ranks, -1), endpoint_connected(num_world_ranks, 0),
    known_agent_names(num_world_ranks), p2p_ptrs_host(num_world_ranks, nullptr) {
    world_to_slot[world_rank] = 0;
    if (num_channels < 1 or num_channels > kNumMaxUCXDeviceChannels)
        throw EPException("NIXL", __FILE__, __LINE__, "UCX connects at most " + std::to_string(kNumMaxUCXDeviceChannels) + " device channels per peer, got " + std::to_string(num_channels) + ": set NIXL_EP_NIXL_NUM_CHANNELS between 1 and " + std::to_string(kNumMaxUCXDeviceChannels));
    const bool direct_only = num_world_ranks == max_num_scaleup_ranks;
    if (not direct_only and not has_gpu_rdma_device_code())
        throw EPExceptionWithLineInfo("NIXL", "Multi-domain NIXL EP requires UCX GPU RDMA device support: "
            "rebuild NIXL EP with GDA-enabled UCX and RDMA development headers (including infiniband/mlx5dv.h)");
    CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
    CUdevice device;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, device_id));

    // A fabric-shareable, GPUDirect-RDMA-capable VMM allocation when the device supports it (peers on other hosts can map it), else cudaMalloc
    int fabric = 0, rdma_vmm = 0;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&rdma_vmm, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device));
    if (fabric and rdma_vmm) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        prop.allocFlags.gpuDirectRDMACapable = 1;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        size_t granularity = 0;
        CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        const size_t aligned = (static_cast<size_t>(num_region_bytes) + granularity - 1) / granularity * granularity;

        CUmemGenericAllocationHandle handle = 0;
        const auto status = lazy_cuMemCreate(&handle, aligned, &prop, 0);
        if (status == CUDA_SUCCESS) {
            region.handle = handle, region.num_bytes = aligned, region.is_vmm = true;
            CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&region.ptr, aligned, 0, 0, 0));
            CUDA_DRIVER_CHECK(lazy_cuMemMap(region.ptr, aligned, 0, handle, 0));
            region.mapped = true;
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device_id;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(region.ptr, aligned, &access, 1));
        } else {
            NIXL_WARN << "NIXL EP context (world rank " << world_rank << "): cuMemCreate(FABRIC) failed ("
                      << static_cast<int>(status) << "), falling back to cudaMalloc; "
                      << "NVLink peers on other hosts cannot map this region and are rejected at commit";
        }
    }
    if (region.ptr == 0) {
        void* ptr = nullptr;
        CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, num_region_bytes));
        region.ptr = reinterpret_cast<CUdeviceptr>(ptr), region.num_bytes = static_cast<size_t>(num_region_bytes);
    }

    // The agent and its UCX backend: a progress thread polled every 1 ms (it serves the peers' wire-up and hellos while this host is in Python; NIXL's default delay 0 spins a core), RW sync for our calls beside it, one worker
    state = std::make_unique<NIXLAgentState>();
    state->agent_name = agent_name;
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.pthrDelay = 1000;  // us; NIXL rounds it up to whole ms for poll()
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    state->agent = std::make_shared<nixlAgent>(agent_name, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t init_params;
    NIXL_CHECK(state->agent->getPluginParams("UCX", mems, init_params));
    init_params["ucx_num_device_channels"] = std::to_string(num_channels);
    init_params["ucx_error_handling_mode"] = "none";
    init_params["num_workers"] = "1";
    init_params["engine_config"] =
        "RC_GDA_NUM_PRE_ALLOC_EPS=" + std::to_string(std::max(1, num_world_ranks / max_num_scaleup_ranks - 1)) +
        ",UD_LINGER_TIMEOUT=1s";
    if (direct_only)
        init_params["engine_config"] += ",TLS=^rc_gda";  // Keep CPU transports for wire-up; omit unused GPU RDMA queues.
    NIXL_CHECK(state->agent->createBackend("UCX", init_params, state->backend));
    state->extra_params.backends.push_back(state->backend);

    state->reg_descs.addDesc(nixlBlobDesc(static_cast<uintptr_t>(region.ptr), static_cast<size_t>(num_region_bytes),
                                          static_cast<uint64_t>(device_id), ""));
    NIXL_CHECK(state->agent->registerMem(state->reg_descs, &state->extra_params));
    state->registered = true;

    if (not direct_only) {
        nixl_local_dlist_t local_descs(VRAM_SEG);
        local_descs.addDesc(nixlBasicDesc(static_cast<uintptr_t>(region.ptr), static_cast<size_t>(num_region_bytes),
                                          static_cast<uint64_t>(device_id)));
        NIXL_CHECK(state->agent->prepMemView(local_descs, state->active_local, &state->extra_params));
    }

    // Device tables (see `nixl.hpp`): the p2p table holds self, the rest null until a commit
    allocate_device(p2p_ptrs_device, num_world_ranks * sizeof(void*));
    CUDA_RUNTIME_CHECK(cudaMemset(p2p_ptrs_device.get(), 0, num_world_ranks * sizeof(void*)));
    void* const self_base = region_base();
    p2p_ptrs_host[world_rank] = self_base;
    CUDA_RUNTIME_CHECK(cudaMemcpy(p2p_ptrs_device.get() + world_rank, &self_base, sizeof(void*), cudaMemcpyHostToDevice));
    allocate_device(p2p_ptrs_probe_device, num_world_ranks * sizeof(void*));
    allocate_device(connected_device, num_world_ranks * sizeof(int));
    allocate_device(ctx_device, sizeof(NIXLDeviceContext));

    ctx_host = NIXLDeviceContext{.local_mvh = state->active_local, .p2p_ptrs = p2p_ptrs_device.get(), .local_base = region_base(),
                                 .world_rank = world_rank, .num_channels = direct_only ? 0 : num_channels, .num_world_ranks = num_world_ranks};
    CUDA_RUNTIME_CHECK(cudaMemcpy(ctx_device.get(), &ctx_host, sizeof(NIXLDeviceContext), cudaMemcpyHostToDevice));

    NIXL_DEBUG << "NIXL EP context created: agent " << agent_name << ", world rank " << world_rank << " of "
               << num_world_ranks << " (" << max_num_scaleup_ranks << " per domain), " << ctx_host.num_channels
               << " channels, region " << region_base() << " (" << num_region_bytes << " bytes, "
               << (region.is_vmm ? "VMM with a fabric handle" : "cudaMalloc") << ")";
}

NIXLEPContext::~NIXLEPContext() {
    try {
        finalize();
    } catch (const std::exception& e) {
        NIXL_ERROR << "NIXL EP context finalization failed: " << e.what();
    }
}

void NIXLEPContext::finalize() {
    if (state == nullptr)
        return;
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    if (state->agent != nullptr) {
        release_staged_views();
        release_active_views();
        for (int wr = 0; wr < num_world_ranks; ++ wr) if (endpoint_connected[wr])
            invalidate_remote_md(*state->agent, known_agent_names[wr]);
    }
    state.reset();
    ctx_device.reset();
    p2p_ptrs_device.reset();
    connected_device.reset();
    p2p_ptrs_probe_device.reset();
    region.reset();
}

std::string NIXLEPContext::get_local_agent_name() const {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    return state->agent_name;
}

std::string NIXLEPContext::get_local_metadata() const {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    nixl_blob_t metadata;
    NIXL_CHECK(state->agent->getLocalMD(metadata));
    return metadata;
}

NIXLPeerInfo NIXLEPContext::get_local_peer_info() const {
    return {static_cast<uint64_t>(region.ptr), device_id, world_rank, static_cast<uint64_t>(num_region_bytes)};
}

void NIXLEPContext::connect_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_agent_names,
                                  const std::vector<NIXLPeerInfo>& peer_infos, const std::vector<int>& next_world_to_slot,
                                  const std::vector<std::string>& remote_metadata, const std::string& connection_id,
                                  const std::vector<std::string>& remote_connection_ids) {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    require_preparation_usable();
    EP_HOST_ASSERT(world_ranks.size() == remote_agent_names.size() and world_ranks.size() == peer_infos.size() and
                   world_ranks.size() == remote_metadata.size() and world_ranks.size() == remote_connection_ids.size());
    if (world_ranks.empty())
        return;

    // Validate the mapping and every peer before loading metadata or changing the peer registry.
    EP_HOST_ASSERT(next_world_to_slot.size() == static_cast<size_t>(num_world_ranks) and
                   "Topology mapping must cover the declared maximum world");
    std::unordered_set<int> assigned_slots;
    for (int wr = 0; wr < num_world_ranks; ++ wr) {
        const int slot = next_world_to_slot[wr];
        if (slot < -1 or slot >= num_world_ranks or (slot >= 0 and not assigned_slots.insert(slot).second))
            refuse(world_rank, wr, "Topology mapping must assign distinct slots within the declared maximum world");
        if ((wr == world_rank or connected[wr]) and slot < 0)
            refuse(world_rank, wr, "Topology mapping omits this rank or a connected peer");
    }
    std::unordered_set<int> seen;
    for (size_t i = 0; i < world_ranks.size(); ++ i) {
        const int wr = world_ranks[i];
        if (wr < 0 or wr >= num_world_ranks or wr == world_rank)
            refuse(world_rank, wr, "Invalid peer world rank");
        if (connected[wr] or not seen.insert(wr).second)
            refuse(world_rank, wr, "Peer already connected");
        if (next_world_to_slot[wr] < 0)
            refuse(world_rank, wr, "Topology mapping omits a requested peer");
        if (peer_infos[i].world_rank != wr or peer_infos[i].region_base == 0)
            refuse(world_rank, wr, "Peer info does not match the world rank");
        if (peer_infos[i].num_region_bytes < static_cast<uint64_t>(num_region_bytes))
            refuse(world_rank, wr, "Peer region is too small");
        if (remote_agent_names[i].empty())
            refuse(world_rank, wr, "Peer agent name is empty");
    }
    auto previous_mapping = world_to_slot;
    auto previous_peers = peers;
    auto previous_connected = connected;
    auto previous_endpoints = endpoint_connected;
    auto previous_names = known_agent_names;
    std::vector<size_t> endpoint_indices;
    endpoint_indices.reserve(world_ranks.size());
    try {
        world_to_slot = next_world_to_slot;
        connect_validated_ranks(world_ranks, remote_agent_names, peer_infos, remote_metadata,
                                connection_id, remote_connection_ids, endpoint_indices);
    } catch (...) {
        // Partial metadata loads can own endpoints without a remote memory section; invalidate them directly.
        for (const size_t i: endpoint_indices) {
            bool cancelled = false;
            cleanup("cancel peer connection", [&] {
                const auto status = state->agent->invalidateRemoteMD(remote_agent_names[i]);
                if (status != NIXL_ERR_NOT_FOUND)
                    NIXL_CHECK(status);
                cancelled = true;
            });
            preparation_failed |= not cancelled;
        }
        world_to_slot.swap(previous_mapping);
        peers.swap(previous_peers);
        connected.swap(previous_connected);
        endpoint_connected.swap(previous_endpoints);
        known_agent_names.swap(previous_names);
        if (preparation_failed)
            NIXL_ERROR << "NIXL EP: connection cancellation failed; the committed world may keep serving, but recreate the buffer before changing membership";
        throw;
    }
}

void NIXLEPContext::require_preparation_usable() const {
    EP_HOST_ASSERT(not preparation_failed and
                   "Connection cancellation failed: recreate the buffer before changing membership; the committed world may keep serving");
}

void NIXLEPContext::connect_validated_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_agent_names,
                                            const std::vector<NIXLPeerInfo>& peer_infos, const std::vector<std::string>& remote_metadata,
                                            const std::string& connection_id, const std::vector<std::string>& remote_connection_ids,
                                            std::vector<size_t>& endpoint_indices) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(get_env<int>("NIXL_EP_NIXL_CONNECT_TIMEOUT_SECS", 120));
    // Record every logical peer, including peers reached through an intermediate rank.
    for (size_t i = 0; i < world_ranks.size(); ++ i) {
        const int wr = world_ranks[i];
        const auto& name = remote_agent_names[i];
        known_agent_names[wr] = name;
        peers[wr] = peer_infos[i];
        connected[wr] = 1;
        // Connected coordinates stay fixed, so only new peers can need new endpoints.
        if (not needs_endpoint(*this, wr))
            continue;
        endpoint_indices.push_back(i);
        std::string loaded_name;
        NIXL_CHECK(state->agent->loadRemoteMD(remote_metadata[i], loaded_name));
        if (loaded_name != name)
            refuse(world_rank, wr, "Peer metadata does not match the exchanged agent name");
    }

    // UCX endpoints connect lazily. Echo the peer's fresh Store identifier so stale hellos cannot satisfy a retry.
    for (const size_t i: endpoint_indices) {
        NIXL_CHECK(state->agent->genNotif(remote_agent_names[i], kHelloPrefix + remote_connection_ids[i], &state->extra_params));
    }
    const auto expected_hello = kHelloPrefix + connection_id;
    std::vector<bool> greeted(endpoint_indices.size(), false);
    size_t num_greeted = 0;
    while (num_greeted < endpoint_indices.size()) {
        nixl_notifs_t notif_map;
        NIXL_CHECK(state->agent->getNotifs(notif_map, &state->extra_params));
        for (size_t i = 0; i < endpoint_indices.size(); ++ i) if (not greeted[i]) {
            const auto it = notif_map.find(remote_agent_names[endpoint_indices[i]]);
            if (it == notif_map.end())
                continue;
            const auto& messages = it->second;
            if (std::find(messages.begin(), messages.end(), expected_hello) != messages.end())
                greeted[i] = true, ++ num_greeted;
        }
        if (num_greeted < endpoint_indices.size()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                std::stringstream ss;
                for (size_t i = 0; i < endpoint_indices.size(); ++ i) if (not greeted[i])
                    ss << " " << world_ranks[endpoint_indices[i]];
                throw EPException("NIXL", __FILE__, __LINE__, "Timed out waiting for a peer's wire-up notification (the peer must call "
                                  "`connect_ranks` with this rank too, in the same connection round); world rank " + std::to_string(world_rank) + " is missing hellos from" + ss.str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Retain existing endpoints while old graphs still use the active memory view.
    for (const size_t i: endpoint_indices)
        endpoint_connected[world_ranks[i]] = 1;
    stage_views();
}

void NIXLEPContext::disconnect_ranks(const std::vector<int>& world_ranks, const topology::EPTopology& next, cudaStream_t stream) {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    require_preparation_usable();
    if (world_ranks.empty())
        return;
    // Input errors are recoverable: validate the whole request before changing any peer or staged view
    std::unordered_set<int> seen;
    for (const int wr: world_ranks) {
        if (wr < 0 or wr >= num_world_ranks or wr == world_rank)
            refuse(world_rank, wr, "Invalid peer world rank");
        if (not connected[wr])
            refuse(world_rank, wr, "Peer is not connected");
        if (not seen.insert(wr).second)
            refuse(world_rank, wr, "Duplicate peer world rank");
    }
    mutate_transport_or_abort(world_rank, "peer retirement", [&] {
        for (const int wr: world_ranks) {
            connected[wr] = 0;
            peers[wr] = NIXLPeerInfo{};
        }
        stage_views();
        commit(next, stream);

        // No view references the departed peers' metadata after commit.
        for (const int wr: world_ranks) if (endpoint_connected[wr]) {
            const auto status = invalidate_remote_md(*state->agent, known_agent_names[wr]);
            EP_HOST_ASSERT((status == NIXL_SUCCESS or status == NIXL_ERR_NOT_FOUND) and "invalidateRemoteMD failed");
            endpoint_connected[wr] = 0;
        }
    });
}

bool NIXLEPContext::is_connected(const int& wr) const {
    if (wr < 0 or wr >= num_world_ranks)
        refuse(world_rank, wr, "Peer world rank outside the declared maximum world (pass one below max_num_scaleout_ranks * max_num_scaleup_ranks)");
    return wr == world_rank or connected[wr] != 0;
}

std::vector<int> NIXLEPContext::get_connected_ranks() const {
    std::vector<int> out;
    for (int i = 0; i < num_world_ranks; ++ i) if (connected[i])
        out.push_back(i);
    return out;
}

void NIXLEPContext::stage_views() {
    // Preserve world-rank indices with null-agent holes for peers without endpoints; an all-null view is invalid.
    nixl_remote_dlist_t remote_descs(VRAM_SEG);
    bool any_endpoint = false;
    for (int r = 0; r < num_world_ranks; ++ r) {
        const bool has_endpoint = connected[r] and endpoint_connected[r];
        any_endpoint |= has_endpoint;
        remote_descs.addDesc(has_endpoint ?
            nixlRemoteDesc(static_cast<uintptr_t>(peers[r].region_base), static_cast<size_t>(num_region_bytes), static_cast<uint64_t>(peers[r].device_id), known_agent_names[r]) :
            nixlRemoteDesc(0, static_cast<size_t>(num_region_bytes), 0, nixl_null_agent));
    }
    nixlMemViewH next_remote = nullptr;
    if (any_endpoint)
        NIXL_CHECK(state->agent->prepMemView(remote_descs, next_remote, &state->extra_params));
    release_staged_views();
    state->staged_remote = next_remote;
    state->has_staged = true;
}

static void release_view(nixlAgent& agent, nixlMemViewH& view) {
    if (view != nullptr)
        agent.releaseMemView(view), view = nullptr;
}

void NIXLEPContext::release_staged_views() {
    release_view(*state->agent, state->staged_remote);
    state->has_staged = false;
}

void NIXLEPContext::release_active_views() {
    release_view(*state->agent, state->active_remote);
}

void NIXLEPContext::activate_staged_views() {
    release_active_views();
    std::swap(state->active_remote, state->staged_remote);
    state->has_staged = false;
    ctx_host.remote_mvh = state->active_remote;
}

void NIXLEPContext::validate_peer_pointers(const topology::EPTopology& next, const std::vector<void*>& pointers) const {
    if (not next.is_active())
        return;
    for (int wr = 0; wr < num_world_ranks; ++ wr) {
        if (not connected[wr] or not endpoint_connected[wr])
            continue;
        const int slot = world_to_slot[wr];
        if (not next.is_active(slot / max_num_scaleup_ranks, slot % max_num_scaleup_ranks))
            continue;
        const bool same_domain = slot / max_num_scaleup_ranks == next.scaleout_rank_idx;
        const bool has_nvlink_ptr = pointers[wr] != nullptr;
        if (same_domain != has_nvlink_ptr) {
            std::stringstream ss;
            if (same_domain) ss << "nixlGetPtr returned no NVLink pointer from rank " << world_rank << " to same-domain peer " << wr
               << ": UCX must provide a CUDA IPC device lane for it (keep cuda_ipc in UCX_TLS or leave UCX_TLS "
                  "unset; for an NVLink domain spanning several hosts, UCX_CUDA_IPC_ENABLE_MNNVL=yes with the IMEX "
                  "service running). The previous views stay active";
            else ss << "nixlGetPtr returned a direct pointer from rank " << world_rank << " to different-domain peer " << wr
                    << ": the scale-out path requires RDMA between discovered NVLink domains; check physical discovery and UCX CUDA IPC/fabric configuration. The previous views stay active";
            throw EPException("NIXL", __FILE__, __LINE__, ss.str());
        }
    }
}

void NIXLEPContext::commit(const topology::EPTopology& next, cudaStream_t stream) {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    require_preparation_usable();
    if (not state->has_staged) {
        validate_peer_pointers(next, p2p_ptrs_host);
        return;
    }

    // No kernel may hold the old views or the old p2p table while they change
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    // Probe the staged remote view into the probe table before anything is swapped (the context travels by value)
    auto probe = ctx_host;
    probe.remote_mvh = state->staged_remote;
    probe.p2p_ptrs = p2p_ptrs_probe_device.get();
    std::vector<void*> probed(num_world_ranks, nullptr);
    std::vector<int> has_endpoint(num_world_ranks);
    for (int wr = 0; wr < num_world_ranks; ++ wr)
        has_endpoint[wr] = connected[wr] and endpoint_connected[wr];
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(connected_device.get(), has_endpoint.data(), num_world_ranks * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_RUNTIME_CHECK(cache_peer_pointers(probe, connected_device.get(), stream));
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(probed.data(), p2p_ptrs_probe_device.get(), num_world_ranks * sizeof(void*), cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

    validate_peer_pointers(next, probed);

    // Activation releases the old views: later failures cannot leave a graph free to replay on them.
    mutate_transport_or_abort(world_rank, "memory-view commit", [&] {
        activate_staged_views();

        // The device context (the new views) and the p2p table, both at their fixed addresses
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(ctx_device.get(), &ctx_host, sizeof(NIXLDeviceContext), cudaMemcpyHostToDevice, stream));
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(p2p_ptrs_device.get(), p2p_ptrs_probe_device.get(), num_world_ranks * sizeof(void*), cudaMemcpyDeviceToDevice, stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

    });
    p2p_ptrs_host.swap(probed);

    if (ABSL_VLOG_IS_ON(1)) {
        std::stringstream ss;
        for (int i = 0; i < num_world_ranks; ++ i) if (i == world_rank or connected[i])
            ss << i << (i == world_rank ? ":self" : not has_endpoint[i] ? ":no-endpoint" : p2p_ptrs_host[i] != nullptr ? ":p2p" : ":rdma") << " ";
        NIXL_DEBUG << "NIXL EP context (world rank " << world_rank << ") committed views (nixlGetPtr), peers: " << ss.str();
    }
}

const NIXLDeviceContext* NIXLEPContext::get_device_context_ptr() const {
    return ctx_device.get();
}

}  // namespace nixl::runtime::ep::elastic::nixl
