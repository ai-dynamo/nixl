// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

// The NIXL transport of the adaptive EP kernels, host side (see `nixl.hpp`)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>  // before the nixl_ep_v2 headers: under g++ their `__device__` annotations are macros from here
#include <nixl.h>

#include <nixl_ep_v2/common/exception.cuh>

#include "nixl.hpp"
#include "nixl_device.cuh"
#include "../../utils/lazy_driver.hpp"
#include "../../utils/system.hpp"

namespace nixl_ep_v2::elastic::nixl {

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
    nixlBackendH* backend = nullptr;
    nixl_opt_args_t extra_params;
    nixl_reg_dlist_t reg_descs{VRAM_SEG};

    // The local view is lifetime-fixed; remote views follow staged membership.
    nixlMemViewH staged_remote = nullptr;
    nixlMemViewH active_local = nullptr, active_remote = nullptr;
    bool has_staged = false;
    bool registered = false;
    std::unordered_map<std::string, std::vector<int>> pending_hellos;

    ~NIXLAgentState() noexcept;
};

static const std::string kHelloPrefix = "nixl_ep_v2_hello:";
static constexpr int kNumMaxUCXDeviceChannels = 64;  // UCX's worker address carries at most 64 device channels (`ucx_num_device_channels`)

// Cleanup must preserve an exception already unwinding, and still attempt the remaining releases
template <typename func_t>
static void cleanup(const char* what, const func_t& func) noexcept {
    try {
        func();
    } catch (const std::exception& e) {
        fprintf(stderr, "NIXL EP v2 cleanup (%s) failed: %s\n", what, e.what());
    } catch (...) {
        fprintf(stderr, "NIXL EP v2 cleanup (%s) failed: unknown exception\n", what);
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
    connect_epoch(num_world_ranks, 0), known_agent_names(num_world_ranks) {
    if (num_channels < 1 or num_channels > kNumMaxUCXDeviceChannels)
        throw EPException("NIXL", __FILE__, __LINE__, "UCX connects at most " + std::to_string(kNumMaxUCXDeviceChannels) + " device channels per peer, got " + std::to_string(num_channels) + ": pass `num_nixl_channels` (or NIXL_EP_NIXL_NUM_CHANNELS) between 1 and " + std::to_string(kNumMaxUCXDeviceChannels));
    CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
    CUdevice device;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, device_id));

    // A fabric-shareable, GPUDirect-RDMA-capable VMM allocation when the device supports it (peers on other hosts can map it), else cudaMalloc
    int fabric = 0, rdma_vmm = 0;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&rdma_vmm, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device));
    const bool try_vmm = fabric and rdma_vmm and get_env<int>("NIXL_EP_NIXL_DISABLE_VMM", 0) == 0;
    if (try_vmm) {
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
            printf("NIXL EP context (world rank %d): cuMemCreate(FABRIC) failed (%d), falling back to cudaMalloc; "
                   "NVLink peers on other hosts cannot map this region and are rejected at commit\n",
                   world_rank, static_cast<int>(status));
        }
    }
    if (region.ptr == 0) {
        void* ptr = nullptr;
        CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, num_region_bytes));
        region.ptr = reinterpret_cast<CUdeviceptr>(ptr), region.num_bytes = static_cast<size_t>(num_region_bytes);
    }

    // The agent and its UCX backend: a progress thread polled every 1 ms (it serves the peers' wire-up and hellos while this host is in Python; NIXL's default delay 0 spins a core), RW sync for our calls beside it, one worker, `num_channels` device channels
    state = std::make_unique<NIXLAgentState>();
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
    NIXL_CHECK(state->agent->createBackend("UCX", init_params, state->backend));
    state->extra_params.backends.push_back(state->backend);

    state->reg_descs.addDesc(nixlBlobDesc(static_cast<uintptr_t>(region.ptr), static_cast<size_t>(num_region_bytes),
                                          static_cast<uint64_t>(device_id), ""));
    NIXL_CHECK(state->agent->registerMem(state->reg_descs, &state->extra_params));
    state->registered = true;

    nixl_local_dlist_t local_descs(VRAM_SEG);
    local_descs.addDesc(nixlBasicDesc(static_cast<uintptr_t>(region.ptr), static_cast<size_t>(num_region_bytes),
                                      static_cast<uint64_t>(device_id)));
    NIXL_CHECK(state->agent->prepMemView(local_descs, state->active_local, &state->extra_params));

    // Device tables (see `nixl.hpp`): the p2p table holds self, the rest null until a commit
    allocate_device(p2p_ptrs_device, num_world_ranks * sizeof(void*));
    CUDA_RUNTIME_CHECK(cudaMemset(p2p_ptrs_device.get(), 0, num_world_ranks * sizeof(void*)));
    void* const self_base = region_base();
    CUDA_RUNTIME_CHECK(cudaMemcpy(p2p_ptrs_device.get() + world_rank, &self_base, sizeof(void*), cudaMemcpyHostToDevice));
    allocate_device(p2p_ptrs_probe_device, num_world_ranks * sizeof(void*));
    allocate_device(connected_device, num_world_ranks * sizeof(int));
    allocate_device(ctx_device, sizeof(NIXLDeviceContext));

    ctx_host = NIXLDeviceContext{.local_mvh = state->active_local, .p2p_ptrs = p2p_ptrs_device.get(), .local_base = region_base(),
                                 .world_rank = world_rank, .num_channels = num_channels, .num_world_ranks = num_world_ranks};
    CUDA_RUNTIME_CHECK(cudaMemcpy(ctx_device.get(), &ctx_host, sizeof(NIXLDeviceContext), cudaMemcpyHostToDevice));

    if (get_env<int>("NIXL_EP_BUFFER_DEBUG"))
        printf("NIXL EP context created: agent %s, world rank %d of %d (%d per domain), %d channels, region %p (%ld bytes, %s)\n",
               agent_name.c_str(), world_rank, num_world_ranks, max_num_scaleup_ranks, num_channels, region_base(),
               static_cast<long>(num_region_bytes), region.is_vmm ? "VMM with a fabric handle" : "cudaMalloc");
}

NIXLEPContext::~NIXLEPContext() {
    if (not finalized) {
        try {
            finalize();
        } catch (const std::exception& e) {
            printf("NIXL EP context finalization failed: %s\n", e.what());
        }
    }
}

void NIXLEPContext::finalize() {
    if (finalized)
        return;
    finalized = true;
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    if (state != nullptr and state->agent != nullptr) {
        release_staged_views();
        release_active_views();
        for (const int wr: get_connected_ranks())
            invalidate_remote_md(*state->agent, known_agent_names[wr]);
    }
    state.reset();
    ctx_device.reset();
    p2p_ptrs_device.reset();
    connected_device.reset();
    p2p_ptrs_probe_device.reset();
    region.reset();
}

std::string NIXLEPContext::get_local_metadata() const {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    nixl_blob_t blob;
    NIXL_CHECK(state->agent->getLocalMD(blob));
    return blob;
}

NIXLPeerInfo NIXLEPContext::get_local_peer_info() const {
    return {static_cast<uint64_t>(region.ptr), device_id, world_rank, region.num_bytes};
}

void NIXLEPContext::connect_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_mds,
                                  const std::vector<NIXLPeerInfo>& peer_infos) {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    EP_HOST_ASSERT(world_ranks.size() == remote_mds.size() and world_ranks.size() == peer_infos.size());
    if (world_ranks.empty())
        return;

    // Validate every peer before loading metadata or changing the peer registry.
    std::unordered_set<int> seen;
    for (size_t i = 0; i < world_ranks.size(); ++ i) {
        const int wr = world_ranks[i];
        if (wr < 0 or wr >= num_world_ranks or wr == world_rank)
            refuse(world_rank, wr, "Invalid peer world rank");
        if (connected[wr] or not seen.insert(wr).second)
            refuse(world_rank, wr, "Peer already connected");
        if (peer_infos[i].world_rank != wr or peer_infos[i].region_base == 0)
            refuse(world_rank, wr, "Peer info does not match the world rank");
        if (peer_infos[i].num_region_bytes < static_cast<uint64_t>(num_region_bytes))
            refuse(world_rank, wr, "Peer region is too small");
        // A pending disconnect of the world rank would invalidate the metadata reloaded here at the next commit
        if (std::find(pending_disconnects.begin(), pending_disconnects.end(), wr) != pending_disconnects.end())
            refuse(world_rank, wr, "Commit the pending disconnect before reconnecting the same rank");
    }
    mutate_transport_or_abort(world_rank, "peer connection", [&] {
        connect_validated_ranks(world_ranks, remote_mds, peer_infos);
    });
}

void NIXLEPContext::connect_validated_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_mds,
                                            const std::vector<NIXLPeerInfo>& peer_infos) {
    // Load every peer's metadata first (the loads proceed in parallel inside the agent).
    for (size_t i = 0; i < world_ranks.size(); ++ i) {
        const int wr = world_ranks[i];
        std::string name;
        NIXL_CHECK(state->agent->loadRemoteMD(remote_mds[i], name));
        // A different agent behind the world rank is a new process (a leaver's exited, a fresh one joined): its epochs start from
        // zero, so ours restart (the per-peer barrier words restart at every connect, see `ElasticBuffer::connect_ranks`)
        if (known_agent_names[wr] != name) {
            connect_epoch[wr] = 0;
            known_agent_names[wr] = name;
        }
    }

    // Wire-up handshake: UCX endpoints connect lazily and `prepMemView` needs them connected, so every new peer gets a
    // hello carrying the pair's connection epoch and we wait for its hello back (both sides call `connect_ranks` with
    // each other); a hello of an earlier epoch cannot satisfy this one, an asymmetric protocol shows as an epoch mismatch
    const auto start = std::chrono::steady_clock::now();
    const auto timeout_secs = get_env<int>("NIXL_EP_NIXL_CONNECT_TIMEOUT_SECS", 120);
    for (size_t i = 0; i < world_ranks.size(); ++ i) {
        const int wr = world_ranks[i];
        connect_epoch[wr] += 1;
        NIXL_CHECK(state->agent->genNotif(known_agent_names[wr], kHelloPrefix + std::to_string(connect_epoch[wr]),
                                          &state->extra_params));
    }
    std::vector<bool> greeted(world_ranks.size(), false);
    size_t num_greeted = 0;
    while (num_greeted < world_ranks.size()) {
        nixl_notifs_t notif_map;
        NIXL_CHECK(state->agent->getNotifs(notif_map, &state->extra_params));
        for (const auto& [from, msgs]: notif_map) {
            for (const auto& msg: msgs) if (msg.rfind(kHelloPrefix, 0) == 0)
                state->pending_hellos[from].push_back(std::atoi(msg.c_str() + kHelloPrefix.size()));
        }
        for (size_t i = 0; i < world_ranks.size(); ++ i) if (not greeted[i]) {
            const int wr = world_ranks[i];
            auto it = state->pending_hellos.find(known_agent_names[wr]);
            if (it == state->pending_hellos.end())
                continue;
            auto& epochs = it->second;
            // Drop stale hellos (earlier epochs), consume the matching one
            epochs.erase(std::remove_if(epochs.begin(), epochs.end(), [&](const int& e) { return e < connect_epoch[wr]; }), epochs.end());
            const auto match = std::find(epochs.begin(), epochs.end(), connect_epoch[wr]);
            if (match != epochs.end()) {
                epochs.erase(match);
                greeted[i] = true, ++ num_greeted;
            }
        }
        if (num_greeted < world_ranks.size()) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_secs) {
                std::stringstream ss;
                for (size_t i = 0; i < world_ranks.size(); ++ i) if (not greeted[i]) {
                    const int wr = world_ranks[i];
                    ss << " peer " << wr << " (epoch " << connect_epoch[wr] << ", received:";
                    if (const auto it = state->pending_hellos.find(known_agent_names[wr]); it != state->pending_hellos.end())
                        for (const int e: it->second) ss << " " << e;
                    ss << ")";
                }
                throw EPException("NIXL", __FILE__, __LINE__, "Timed out waiting for a peer's wire-up notification (the peer must call "
                                  "`connect_ranks` with this rank too, in the same connection epoch); world rank " + std::to_string(world_rank) + " is missing hellos from" + ss.str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Record peers and let NIXL/UCX prepare their mappings with the staged memory views.
    for (size_t i = 0; i < world_ranks.size(); ++ i) {
        const int wr = world_ranks[i];
        peers[wr] = peer_infos[i];
        connected[wr] = 1;
    }
    stage_views();
}

void NIXLEPContext::disconnect_ranks(const std::vector<int>& world_ranks) {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
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
    pending_disconnects.reserve(pending_disconnects.size() + world_ranks.size());
    for (const int wr: world_ranks) {
        connected[wr] = 0;
        peers[wr] = NIXLPeerInfo{};
        pending_disconnects.push_back(wr);
        state->pending_hellos.erase(known_agent_names[wr]);
    }
    mutate_transport_or_abort(world_rank, "memory-view preparation", [&] { stage_views(); });
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
    release_staged_views();

    // One entry per world rank, a null-agent hole for an absent peer (an all-null view is not a valid request)
    nixl_remote_dlist_t remote_descs(VRAM_SEG);
    for (int r = 0; r < num_world_ranks; ++ r)
        remote_descs.addDesc(connected[r] ?
            nixlRemoteDesc(static_cast<uintptr_t>(peers[r].region_base), static_cast<size_t>(num_region_bytes), static_cast<uint64_t>(peers[r].device_id), known_agent_names[r]) :
            nixlRemoteDesc(0, static_cast<size_t>(num_region_bytes), 0, nixl_null_agent));
    if (std::find(connected.begin(), connected.end(), 1) != connected.end())
        NIXL_CHECK(state->agent->prepMemView(remote_descs, state->staged_remote, &state->extra_params));
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

bool NIXLEPContext::has_staged_views() const {
    return state != nullptr and state->has_staged;
}

void NIXLEPContext::commit(cudaStream_t stream) {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    if (not state->has_staged)
        return;

    // No kernel may hold the old views or the old p2p table while they change
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    // Probe the staged remote view into the probe table before anything is swapped (the context travels by value)
    auto probe = ctx_host;
    probe.remote_mvh = state->staged_remote;
    probe.p2p_ptrs = p2p_ptrs_probe_device.get();
    std::vector<void*> probed(num_world_ranks, nullptr);
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(connected_device.get(), connected.data(), num_world_ranks * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_RUNTIME_CHECK(cache_peer_pointers(probe, connected_device.get(), stream));
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(probed.data(), p2p_ptrs_probe_device.get(), num_world_ranks * sizeof(void*), cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

    // Require CUDA IPC pointers within each declared domain and RDMA lanes between domains.
    // Refuse unsupported mappings with the active views untouched (the staged views stay staged).
    const auto domain_of = [&](const int& wr) { return wr / max_num_scaleup_ranks; };
    for (int wr = 0; wr < num_world_ranks; ++ wr) {
        const bool same_domain = domain_of(wr) == domain_of(world_rank);
        const bool has_nvlink_ptr = probed[wr] != nullptr;
        if (connected[wr] and same_domain != has_nvlink_ptr) {
            std::stringstream ss;
            if (same_domain) ss << "nixlGetPtr returned no NVLink pointer from rank " << world_rank << " to same-domain peer " << wr
               << ": UCX must provide a CUDA IPC device lane for it (keep cuda_ipc in UCX_TLS or leave UCX_TLS "
                  "unset; for an NVLink domain spanning several hosts, UCX_CUDA_IPC_ENABLE_MNNVL=yes with the IMEX "
                  "service running). The previous views stay active";
            else ss << "nixlGetPtr returned an NVLink pointer from rank " << world_rank << " to rail peer " << wr << ": the declared "
                  "NVLink domains split one CUDA IPC clique, which the scale-out path (RDMA writes and completion atomics) does not "
                  "support; declare the whole clique as one domain, or set UCX_CUDA_IPC_ENABLE_MNNVL=n when no domain spans hosts. The previous views stay active";
            throw EPException("NIXL", __FILE__, __LINE__, ss.str());
        }
    }

    // Activation releases the old views: later failures cannot leave a graph free to replay on them.
    mutate_transport_or_abort(world_rank, "memory-view commit", [&] {
        activate_staged_views();

        // The device context (the new views) and the p2p table, both at their fixed addresses
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(ctx_device.get(), &ctx_host, sizeof(NIXLDeviceContext), cudaMemcpyHostToDevice, stream));
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(p2p_ptrs_device.get(), p2p_ptrs_probe_device.get(), num_world_ranks * sizeof(void*), cudaMemcpyDeviceToDevice, stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

        // Peers that left: no view references their metadata any more
        for (const int wr: pending_disconnects) {
            const auto status = invalidate_remote_md(*state->agent, known_agent_names[wr]);
            EP_HOST_ASSERT((status == NIXL_SUCCESS or status == NIXL_ERR_NOT_FOUND) and "invalidateRemoteMD failed");
        }
        pending_disconnects.clear();
    });

    if (get_env<int>("NIXL_EP_BUFFER_DEBUG")) {
        std::stringstream ss;
        for (int i = 0; i < num_world_ranks; ++ i) if (i == world_rank or connected[i])
            ss << i << (i == world_rank ? ":self" : probed[i] != nullptr ? ":p2p" : ":rdma") << " ";
        printf("NIXL EP context (world rank %d) committed views (nixlGetPtr), peers: %s\n", world_rank, ss.str().c_str());
    }
}

const NIXLDeviceContext* NIXLEPContext::get_device_context_ptr() const {
    return ctx_device.get();
}

}  // namespace nixl_ep_v2::elastic::nixl
