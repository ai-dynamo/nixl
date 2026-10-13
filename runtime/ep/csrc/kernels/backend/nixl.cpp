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
#include <nixl_ep_v2/common/topology.cuh>

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

    // Staged and active memory views; the wire-up hellos (connection epochs) received per remote agent and not yet consumed
    nixlMemViewH staged_local = nullptr, staged_remote = nullptr;
    nixlMemViewH active_local = nullptr, active_remote = nullptr;
    bool has_staged = false;
    std::unordered_map<std::string, std::vector<int>> pending_hellos;
};

static const std::string kHelloPrefix = "nixl_ep_v2_hello:";

NIXLEPContext::NIXLEPContext(const std::string& agent_name, const int& rank_id, const int& num_rank_ids,
                             const int& num_scaleup_ranks_per_domain, const int64_t& num_region_bytes, const int& num_channels):
    rank_id(rank_id), num_rank_ids(num_rank_ids), num_scaleup_ranks_per_domain(num_scaleup_ranks_per_domain),
    num_channels(num_channels), num_region_bytes(num_region_bytes), peers(num_rank_ids), connected(num_rank_ids, 0),
    remote_agent_names(num_rank_ids), connect_epoch(num_rank_ids, 0), known_agent_names(num_rank_ids) {
    EP_HOST_ASSERT(0 <= rank_id and rank_id < num_rank_ids and num_rank_ids <= topology::kNumMaxRankIds);
    EP_HOST_ASSERT(num_scaleup_ranks_per_domain >= 1 and num_rank_ids % num_scaleup_ranks_per_domain == 0);
    EP_HOST_ASSERT(num_channels >= 1);
    CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
    CUdevice device;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, device_id));

    // A fabric-shareable, GPUDirect-RDMA-capable VMM allocation when the device supports it (peers on other hosts can map it), else cudaMalloc
    int fabric = 0, rdma_vmm = 0;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&rdma_vmm, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device));
    const bool try_vmm = fabric and rdma_vmm and get_env<int>("EP_NIXL_DISABLE_VMM", 0) == 0;
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
            CUdeviceptr ptr = 0;
            CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&ptr, aligned, 0, 0, 0));
            CUDA_DRIVER_CHECK(lazy_cuMemMap(ptr, aligned, 0, handle, 0));
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device_id;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(ptr, aligned, &access, 1));
            region = Region{ptr, aligned, handle, true};
        } else {
            printf("NIXL EP context (rank id %d): cuMemCreate(FABRIC) failed (%d), falling back to cudaMalloc; "
                   "NVLink peers on other hosts cannot map this region and are rejected at commit\n",
                   rank_id, static_cast<int>(status));
        }
    }
    if (region.ptr == 0) {
        void* ptr = nullptr;
        CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, num_region_bytes));
        region = Region{reinterpret_cast<CUdeviceptr>(ptr), static_cast<size_t>(num_region_bytes), 0, false};
    }
    region_base = reinterpret_cast<void*>(static_cast<uintptr_t>(region.ptr));

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

    state->reg_descs.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(region_base), static_cast<size_t>(num_region_bytes),
                                          static_cast<uint64_t>(device_id), ""));
    NIXL_CHECK(state->agent->registerMem(state->reg_descs, &state->extra_params));

    // Device tables (see `nixl.hpp`): the p2p table holds self, the rest null until a commit
    CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&p2p_ptrs_device), num_rank_ids * sizeof(void*)));
    CUDA_RUNTIME_CHECK(cudaMemset(p2p_ptrs_device, 0, num_rank_ids * sizeof(void*)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(p2p_ptrs_device + rank_id, &region_base, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&p2p_ptrs_probe_device), num_rank_ids * sizeof(void*)));
    CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&connected_device), num_rank_ids * sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx_device), sizeof(NIXLDeviceContext)));

    ctx_host = NIXLDeviceContext{.p2p_ptrs = p2p_ptrs_device, .local_base = region_base,
                                 .rank_id = rank_id, .num_channels = num_channels, .num_rank_ids = num_rank_ids};

    // The local view exists from the start (a lone rank may barrier with itself); self needs no probe
    stage_views();
    activate_staged_views();
    CUDA_RUNTIME_CHECK(cudaMemcpy(ctx_device, &ctx_host, sizeof(NIXLDeviceContext), cudaMemcpyHostToDevice));

    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("NIXL EP context created: agent %s, rank id %d of %d (%d per domain), %d channels, region %p (%ld bytes, %s)\n",
               agent_name.c_str(), rank_id, num_rank_ids, num_scaleup_ranks_per_domain, num_channels, region_base,
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
        for (int i = 0; i < num_rank_ids; ++ i) if (connected[i]) {
            nixl_xfer_dlist_t empty_descs(VRAM_SEG);
            if (state->agent->checkRemoteMD(remote_agent_names[i], empty_descs) == NIXL_SUCCESS)
                state->agent->invalidateRemoteMD(remote_agent_names[i]);
        }
        state->agent->deregisterMem(state->reg_descs, &state->extra_params);
        state->agent.reset();
    }
    state.reset();
    CUDA_RUNTIME_CHECK(cudaFree(ctx_device));
    CUDA_RUNTIME_CHECK(cudaFree(p2p_ptrs_device));
    CUDA_RUNTIME_CHECK(cudaFree(connected_device));
    CUDA_RUNTIME_CHECK(cudaFree(p2p_ptrs_probe_device));
    if (region.ptr == 0)
        return;
    if (region.is_vmm) {
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(region.ptr, region.num_bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(region.handle));
        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(region.ptr, region.num_bytes));
    } else {
        CUDA_RUNTIME_CHECK(cudaFree(region_base));
    }
    region = Region{};
    region_base = nullptr;
}

std::string NIXLEPContext::get_local_metadata() const {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    nixl_blob_t blob;
    NIXL_CHECK(state->agent->getLocalMD(blob));
    return blob;
}

NIXLPeerInfo NIXLEPContext::get_local_peer_info() const {
    return {reinterpret_cast<uint64_t>(region_base), device_id, rank_id, region.num_bytes};
}

void NIXLEPContext::connect_ranks(const std::vector<int>& rank_ids, const std::vector<std::string>& remote_mds,
                                  const std::vector<NIXLPeerInfo>& peer_infos) {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    EP_HOST_ASSERT(rank_ids.size() == remote_mds.size() and rank_ids.size() == peer_infos.size());
    if (rank_ids.empty())
        return;

    // Load every peer's metadata first (the loads proceed in parallel inside the agent)
    std::unordered_set<int> seen;
    for (size_t i = 0; i < rank_ids.size(); ++ i) {
        const int rid = rank_ids[i];
        EP_HOST_ASSERT(0 <= rid and rid < num_rank_ids and rid != rank_id and "Invalid peer rank id");
        EP_HOST_ASSERT(not connected[rid] and seen.insert(rid).second and "Peer already connected");
        EP_HOST_ASSERT(peer_infos[i].rank_id == rid and peer_infos[i].region_base != 0 and "Peer info does not match the rank id");
        EP_HOST_ASSERT(peer_infos[i].num_region_bytes >= static_cast<uint64_t>(num_region_bytes) and "Peer region is too small");
        // A pending disconnect of the id would invalidate the metadata reloaded here at the next commit
        EP_HOST_ASSERT(std::find(pending_disconnects.begin(), pending_disconnects.end(), rid) == pending_disconnects.end() and
                       "Commit the pending disconnect before reconnecting the same rank");
        std::string name;
        NIXL_CHECK(state->agent->loadRemoteMD(remote_mds[i], name));
        remote_agent_names[rid] = name;
        // A different agent behind the id is a new process (a leaver's exited, a fresh one joined): its epochs start from
        // zero, so ours restart (the per-peer barrier words restart at every connect, see `ElasticBuffer::connect_ranks`)
        if (known_agent_names[rid] != name) {
            connect_epoch[rid] = 0;
            known_agent_names[rid] = name;
        }
    }

    // Wait until every peer is usable (the metadata carries the connection info, the wire-up is lazy)
    const auto start = std::chrono::steady_clock::now();
    const auto timeout_secs = get_env<int>("EP_NIXL_CONNECT_TIMEOUT_SECS", 120);
    std::vector<bool> ready(rank_ids.size(), false);
    size_t num_ready = 0;
    while (num_ready < rank_ids.size()) {
        for (size_t i = 0; i < rank_ids.size(); ++ i) if (not ready[i]) {
            nixl_xfer_dlist_t empty_descs(VRAM_SEG);
            if (state->agent->checkRemoteMD(remote_agent_names[rank_ids[i]], empty_descs) == NIXL_SUCCESS)
                ready[i] = true, ++ num_ready;
        }
        if (num_ready < rank_ids.size()) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_secs) {
                std::stringstream ss;
                for (size_t i = 0; i < rank_ids.size(); ++ i) if (not ready[i]) ss << " " << rank_ids[i];
                throw EPException("NIXL", __FILE__, __LINE__, "Timed out waiting for the NIXL metadata of rank ids" + ss.str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // Wire-up handshake: UCX endpoints connect lazily and `prepMemView` needs them connected, so every new peer gets a
    // hello carrying the pair's connection epoch and we wait for its hello back (both sides call `connect_ranks` with
    // each other); a hello of an earlier epoch cannot satisfy this one, an asymmetric protocol shows as an epoch mismatch
    for (size_t i = 0; i < rank_ids.size(); ++ i) {
        const int rid = rank_ids[i];
        connect_epoch[rid] += 1;
        NIXL_CHECK(state->agent->genNotif(remote_agent_names[rid], kHelloPrefix + std::to_string(connect_epoch[rid]),
                                          &state->extra_params));
    }
    std::vector<bool> greeted(rank_ids.size(), false);
    size_t num_greeted = 0;
    while (num_greeted < rank_ids.size()) {
        nixl_notifs_t notif_map;
        NIXL_CHECK(state->agent->getNotifs(notif_map, &state->extra_params));
        for (const auto& [from, msgs]: notif_map) {
            for (const auto& msg: msgs) if (msg.rfind(kHelloPrefix, 0) == 0)
                state->pending_hellos[from].push_back(std::atoi(msg.c_str() + kHelloPrefix.size()));
        }
        for (size_t i = 0; i < rank_ids.size(); ++ i) if (not greeted[i]) {
            const int rid = rank_ids[i];
            auto it = state->pending_hellos.find(remote_agent_names[rid]);
            if (it == state->pending_hellos.end())
                continue;
            auto& epochs = it->second;
            // Drop stale hellos (earlier epochs), consume the matching one
            epochs.erase(std::remove_if(epochs.begin(), epochs.end(), [&](const int& e) { return e < connect_epoch[rid]; }), epochs.end());
            const auto match = std::find(epochs.begin(), epochs.end(), connect_epoch[rid]);
            if (match != epochs.end()) {
                epochs.erase(match);
                greeted[i] = true, ++ num_greeted;
            }
        }
        if (num_greeted < rank_ids.size()) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_secs) {
                std::stringstream ss;
                for (size_t i = 0; i < rank_ids.size(); ++ i) if (not greeted[i]) {
                    const int rid = rank_ids[i];
                    ss << " peer " << rid << " (epoch " << connect_epoch[rid] << ", received:";
                    if (const auto it = state->pending_hellos.find(remote_agent_names[rid]); it != state->pending_hellos.end())
                        for (const int e: it->second) ss << " " << e;
                    ss << ")";
                }
                throw EPException("NIXL", __FILE__, __LINE__, "Timed out waiting for a peer's wire-up notification (the peer must call "
                                  "`connect_ranks` with this rank too, in the same connection epoch); rank id " + std::to_string(rank_id) + " is missing hellos from" + ss.str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Record peers and let NIXL/UCX prepare their mappings with the staged memory views.
    for (size_t i = 0; i < rank_ids.size(); ++ i) {
        const int rid = rank_ids[i];
        peers[rid] = peer_infos[i];
        connected[rid] = 1;
    }
    stage_views();
}

void NIXLEPContext::disconnect_ranks(const std::vector<int>& rank_ids) {
    EP_HOST_ASSERT(state != nullptr and state->agent != nullptr);
    if (rank_ids.empty())
        return;
    for (const int rid: rank_ids) {
        EP_HOST_ASSERT(0 <= rid and rid < num_rank_ids and rid != rank_id and "Invalid peer rank id");
        EP_HOST_ASSERT(connected[rid] and "Peer is not connected");
        connected[rid] = 0;
        peers[rid] = NIXLPeerInfo{};
        pending_invalidations.push_back(remote_agent_names[rid]);
        pending_disconnects.push_back(rid);
        state->pending_hellos.erase(remote_agent_names[rid]);
        remote_agent_names[rid] = "";
    }
    stage_views();
}

bool NIXLEPContext::is_connected(const int& rid) const {
    EP_HOST_ASSERT(0 <= rid and rid < num_rank_ids);
    return rid == rank_id or connected[rid] != 0;
}

std::vector<int> NIXLEPContext::get_connected_ranks() const {
    std::vector<int> out;
    for (int i = 0; i < num_rank_ids; ++ i) if (connected[i])
        out.push_back(i);
    return out;
}

void NIXLEPContext::stage_views() {
    release_staged_views();

    // Local view: our region
    nixl_local_dlist_t local_descs(VRAM_SEG);
    local_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(region_base), static_cast<size_t>(num_region_bytes),
                                      static_cast<uint64_t>(device_id)));
    NIXL_CHECK(state->agent->prepMemView(local_descs, state->staged_local, &state->extra_params));

    // One entry per rank id, a null-agent hole for an absent peer (an all-null view is not a valid request)
    nixl_remote_dlist_t remote_descs(VRAM_SEG);
    for (int r = 0; r < num_rank_ids; ++ r)
        remote_descs.addDesc(connected[r] ?
            nixlRemoteDesc(static_cast<uintptr_t>(peers[r].region_base), static_cast<size_t>(num_region_bytes), static_cast<uint64_t>(peers[r].device_id), remote_agent_names[r]) :
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
    release_view(*state->agent, state->staged_local);
    release_view(*state->agent, state->staged_remote);
    state->has_staged = false;
}

void NIXLEPContext::release_active_views() {
    release_view(*state->agent, state->active_local);
    release_view(*state->agent, state->active_remote);
}

void NIXLEPContext::activate_staged_views() {
    release_active_views();
    std::swap(state->active_local, state->staged_local);
    std::swap(state->active_remote, state->staged_remote);
    state->has_staged = false;
    ctx_host.local_mvh = state->active_local;
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
    probe.p2p_ptrs = p2p_ptrs_probe_device;
    std::vector<void*> probed(num_rank_ids, nullptr);
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(connected_device, connected.data(), num_rank_ids * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_RUNTIME_CHECK(cache_peer_pointers(probe, connected_device, stream));
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(probed.data(), p2p_ptrs_probe_device, num_rank_ids * sizeof(void*), cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

    // A connected same-domain peer without a pointer, or a rail peer with one (the declared domains split a CUDA IPC clique:
    // UCX would then carry every scale-out put as plain stores of one thread, with no release before the arrival flags, see
    // `rdma_put`): refuse with the active views untouched (the staged views stay staged)
    for (int rid = 0; rid < num_rank_ids; ++ rid) {
        const bool same_domain = rid / num_scaleup_ranks_per_domain == rank_id / num_scaleup_ranks_per_domain;
        if (connected[rid] and same_domain == (probed[rid] == nullptr)) {
            std::stringstream ss;
            if (same_domain) ss << "nixlGetPtr returned no NVLink pointer from rank " << rank_id << " to same-domain peer " << rid
               << ": UCX must provide a CUDA IPC device lane for it (keep cuda_ipc in UCX_TLS or leave UCX_TLS "
                  "unset; for an NVLink domain spanning several hosts, UCX_CUDA_IPC_ENABLE_MNNVL=yes with the IMEX "
                  "service running). The previous views stay active";
            else ss << "nixlGetPtr returned an NVLink pointer from rank " << rank_id << " to rail peer " << rid << ": the declared "
                  "NVLink domains split one CUDA IPC clique, which the scale-out path (RDMA writes with arrival flags) does not "
                  "support; declare the whole clique as one domain, or set UCX_CUDA_IPC_ENABLE_MNNVL=n when no domain spans hosts. The previous views stay active";
            throw EPException("NIXL", __FILE__, __LINE__, ss.str());
        }
    }

    // Swap the views in; peers that left: no view references their metadata any more
    activate_staged_views();
    for (const auto& name: pending_invalidations) {
        nixl_xfer_dlist_t empty_descs(VRAM_SEG);
        if (state->agent->checkRemoteMD(name, empty_descs) == NIXL_SUCCESS) {
            const auto status = state->agent->invalidateRemoteMD(name);
            EP_HOST_ASSERT((status == NIXL_SUCCESS or status == NIXL_ERR_NOT_FOUND) and "invalidateRemoteMD failed");
        }
    }
    pending_invalidations.clear();
    pending_disconnects.clear();

    // The device context (the new views) and the p2p table, both at their fixed addresses
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(ctx_device, &ctx_host, sizeof(NIXLDeviceContext), cudaMemcpyHostToDevice, stream));
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(p2p_ptrs_device, p2p_ptrs_probe_device, num_rank_ids * sizeof(void*), cudaMemcpyDeviceToDevice, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

    if (get_env<int>("EP_BUFFER_DEBUG")) {
        std::stringstream ss;
        for (int i = 0; i < num_rank_ids; ++ i) if (i == rank_id or connected[i])
            ss << i << (i == rank_id ? ":self" : probed[i] != nullptr ? ":p2p" : ":rdma") << " ";
        printf("NIXL EP context (rank id %d) committed views (nixlGetPtr), peers: %s\n", rank_id, ss.str().c_str());
    }
}

const NIXLDeviceContext* NIXLEPContext::get_device_context_ptr() const {
    return ctx_device;
}

}  // namespace nixl_ep_v2::elastic::nixl
