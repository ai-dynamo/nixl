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

#include <cuda_runtime.h>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>
#include <pybind11/functional.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/topology.cuh>

#include "../kernels/backend/nixl.hpp"
#include "../kernels/elastic/api.hpp"
#include "../utils/event.hpp"
#include "utils.hpp"

namespace nixl_ep_v2::elastic {

using namespace adaptive;
class ElasticBuffer {
    // Buffer bytes (excludes workspace)
    // Memory layout: [[Workspace] GPU buffer [zero-copy arena]]
    int64_t num_buffer_bytes;
    void* buffer;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // Workspace
    // NOTES: for all workspace, we must keep them as zeros
    void *workspace;
    void *host_workspace, *mapped_host_workspace;
    std::shared_ptr<layout::WorkspaceLayout> workspace_layout_wo_expert;

    // CUDA streams
    at::cuda::CUDAStream comm_stream;

    // Whether to allow multiple reductions
    bool allow_multiple_reduction;

    // Whether to prefer overlapping communication with compute (use more SMs and channels if false)
    bool prefer_overlap_with_compute;

    // Dynamic token take of the direct paths (the kernels' `dynamic_take`): 0 off (the default: the static partition),
    // 1 on, 2 automatic = on in direct mode (measured 1x16: the static shares meet the per-SM egress arbitration tiers from
    // 16 active SMs up and the take removes them at every batch size, never slower); the hybrid paths' division is pinned
    // by the notify count protocol and ignores the flag
    int dynamic_take_mode;

    // Arrival-flag protocol of the scale-out receivers (the forwarder and the combine's reduce epilogue): false (the default) = the
    // review tip's relaxed polls of a record's per-packet flags followed by one fence.acq_rel before the landed record is read;
    // true = acquire polls and, in the forwarder, only the proxy fence its bulk load needs (4x8 dispatch measured -10 to -13 % at
    // 16-128 tokens, -5 % at 1024, -2 % at 4096; combine unchanged). Compile-time in the kernels (one instantiation per setting)
    bool acquire_flags;

    // Timeout settings
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;

    // The transport: this rank's NIXL agent, its registered region (workspace + GPU buffer) and the peers' views
    std::shared_ptr<nixl::NIXLEPContext> nixl_context;

    // The declared maximum world and this rank's coordinates in it; the staged topology descriptor (`EPTopology`) awaits `commit_topology`
    int max_num_scaleout_ranks = 0, max_num_scaleup_ranks = 0;
    int scaleout_rank_idx = 0, scaleup_rank_idx = 0;
    // Fixed expert slots per rank: the live expert count is the live rank count times this, so a rank's slots
    // (and the expert-indexed handle tensors) keep their size across topology changes
    int num_experts_per_rank = 0;
    // Adaptive kernels: device-side topology descriptor (+ host mirror) and the fixed channel count per SM
    torch::Tensor topology_tensor;
    topology::EPTopology topology_host{}, staged_topology{};
    bool has_staged_topology = false;
    mutable std::optional<int> adaptive_num_channels_per_sm;

    // Zero copy: the output arena at the end of the registered region, at the same offsets on every rank
    struct ZeroCopyArena {
        bool reserved = false;
        int num_slots = 0;
        int num_max_tokens_per_rank = 0, num_hidden_bytes = 0, num_sf_packs = 0;   // the wire: bytes per row, scale-factor packs per row
        int num_topk = 0, num_experts_per_rank = 0, expert_alignment = 1;
        bool do_expand = false, use_tma_aligned_col_major_sf = false;
        int64_t num_compact_rows = 0, num_rows = 0;
        int sf_token_stride = 0, sf_hidden_stride = 0;
        int64_t recv_x_offset = 0, recv_sf_offset = 0, recv_topk_idx_offset = 0;
        int64_t recv_topk_weights_offset = 0, recv_src_metadata_offset = 0, slot_bytes = 0;
        int64_t count_matrix_offset = 0, count_matrix_bytes = 0;
        int count_row_stride = 0, count_matrix_rows = 0;
        int64_t num_total_bytes = 0;
        uint8_t* base = nullptr;
    } zc;
    int64_t num_zero_copy_bytes = 0;

public:
    // The 2 MB unit of the workspace, the GPU buffer and the size functions (`get_elastic_buffer_alignment`)
    static constexpr int64_t kNumAlignmentBytes = 2097152;

    ElasticBuffer(const int64_t& num_buffer_bytes,
                  const bool& allow_multiple_reduction,
                  const bool& prefer_overlap_with_compute,
                  const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
                  const bool& explicitly_destroy,
                  const int& max_num_scaleout_ranks, const int& max_num_scaleup_ranks,
                  const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                  const int& num_nixl_channels, const std::string& agent_name_prefix,
                  const int& num_experts_per_rank,
                  const int64_t& num_zero_copy_bytes,
                  const int& dynamic_take_mode,
                  const bool& acquire_flags):
        num_buffer_bytes(num_buffer_bytes),
        explicitly_destroy(explicitly_destroy),
        comm_stream(get_global_comm_stream()),
        allow_multiple_reduction(allow_multiple_reduction),
        prefer_overlap_with_compute(prefer_overlap_with_compute),
        dynamic_take_mode(dynamic_take_mode), acquire_flags(acquire_flags),
        max_num_scaleout_ranks(max_num_scaleout_ranks), max_num_scaleup_ranks(max_num_scaleup_ranks), scaleout_rank_idx(scaleout_rank_idx), scaleup_rank_idx(scaleup_rank_idx),
        num_experts_per_rank(num_experts_per_rank), num_zero_copy_bytes(num_zero_copy_bytes) {
        // Check buffer bytes alignment (2 MB)
        EP_HOST_ASSERT(num_buffer_bytes > 0 and num_buffer_bytes % kNumAlignmentBytes == 0);
        EP_HOST_ASSERT(0 <= dynamic_take_mode and dynamic_take_mode <= 2 and "dynamic_take_mode: 0 off, 1 on, 2 automatic");
        EP_HOST_ASSERT(num_zero_copy_bytes >= 0 and num_zero_copy_bytes % kNumAlignmentBytes == 0 and "The zero-copy arena is a whole number of 2 MB units");

        // Fixed expert slots per rank (0: a buffer that never dispatches); the slots of the declared maximum world
        // must fit the workspace's expert capacity
        EP_HOST_ASSERT(0 <= num_experts_per_rank and num_experts_per_rank <= layout::WorkspaceLayout::kNumMaxExpertsPerRank and
                       "Bad number of expert slots per rank");
        EP_HOST_ASSERT(static_cast<int64_t>(max_num_scaleout_ranks) * max_num_scaleup_ranks * num_experts_per_rank <=
                       layout::WorkspaceLayout::kNumMaxExperts and
                       "The expert slots of the declared maximum world exceed the workspace capacity");

        // Workspace is aligned to 2 MB so that it sits cleanly at the front of the symmetric memory
        const auto num_workspace_bytes = math::align<int64_t>(
            layout::WorkspaceLayout::get_num_bytes(), kNumAlignmentBytes);

        // The declared maximum world and this rank's coordinates in it
        EP_HOST_ASSERT(0 < max_num_scaleout_ranks and max_num_scaleout_ranks <= topology::kNumMaxScaleoutRanksCap and
                       0 < max_num_scaleup_ranks and max_num_scaleup_ranks <= topology::kNumMaxScaleupRanksCap and "Bad declared maximum world");
        EP_HOST_ASSERT(0 <= scaleout_rank_idx and scaleout_rank_idx < max_num_scaleout_ranks and
                       0 <= scaleup_rank_idx and scaleup_rank_idx < max_num_scaleup_ranks and
                       "This rank's coordinates are outside the declared maximum world");

        // Create the transport: one symmetric region per rank, registered with this rank's agent
        // Symmetric memory layout: [[Workspace] GPU buffer [zero-copy arena]]
        const int rank_id = topology::rank_id(scaleout_rank_idx, scaleup_rank_idx, max_num_scaleup_ranks);
        this->nixl_context = std::make_shared<nixl::NIXLEPContext>(
            agent_name_prefix + "_" + std::to_string(rank_id), rank_id, max_num_scaleout_ranks * max_num_scaleup_ranks,
            max_num_scaleup_ranks, num_workspace_bytes + num_buffer_bytes + num_zero_copy_bytes, num_nixl_channels);
        workspace = this->nixl_context->region_base;

        // Topology descriptor (`EPTopology`); its `mode` follows the live domains: the Python layer derives it for every world it stages
        topology_tensor = torch::empty({static_cast<int64_t>(sizeof(topology::EPTopology))},
                                       torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

        // Timeout
        this->num_cpu_timeout_secs = num_cpu_timeout_secs;
        this->num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs);
        this->num_gpu_timeout_cycles *= jit::device_runtime->get_clock_rate();

        // JIT the barrier now: `destroy()` runs it, and a leaver that never ran it would otherwise compile it during teardown
        build_barrier(max_num_scaleup_ranks, num_gpu_timeout_cycles);

        // Assign workspaces and buffers
        workspace_layout_wo_expert = std::make_shared<layout::WorkspaceLayout>(
            workspace, this->max_num_scaleout_ranks, this->max_num_scaleup_ranks, 0);
        buffer = static_cast<uint8_t*>(workspace) + num_workspace_bytes;
        zc.base = num_zero_copy_bytes > 0 ? static_cast<uint8_t*>(buffer) + num_buffer_bytes : nullptr;
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, num_workspace_bytes));

        // The initial topology descriptor (`EPTopology`): this rank alone (one live domain, hence direct), every launched block active
        std::vector<uint64_t> words(scaleup_rank_idx / 64 + 1); words.back() = 1ull << (scaleup_rank_idx % 64);
        stage_topology_masks(topology::kModeDirect, 1u << scaleout_rank_idx, words, topology::get_num_launch_sms());
        commit_topology();

        // Allocate host workspaces
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
        std::memset(host_workspace, 0, layout::WorkspaceLayout::get_num_bytes());

        // The constructor ends without a barrier: the first `commit_topology` of `connect_group` orders the cohort
    }

    ~ElasticBuffer() noexcept(false) {
        if (not explicitly_destroy)
            destroy();

        if (not destroyed) {
            printf("`destroy()` is not called before NIXL EP V2 elastic buffer destruction, which can leak resources.\n");
            fflush(stdout);
        }
    }

    void destroy() {
        EP_HOST_ASSERT(not destroyed);

        // Finish all works on all GPUs
        barrier(true, true);

        // Deallocate host workspaces
        CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));

        // Destroy the transport
        nixl_context->finalize();

        // Cannot use anymore
        destroyed = true;
    }

    torch::Stream get_comm_stream() const {
        return comm_stream;
    }

    std::tuple<int, int> get_physical_domain_size() const {
        return {max_num_scaleout_ranks, max_num_scaleup_ranks};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        return {max_num_scaleout_ranks, max_num_scaleup_ranks};
    }

    // ---- Topology descriptor (`EPTopology`) ----
    // Two-step and collective: `stage_topology_masks` validates a world on the host, `commit_topology` writes it to the device
    // with one stream-ordered copy (after the transport's view swap when peers changed). Membership: see `topology.cuh`

    // Python-facing: the position mask as 64-bit words, least significant first (missing words are zero)
    void stage_topology_masks(const int& mode, const uint32_t& scaleout_mask, const std::vector<uint64_t>& scaleup_mask_words,
                              const int& num_active_sms) {
        EP_HOST_ASSERT(scaleup_mask_words.size() <= static_cast<size_t>(topology::kNumScaleupMaskWords) and
                       "The position mask exceeds `kNumMaxScaleupRanksCap` bits");
        topology::ScaleupMask scaleup_mask = {};
        for (size_t i = 0; i < scaleup_mask_words.size(); ++ i)
            scaleup_mask.words[i] = scaleup_mask_words[i];
        EP_HOST_ASSERT(mode == topology::kModeDirect or mode == topology::kModeHybrid);
        EP_HOST_ASSERT(scaleout_mask != 0 and scaleup_mask.popc() > 0 and "The live world is empty");
        const int num_scaleout_ranks = 32 - __builtin_clz(scaleout_mask);
        const int num_scaleup_ranks = scaleup_mask.bound();
        EP_HOST_ASSERT(num_scaleout_ranks <= max_num_scaleout_ranks and num_scaleup_ranks <= max_num_scaleup_ranks and
                       "The world exceeds the declared maximum (`max_num_scaleout_ranks` / `max_num_scaleup_ranks`)");
        EP_HOST_ASSERT((mode == topology::kModeHybrid or __builtin_popcount(scaleout_mask) == 1) and
                       "Direct mode needs one flat NVLink team (exactly one live NVLink domain)");
        EP_HOST_ASSERT(2 <= num_active_sms and num_active_sms <= topology::get_num_launch_sms() and num_active_sms % 2 == 0);

        // A participant's kernels address every live peer (a rank outside the live world launches nothing and may stage anything)
        if (((scaleout_mask >> scaleout_rank_idx) & 1u) and scaleup_mask.test(scaleup_rank_idx))
            for (int d = 0; d < num_scaleout_ranks; ++ d) for (int k = 0; k < num_scaleup_ranks; ++ k)
                EP_HOST_ASSERT((not (((scaleout_mask >> d) & 1u) and scaleup_mask.test(k)) or
                                nixl_context->is_connected(topology::rank_id(d, k, max_num_scaleup_ranks))) and "A live rank is not connected: call `connect_ranks` first");
        staged_topology.transport = nixl_context->get_device_context_ptr();

        staged_topology.mode = static_cast<topology::Mode>(mode);
        staged_topology.num_scaleout_ranks = num_scaleout_ranks;
        staged_topology.num_scaleup_ranks = num_scaleup_ranks;
        staged_topology.scaleout_rank_idx = scaleout_rank_idx;
        staged_topology.scaleup_rank_idx = scaleup_rank_idx;
        staged_topology.num_active_sms = num_active_sms;
        staged_topology.scaleout_mask = scaleout_mask;
        staged_topology.scaleup_mask = scaleup_mask;
        has_staged_topology = true;
    }

    void commit_topology() {
        EP_HOST_ASSERT((has_staged_topology or nixl_context->has_staged_views()) and
                       "Nothing staged: call `stage_topology`, `connect_ranks` or `disconnect_ranks` first");
        EP_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None and
                       "`commit_topology` inside a CUDA-graph capture: commit between launches");
        // Peers changed but the world did not (e.g. dropping the peers a previous commit excluded): recommit it
        if (not has_staged_topology)
            staged_topology = topology_host;
        const auto stream = at::cuda::getCurrentCUDAStream();

        // Peers changed since the last commit: swap the transport's views in first (a device sync inside)
        if (nixl_context->has_staged_views()) {
            nixl_context->commit(stream);
        }

        // A world change resets the call-tagged words (so every rank commits every world change, participant or not)
        bool world_changed = staged_topology.scaleout_mask != topology_host.scaleout_mask;
        for (int i = 0; i < topology::kNumScaleupMaskWords; ++ i)
            world_changed |= staged_topology.scaleup_mask.words[i] != topology_host.scaleup_mask.words[i];
        if (zc.reserved and world_changed) {
            stream_wait(stream, comm_stream);
            const auto& wl = *workspace_layout_wo_expert;
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(zc.base + zc.count_matrix_offset, 0, zc.count_matrix_bytes, stream));
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(wl.get_zc_fresh_call_counter_ptr(), 0,
                                               layout::WorkspaceLayout::kNumZeroCopyControlBytes, stream));
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(wl.get_zc_rank_base_ptr(), 0,
                                               topology::kNumMaxScaleupRanksCap * sizeof(int64_t), stream));
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        }

        // One stream-ordered write of the staged topology descriptor (`EPTopology`) on the current stream, which the
        // communication stream then waits for (a pageable source: CUDA takes the bytes now and refuses it under capture)
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(topology_tensor.data_ptr(), &staged_topology, sizeof(topology::EPTopology),
                                           cudaMemcpyHostToDevice, stream));
        stream_wait(comm_stream, stream);

        topology_host = staged_topology;
        has_staged_topology = false;
    }

    // Mode and/or active SM count only: stage with the committed world (a pending staged one is discarded) and commit at once; `dispatch` reaches it through `num_sms`
    void set_topology(const int& mode, const int& num_active_sms) {
        stage_topology_masks(mode, topology_host.scaleout_mask, std::get<1>(get_topology_masks()), num_active_sms);
        commit_topology();
    }

    std::tuple<int, int> get_topology() const {
        return {static_cast<int>(topology_host.mode), topology_host.num_active_sms};
    }

    // The take setting of the next launch (see `dynamic_take_mode`): automatic follows the committed mode
    bool resolve_dynamic_take() const {
        const bool take = dynamic_take_mode == 2 ? topology_host.mode == topology::kModeDirect : dynamic_take_mode == 1;
        if (get_env<int>("NIXL_EP_BUFFER_DEBUG")) {
            static int last_printed = -1;
            if (last_printed != static_cast<int>(take)) {
                printf("Elastic buffer dynamic take: %s (setting %s, mode %s)\n", take ? "on" : "off",
                       dynamic_take_mode == 2 ? "automatic" : (dynamic_take_mode == 1 ? "on" : "off"),
                       topology_host.mode == topology::kModeDirect ? "direct" : "hybrid");
                last_printed = static_cast<int>(take);
            }
        }
        return take;
    }

    const topology::EPTopology* get_topology_ptr() const {
        return static_cast<const topology::EPTopology*>(topology_tensor.data_ptr());
    }

    // The live masks of the committed topology descriptor (`EPTopology`), the position mask as its 64-bit words
    std::tuple<uint32_t, std::vector<uint64_t>> get_topology_masks() const {
        std::vector<uint64_t> words(topology_host.scaleup_mask.words, topology_host.scaleup_mask.words + topology::kNumScaleupMaskWords);
        return {topology_host.scaleout_mask, words};
    }

    // The live world of the committed topology descriptor (`EPTopology`): the bounds of the two masks
    std::tuple<int, int> get_topology_ranks() const {
        return {topology_host.num_scaleout_ranks, topology_host.num_scaleup_ranks};
    }

    // Whether this rank is inside the live world (a rank outside it must not call `dispatch`/`combine`)
    bool is_participant() const {
        return topology_host.is_participant();
    }

    // ---- Peers ----
    // Join: publish the metadata and peer info, `connect_ranks`, then stage and commit. Leave: commit a world without the rank, then `disconnect_ranks`
    pybind11::bytes get_local_metadata() const {
        return pybind11::bytes(nixl_context->get_local_metadata());
    }

    // `(region base, device, rank id, allocation bytes)`
    using peer_info_t = std::tuple<uint64_t, int, int, uint64_t>;

    peer_info_t get_local_peer_info() const {
        const auto info = nixl_context->get_local_peer_info();
        return {info.region_base, info.device_id, info.rank_id, info.num_region_bytes};
    }

    void connect_ranks(const std::vector<int>& rank_ids, const std::vector<std::string>& remote_mds,
                       const std::vector<peer_info_t>& peer_infos) {
        EP_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None and
                       "`connect_ranks` inside a CUDA-graph capture: connect between launches");
        // A connected pair counts its barriers from zero on both sides (the peer runs this for us too): zero our words for
        // the id now, before our hello goes out, since the peer can write to us only once it has our hello and committed
        const auto stream = at::cuda::getCurrentCUDAStream();
        for (const int rid: rank_ids) {
            EP_HOST_ASSERT(not nixl_context->is_connected(rid) and "Peer already connected");
            const auto& wl = *workspace_layout_wo_expert;
            for (int parity = 0; parity < 2; ++ parity) {
                CUDA_RUNTIME_CHECK(cudaMemsetAsync(wl.get_peer_barrier_signal_ptr(rid, parity), 0, sizeof(int64_t), stream));
                CUDA_RUNTIME_CHECK(cudaMemsetAsync(wl.get_peer_barrier_staging_ptr(rid, parity), 0, sizeof(int64_t), stream));
            }
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(wl.get_peer_barrier_expected_ptr(rid), 0, sizeof(int64_t), stream));
        }
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        std::vector<nixl::NIXLPeerInfo> infos;
        for (const auto& [base, device, id, num_bytes]: peer_infos)
            infos.push_back({base, device, id, num_bytes});
        {
            // Let other Python threads progress while the native connection handshake waits for peers.
            pybind11::gil_scoped_release release;
            nixl_context->connect_ranks(rank_ids, remote_mds, infos);
        }
    }

    void disconnect_ranks(const std::vector<int>& rank_ids) {
        // A participant's kernels address every live peer: commit a world without the rank first (a non-participant may drop anyone)
        if (topology_host.is_participant()) {
            for (const int rid: rank_ids) {
                EP_HOST_ASSERT(not topology_host.is_live(rid / max_num_scaleup_ranks, rid % max_num_scaleup_ranks) and
                               "Commit a topology without the rank before disconnecting it");
            }
        }
        nixl_context->disconnect_ranks(rank_ids);
    }

    std::vector<int> get_connected_ranks() const {
        return nixl_context->get_connected_ranks();
    }

    // Debug: the phase timestamps written by kernels compiled with EP_ADAPTIVE_PROFILE=1 (ns, globaltimer)
    std::vector<int64_t> read_profile() const {
        std::vector<int64_t> values(layout::WorkspaceLayout::kNumProfileSlots);
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        CUDA_RUNTIME_CHECK(cudaMemcpy(values.data(), workspace_layout_wo_expert->get_profile_ptr(),
                                      values.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
        // The "last block done" slots accumulate with a max, so they must start from zero every iteration
        CUDA_RUNTIME_CHECK(cudaMemset(workspace_layout_wo_expert->get_profile_ptr(), 0, values.size() * sizeof(int64_t)));
        return values;
    }

    // ReSharper disable once CppMemberFunctionMayBeStatic
    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync) const {
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto stream = use_comm_stream ? comm_stream : compute_stream;
        if (use_comm_stream)
            stream_wait(comm_stream, compute_stream);

        // Wait all streams to finish on this GPU
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Launch GPU barrier
        launch_barrier(workspace, get_topology_ptr(), max_num_scaleup_ranks, num_gpu_timeout_cycles, stream);

        // Let CPU wait
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Compute stream should also wait for the barrier
        if (use_comm_stream)
            stream_wait(compute_stream, comm_stream);
    }

    torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                    const bool& allocate_on_comm_stream,
                                                    const bool& async_with_compute_stream) const {
        // Allocate all tensors on communication stream if set
        // NOTES: do not allocate tensors upfront!
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(comm_stream);

        // Assertion for safety
        // `previous_event` implicitly means the overlapping computation kernels are launched first,
        // in order not to use the memory on the compute stream, we must allocate on the communication stream.
        // If you launch the communication kernels firstly, then `previous_event` must be unnecessary.
        if (previous_event.has_value())
            EP_HOST_ASSERT(allocate_on_comm_stream);

        // Wait previous tasks to finish
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }
        return compute_stream;
    }

    void stream_control_before_epilogue(const std::optional<EventHandle>& previous_event_before_epilogue) const {
        if (previous_event_before_epilogue.has_value())
            stream_wait(comm_stream, previous_event_before_epilogue.value());
    }

    std::optional<EventHandle> stream_control_epilogue(const std::vector<std::optional<torch::Tensor>>& tensors,
                                                       const at::cuda::CUDAStream& compute_stream,
                                                       const bool& allocate_on_comm_stream,
                                                       const bool& async_with_compute_stream) const {
        // Ensure memory access safety between two streams
        std::optional<EventHandle> event;
        if (async_with_compute_stream) {
            event = EventHandle(comm_stream);

            // NOTES: this environment only applies to V2 APIs
            if (get_env<int>("NIXL_EP_AVOID_RECORD_STREAM", 0)) {
                event->tensors_to_record = tensors;
            } else {
                for (auto& t: tensors) if (t.has_value()) {
                    t->record_stream(compute_stream);
                    t->record_stream(comm_stream);
                }
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // The CUDA event marking the finishing
        return event;
    }

    static std::tuple<int64_t, int64_t> get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& num_hidden_bytes, const int& num_sf_packs, const int& num_topk,
                                            const int& num_scaleout_ranks, const int& num_scaleup_ranks) {
        const auto token_layout = get_dispatch_token_layout(num_hidden_bytes, num_sf_packs, num_topk);

        const auto scaleup_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);
        const auto scaleout_send_buffer = layout::BufferLayout<false>(
            token_layout, 1, num_max_tokens_per_rank);
        const auto scaleout_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_scaleout_ranks,
            /* num_channels * num_max_tokens_per_channel */ num_max_tokens_per_rank + adaptive::kNumMaxChannels);
        return {scaleup_recv_buffer.get_num_bytes() + scaleout_recv_buffer.get_num_bytes(),
                scaleout_send_buffer.get_num_bytes()};
    }

    static std::tuple<int64_t, int64_t> get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const bool& allow_multiple_reduction) {
        const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

        const int num_tokens_in_scaleup_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;
        const int num_tokens_in_scaleout_layout = allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;
        const auto scaleup_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);
        const auto scaleout_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
        const auto scaleout_send_buffer = layout::BufferLayout<false>(
            token_layout, allow_multiple_reduction ? 1 : num_topk,
            /* num_channels * num_scaleout_ranks * num_max_tokens_per_channel */
            num_scaleout_ranks * (num_max_tokens_per_rank + adaptive::kNumMaxChannels));
        return {scaleup_recv_buffer.get_num_bytes() + scaleout_recv_buffer.get_num_bytes(),
                scaleout_send_buffer.get_num_bytes()};
    }

    // How the token buffer is carved: the dispatch and the combine overlay their RECEIVE regions on `[0, num_shared_bytes)`
    // (sized for the larger of the two); the scale-out SEND buffers of the two kernels sit at the END of the GPU buffer, each
    // in a region of its own, because the NIC keeps reading the sources of RDMA puts after the kernel that issued them has ended
    static std::tuple<int64_t, int64_t, int64_t> get_buffer_arena(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                                                  const bool& allow_multiple_reduction,
                                                                  const int& num_scaleout_ranks, const int& num_scaleup_ranks) {
        // The BF16-wire dispatch record is the larger one (the FP8 and NVFP4 wires add their SF bytes to a half- or quarter-size row)
        const auto [num_dispatch_recv_bytes, num_dispatch_send_bytes] = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden * sizeof(nv_bfloat16), 0, num_topk, num_scaleout_ranks, num_scaleup_ranks);
        const auto [num_combine_recv_bytes, num_combine_send_bytes] = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk, num_scaleout_ranks, num_scaleup_ranks, allow_multiple_reduction);
        return {std::max(num_dispatch_recv_bytes, num_combine_recv_bytes), num_dispatch_send_bytes, num_combine_send_bytes};
    }

    // Buffer size for a declared maximum world (the live sub-layouts are prefixes of the maximum ones)
    static int64_t calculate_buffer_size(const int& max_num_scaleout_ranks, const int& max_num_scaleup_ranks,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk, const bool& use_fp8_dispatch,
                                         const bool& allow_multiple_reduction) {
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

        // The worst case SF bytes must be less than the main part
        EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

        // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Shared receive regions plus the two send buffers (see `get_buffer_arena`), aligned to 2 MB
        const auto [num_shared_bytes, num_dispatch_send_bytes, num_combine_send_bytes] = get_buffer_arena(
            num_max_tokens_per_rank, hidden, num_topk, allow_multiple_reduction, max_num_scaleout_ranks, max_num_scaleup_ranks);
        return math::align(num_shared_bytes + num_dispatch_send_bytes + num_combine_send_bytes, kNumAlignmentBytes);
    }

    // The arena's wire is its row in bytes and its scale-factor packs per row (BF16: hidden * 2 and 0; FP8: hidden and hidden / 128;
    // NVFP4: hidden / 2 and hidden / 64), the two things the kernels move
    static ZeroCopyArena get_zero_copy_layout(const int& max_num_scaleout_ranks, const int& max_num_scaleup_ranks,
                                              const int& num_max_tokens_per_rank, const int& num_hidden_bytes,
                                              const int& num_sf_packs, const int& num_topk, const int& num_experts_per_rank,
                                              const int& expert_alignment, const bool& do_expand,
                                              const bool& use_tma_aligned_col_major_sf, const int& num_slots) {
        ZeroCopyArena zc;
        EP_HOST_ASSERT(0 < max_num_scaleout_ranks and max_num_scaleout_ranks <= topology::kNumMaxScaleoutRanksCap and 0 < max_num_scaleup_ranks and max_num_scaleup_ranks <= topology::kNumMaxScaleupRanksCap and "Bad declared maximum world");
        EP_HOST_ASSERT(1 <= num_slots and num_slots <= layout::WorkspaceLayout::kNumMaxZeroCopySlots and "The zero-copy arena holds 1 to 8 slots");
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and num_hidden_bytes > 0 and num_sf_packs >= 0 and num_topk > 0 and num_experts_per_rank > 0 and expert_alignment >= 1 and "The zero-copy arena needs the MoE sizes");
        EP_HOST_ASSERT(num_hidden_bytes % sizeof(int4) == 0 and "A zero-copy row is one 16-byte-granular bulk store");
        EP_HOST_ASSERT(num_max_tokens_per_rank + 1 < (1 << layout::WorkspaceLayout::kNumZcCountBits) and "The count allgather's word holds `count + 1` in 20 bits");
        EP_HOST_ASSERT(not do_expand and expert_alignment == 1 and not use_tma_aligned_col_major_sf and "The zero-copy arena is compact with row-major scaling factors");

        zc.num_slots = num_slots;
        zc.num_max_tokens_per_rank = num_max_tokens_per_rank;
        zc.num_hidden_bytes = num_hidden_bytes, zc.num_sf_packs = num_sf_packs;
        zc.num_topk = num_topk, zc.num_experts_per_rank = num_experts_per_rank, zc.expert_alignment = expert_alignment;
        zc.do_expand = do_expand, zc.use_tma_aligned_col_major_sf = use_tma_aligned_col_major_sf;

        const int64_t max_num_ranks = static_cast<int64_t>(max_num_scaleout_ranks) * max_num_scaleup_ranks;
        zc.num_compact_rows = static_cast<int64_t>(num_max_tokens_per_rank) * max_num_ranks;
        zc.num_rows = zc.num_compact_rows;

        int64_t num_sf_elems = 0;
        if (num_sf_packs > 0) {
            zc.sf_token_stride = num_sf_packs, zc.sf_hidden_stride = 1;
            num_sf_elems = zc.num_rows * num_sf_packs;
        }

        // Slot fields, 256-byte aligned; then the count matrix: two parities of one row per position plus a scratch row
        const auto align256 = [](const int64_t& num_bytes) { return math::align<int64_t>(num_bytes, 256); };
        int64_t offset = 0;
        zc.recv_x_offset = offset, offset += align256(zc.num_rows * num_hidden_bytes);
        zc.recv_sf_offset = offset, offset += align256(num_sf_elems * sizeof(sf_pack_t));
        zc.recv_topk_idx_offset = offset, offset += align256(zc.num_compact_rows * num_topk * sizeof(topk_idx_t));
        zc.recv_topk_weights_offset = offset, offset += align256(zc.num_compact_rows * num_topk * sizeof(float));
        zc.recv_src_metadata_offset = offset, offset += align256(zc.num_compact_rows * (num_topk + 2) * sizeof(int));
        zc.slot_bytes = align256(offset);

        zc.count_matrix_rows = max_num_scaleup_ranks;
        // A count row carries a writer's rank and expert counts in the declared world's GLOBAL numbering (direct mode runs on one live
        // domain at any index and keeps the global numbering); the matrix has one row per position of a domain
        zc.count_row_stride = math::align(max_num_scaleout_ranks * max_num_scaleup_ranks * (1 + num_experts_per_rank), 4);
        zc.count_matrix_offset = align256(zc.slot_bytes * num_slots);
        zc.count_matrix_bytes = align256((2 * zc.count_matrix_rows + 1) * static_cast<int64_t>(zc.count_row_stride) * sizeof(int));
        zc.num_total_bytes = math::align<int64_t>(zc.count_matrix_offset + zc.count_matrix_bytes, kNumAlignmentBytes);
        return zc;
    }

    static int64_t calculate_zero_copy_arena_size(const int& max_num_scaleout_ranks, const int& max_num_scaleup_ranks,
                                                  const int& num_max_tokens_per_rank, const int& num_hidden_bytes,
                                                  const int& num_sf_packs, const int& num_topk, const int& num_experts_per_rank,
                                                  const int& expert_alignment, const bool& do_expand,
                                                  const bool& use_tma_aligned_col_major_sf, const int& num_slots) {
        return get_zero_copy_layout(max_num_scaleout_ranks, max_num_scaleup_ranks, num_max_tokens_per_rank, num_hidden_bytes,
                                    num_sf_packs, num_topk, num_experts_per_rank, expert_alignment, do_expand,
                                    use_tma_aligned_col_major_sf, num_slots).num_total_bytes;
    }

    // `hidden` is the model's (the channel count's input); `num_hidden_bytes` / `num_sf_packs` the arena's wire
    void configure_zero_copy_arena(const int& num_slots,
                                   const int& num_max_tokens_per_rank, const int& hidden, const int& num_hidden_bytes,
                                   const int& num_sf_packs, const int& num_topk,
                                   const int& expert_alignment,
                                   const bool& do_expand, const bool& use_tma_aligned_col_major_sf) {
        EP_HOST_ASSERT(not zc.reserved and "The zero-copy arena is configured once per buffer");
        EP_HOST_ASSERT(num_zero_copy_bytes > 0 and zc.base != nullptr and "The buffer was constructed without an arena");
        EP_HOST_ASSERT(not adaptive_num_channels_per_sm.has_value() and "Configure the zero-copy arena before the first dispatch");
        EP_HOST_ASSERT(num_experts_per_rank > 0 and "Zero copy needs `num_experts_per_rank` at buffer construction");
        EP_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None and "`configure_zero_copy_arena` inside a CUDA-graph capture");
        auto arena = get_zero_copy_layout(max_num_scaleout_ranks, max_num_scaleup_ranks, num_max_tokens_per_rank, num_hidden_bytes,
                                          num_sf_packs, num_topk, num_experts_per_rank, expert_alignment, do_expand,
                                          use_tma_aligned_col_major_sf, num_slots);
        EP_HOST_ASSERT(arena.num_total_bytes <= num_zero_copy_bytes and "The arena layout exceeds the constructor's reservation");

        const int max_num_ranks = max_num_scaleout_ranks * max_num_scaleup_ranks;
        const int num_channels_per_sm = get_num_channels_per_sm(
            hidden, num_hidden_bytes, num_sf_packs, num_topk, max_num_ranks * num_experts_per_rank,
            max_num_ranks, max_num_scaleout_ranks, jit::device_runtime->get_num_smem_bytes(), prefer_overlap_with_compute);
        const int num_threads = (kNumNotifyWarps + 2 * num_channels_per_sm) * 32;
        EP_HOST_ASSERT(static_cast<int64_t>(max_num_scaleup_ranks) * math::ceil_div(max_num_ranks * (1 + num_experts_per_rank), 4) <= 64 * static_cast<int64_t>(num_threads) and "The declared world's count rows exceed the sweep's chunk budget (one row per position, the world's rank and expert words each)");

        arena.base = zc.base;
        zc = arena;

        // The matrix must start without a tag the counter will use (fresh memory is not zero)
        const auto stream = at::cuda::getCurrentCUDAStream();
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(zc.base + zc.count_matrix_offset, 0, zc.count_matrix_bytes, stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        zc.reserved = true;
    }

    bool has_zero_copy() const {
        return zc.reserved;
    }

    int get_zero_copy_num_slots() const {
        return zc.num_slots;
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               int, int,
               std::vector<int>,
               torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor,
               torch::Tensor, torch::Tensor,
               std::optional<EventHandle>>
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const int& num_max_tokens_per_rank,
             const int& expert_alignment,
             const int& num_sms,
             const std::optional<EventHandle>& previous_event,
             const std::optional<EventHandle>& previous_event_before_epilogue,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_handle_copy, const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf,
             const bool& zero_copy, const int& zero_copy_slot) const {
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0);

        // Zero padding only makes sense with expand mode
        EP_HOST_ASSERT(not do_zero_padding or do_expand);

        EP_HOST_ASSERT(topology_host.is_participant() and
                       "This rank is outside the live world of the topology descriptor (see `stage_topology`)");
        const auto& topo = topology_host;

        if (zero_copy) {
            EP_HOST_ASSERT(zc.reserved and "Zero copy needs an arena: construct with `num_zero_copy_slots > 0`");
            // Zero copy writes the slots in place over NVLink: a direct-mode feature. Under `kZeroCopy` in hybrid mode the kernels run
            // the staged arms and the copy epilogue writes the slot, a fallback only a graph replay reaches (captured in direct mode,
            // replayed after the world grew to two domains): an eager call or a capture in hybrid mode is refused here
            EP_HOST_ASSERT(topo.mode == topology::kModeDirect and "Zero copy is a direct-mode feature (one NVLink domain): not available while the topology is hybrid");
            EP_HOST_ASSERT(0 <= zero_copy_slot and zero_copy_slot < zc.num_slots and "Invalid zero-copy slot");
        }

        // Check data tensor
        // NVFP4 wire: `x` holds two e2m1 codes per byte (`[tokens, hidden / 2]` uint8) and `sf` one e4m3 scale per 16 elements
        // as int32 packs of four (`[tokens, hidden / 64]`); the model hidden sizes the arena and the combine layout
        const auto [num_tokens, num_cols] = get_shape<2>(x);
        const bool nvfp4 = x.scalar_type() == torch::kByte;
        const int hidden = nvfp4 ? 2 * num_cols : num_cols;
        const auto num_hidden_bytes = num_cols * static_cast<int>(x.element_size());
        EP_HOST_ASSERT((not nvfp4 or (sf.has_value() and hidden % 64 == 0)) and "NVFP4 dispatch takes (codes [tokens, hidden / 2] uint8, scales [tokens, hidden / 64] int32 packs) with hidden % 64 == 0");
        EP_HOST_ASSERT(not (nvfp4 and use_tma_aligned_col_major_sf) and "NVFP4 scales are returned row-major only");
        // Fixed slots: the expert count of the live world is its rank count times the slots per rank
        EP_HOST_ASSERT(num_experts_per_rank > 0 and "Dispatch needs `num_experts_per_rank` at buffer construction");
        const int num_experts = topo.num_scaleout_ranks * topo.num_scaleup_ranks * num_experts_per_rank;
        const auto num_local_experts = num_experts_per_rank;
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

        // Capacities of the handle tensors: the maximum world, every expert local, the whole launch grid
        const int max_num_ranks = max_num_scaleout_ranks * max_num_scaleup_ranks;
        const int num_max_channels = topology::get_num_launch_sms() * kNumMaxChannelsPerSM;

        // Check SF stuffs
        int num_sf_packs = 0;
        void* sf_ptr = nullptr;
        int sf_token_stride = 0, sf_hidden_stride = 0;
        if (sf.has_value()) {
            // SF must be FP32 or packed UE8M0x4
            const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());
            EP_HOST_ASSERT(num_tokens == num_tokens_);
            EP_HOST_ASSERT(sf->is_cuda());
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
            EP_HOST_ASSERT((not nvfp4 or num_sf_packs_ == hidden / 64) and "NVFP4 scales must be [tokens, hidden / 64] int32 packs (an FP8 payload is float8_e4m3fn, not uint8)");
            num_sf_packs = num_sf_packs_;
            sf_ptr = sf->data_ptr();
            sf_token_stride = sf->stride(0);
            sf_hidden_stride = sf->stride(1);
        }

        // Check top-k stuffs
        const auto [num_tokens_, num_topk] = get_shape<2>(topk_idx);
        EP_HOST_ASSERT(num_tokens == num_tokens_);
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());

        // Weights are optional for training backward
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            const auto [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());
            EP_HOST_ASSERT(num_tokens == num_tokens__);
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // Expert receiving counter
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            const auto [num_local_experts_] = get_shape<1>(cumulative_local_expert_recv_stats.value());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                           cumulative_local_expert_recv_stats->is_contiguous());
            EP_HOST_ASSERT(num_local_experts_ >= num_local_experts and
                           "`cumulative_local_expert_recv_stats` must hold the rank's expert slots");
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        if (zero_copy) {
            EP_HOST_ASSERT(num_hidden_bytes == zc.num_hidden_bytes and num_sf_packs == zc.num_sf_packs and num_topk == zc.num_topk and num_experts_per_rank == zc.num_experts_per_rank and "The dispatch does not match the zero-copy arena's wire (bytes per row, scale packs) or shape");
            EP_HOST_ASSERT(expert_alignment == 1 and not do_expand and not use_tma_aligned_col_major_sf and "Zero copy is compact with row-major scaling factors");
            EP_HOST_ASSERT(num_max_tokens_per_rank <= zc.num_max_tokens_per_rank and "The call's `num_max_tokens_per_rank` exceeds the zero-copy arena's");
            EP_HOST_ASSERT(num_tokens + 1 < (1 << layout::WorkspaceLayout::kNumZcCountBits) and "The count allgather's word holds `count + 1` in 20 bits");
        }

        // Under stream capture the kernel does not run, so the host spin of a CPU sync would end in the CPU timeout: refuse it
        if (do_cpu_sync)
            EP_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None and
                           "`dispatch(do_cpu_sync=True)` inside a CUDA-graph capture: capture without the CPU sync");

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);

        // Handle tensors: capacity storage (the maximum world, the whole launch grid) handed back as live-shaped views;
        // the expert-indexed ones have the fixed slot count of this rank in every world
        const auto int_options = at::TensorOptions(torch::kCUDA).dtype(torch::kInt);
        const auto alloc_flat = [&](const int64_t& capacity, const int64_t& live_numel, const std::vector<int64_t>& live_shape) {
            EP_HOST_ASSERT(live_numel <= capacity);
            return torch::empty({capacity}, int_options).slice(0, 0, live_numel).view(live_shape);
        };

        // The number of received tokens per expert
        // This is useful for expanding mode
        // NOTES: for expand mode, the input is exclusive prefix sum, while for non-expand, it is inclusive
        auto psum_num_recv_tokens_per_expert = alloc_flat(num_local_experts + 1, num_local_experts + 1, {num_local_experts + 1});

        // The unaligned (actual) number of received tokens per expert
        // Written by the dispatch kernel's notify warps, used by the epilogue for zero padding
        auto num_unaligned_recv_tokens_per_expert = alloc_flat(num_local_experts, num_local_experts, {num_local_experts});
        int* num_unaligned_recv_tokens_per_expert_ptr = num_unaligned_recv_tokens_per_expert.data_ptr<int>();

        // The prefix sum tensor of number of received tokens from each rank
        // Will also be used in combine as the dispatch handle
        // `[max_num_scaleup_ranks + 1]`: the total goes into the fixed last slot; zeroed on the communication stream, which the kernel follows
        auto psum_num_recv_tokens_per_scaleup_rank = torch::empty({max_num_scaleup_ranks + 1}, int_options);
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(psum_num_recv_tokens_per_scaleup_rank.data_ptr(), 0,
                                           psum_num_recv_tokens_per_scaleup_rank.nbytes(), comm_stream));

        // Decide number of channels by shared memory consumption
        const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
        // The block shape is part of the captured launch, so the channel count per SM is fixed once
        const int num_channels_per_sm = get_num_channels_per_sm(
            hidden, num_hidden_bytes, num_sf_packs, num_topk, max_num_ranks * num_experts_per_rank,
            max_num_ranks, max_num_scaleout_ranks, num_smem_bytes, prefer_overlap_with_compute);
        if (not adaptive_num_channels_per_sm.has_value()) {
            adaptive_num_channels_per_sm = num_channels_per_sm;
            if (get_env<int>("NIXL_EP_BUFFER_DEBUG"))
                printf("Elastic buffer uses %d channels per SM\n", num_channels_per_sm);
        }
        EP_HOST_ASSERT(*adaptive_num_channels_per_sm == num_channels_per_sm and
                       "The adaptive kernels fix the channel count per SM at the first dispatch; "
                       "changing hidden/top-k/SF settings afterwards is not supported");
        // The kernel is launched with `get_num_launch_sms()` blocks; `num_sms` only bounds the active count
        EP_HOST_ASSERT(num_sms >= topology_host.num_active_sms and
                       "`num_sms` is below the active SM count of the topology descriptor, call `set_topology` first");
        const int num_channels = topology_host.num_active_sms * num_channels_per_sm;

        // A domain's count row (its expert slots, rank counts, per-channel record counts) must fit the table's row stride
        // under every world: the widest domain of the declared maximum world
        EP_HOST_ASSERT(max_num_scaleup_ranks * num_experts_per_rank + max_num_scaleup_ranks + num_max_channels <= layout::WorkspaceLayout::kNumMaxExperts and
                       "Too many expert slots for the notify row (see `WorkspaceLayout::kNumMaxExperts`)");

        // Handles: the forwarding metadata and the linked list are allocated in both modes (the direct mode
        // leaves them unused), so that a captured graph can switch modes
        // Capacities: `num_channels * num_max_tokens_per_channel <= num_max_tokens_per_rank + num_channels` at the whole launch grid
        const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
        const int64_t max_channel_tokens = static_cast<int64_t>(num_max_tokens_per_rank) + num_max_channels;
        const int64_t max_linked_list_len = max_num_scaleout_ranks * max_channel_tokens + num_max_channels;

        // The token metadata during forward
        // `[i, j]` means: in channel i, the j-th forwarded token's metadata
        // Info contains:
        //   - Scaleout rank index and source token index in the original rank (0)
        //   - Whether the token is the last one in the chunk (1)
        //   - the top-k scale-up peer indices (top-k)
        //   - each selections' destination slot indices (top-k)
        const auto num_max_forwarded_tokens = topo.num_scaleout_ranks * num_max_tokens_per_channel + 1;
        const auto num_forward_metadata_dims = 2 + num_topk * 2;
        auto token_metadata_at_forward = alloc_flat(
            max_linked_list_len * num_forward_metadata_dims,
            static_cast<int64_t>(num_channels) * num_max_forwarded_tokens * num_forward_metadata_dims,
            {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims});
        int* token_metadata_at_forward_ptr = token_metadata_at_forward.data_ptr<int>();

        // Per-scaleup-peer-per-channel linked list
        // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's index in the combine's input
        auto channel_linked_list = alloc_flat(
            // Index 0 of the list means the starting item
            max_linked_list_len * max_num_scaleup_ranks,
            static_cast<int64_t>(num_channels) * num_max_forwarded_tokens * topo.num_scaleup_ranks,
            {num_channels, num_max_forwarded_tokens, topo.num_scaleup_ranks});
        int* channel_linked_list_ptr = channel_linked_list.data_ptr<int>();

        // Clone `topk_idx` for saving in the handle (to prevent users' modification)
        auto copied_topk_idx = std::optional<torch::Tensor>();
        topk_idx_t* copied_topk_idx_ptr = nullptr;
        if (do_handle_copy) {
            copied_topk_idx = torch::empty_like(topk_idx);
            copied_topk_idx_ptr = copied_topk_idx->data_ptr<topk_idx_t>();
        }

        // Check buffer size
        const auto [num_shared_bytes, num_dispatch_send_bytes, num_combine_send_bytes] = get_buffer_arena(
            num_max_tokens_per_rank, hidden, num_topk, allow_multiple_reduction, max_num_scaleout_ranks, max_num_scaleup_ranks);
        EP_HOST_ASSERT(num_shared_bytes + num_dispatch_send_bytes + num_combine_send_bytes <= num_buffer_bytes);
        const auto [num_wire_recv_bytes, num_wire_send_bytes] = get_dispatch_buffer_size(
                       num_max_tokens_per_rank, num_hidden_bytes, num_sf_packs, num_topk,
                       max_num_scaleout_ranks, max_num_scaleup_ranks);
        EP_HOST_ASSERT(num_wire_recv_bytes <= num_buffer_bytes - num_dispatch_send_bytes - num_combine_send_bytes and
                       num_wire_send_bytes <= num_dispatch_send_bytes);

        // Ready and clean host workspace for this round
        const auto host_workspace_layout = layout::WorkspaceLayout(
            host_workspace,
            topo.num_scaleout_ranks,
            topo.num_scaleup_ranks,
            num_experts);
        std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(), topo.num_scaleup_ranks, 0);
        std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(), num_local_experts, 0);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        uint8_t* zc_slot_base = nullptr;
        void* zc_recv_x_ptr = nullptr;
        void* zc_recv_sf_ptr = nullptr;
        topk_idx_t* zc_recv_topk_idx_ptr = nullptr;
        float* zc_recv_topk_weights_ptr = nullptr;
        int* zc_recv_src_metadata_ptr = nullptr;
        int* zc_count_matrix_ptr = nullptr;
        if (zero_copy) {
            zc_slot_base = zc.base + static_cast<int64_t>(zero_copy_slot) * zc.slot_bytes;
            zc_recv_x_ptr = zc_slot_base + zc.recv_x_offset;
            if (sf.has_value())
                zc_recv_sf_ptr = zc_slot_base + zc.recv_sf_offset;
            zc_recv_topk_idx_ptr = reinterpret_cast<topk_idx_t*>(zc_slot_base + zc.recv_topk_idx_offset);
            if (topk_weights.has_value())
                zc_recv_topk_weights_ptr = reinterpret_cast<float*>(zc_slot_base + zc.recv_topk_weights_offset);
            zc_recv_src_metadata_ptr = reinterpret_cast<int*>(zc_slot_base + zc.recv_src_metadata_offset);
            zc_count_matrix_ptr = reinterpret_cast<int*>(zc.base + zc.count_matrix_offset);
        }

        // Do dispatch into the buffers (with SM limitation)
        EP_HOST_ASSERT(num_sms <= jit::device_runtime->get_num_sms());
        launch_dispatch(x.data_ptr(), sf_ptr,
                        topk_idx.data_ptr<topk_idx_t>(), topk_weights_ptr,
                        copied_topk_idx_ptr,
                        cumulative_local_expert_recv_stats_ptr,
                        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                        psum_num_recv_tokens_per_expert.data_ptr<int>(),
                        num_unaligned_recv_tokens_per_expert_ptr,
                        token_metadata_at_forward_ptr,
                        num_tokens, num_max_tokens_per_rank,
                        num_hidden_bytes,
                        num_sf_packs, sf_token_stride, sf_hidden_stride,
                        num_experts_per_rank, num_topk, expert_alignment,
                        buffer, math::advance_ptr(buffer, num_buffer_bytes - num_dispatch_send_bytes - num_combine_send_bytes),
                        workspace, mapped_host_workspace,
                        get_topology_ptr(),
                        max_num_scaleout_ranks, max_num_scaleup_ranks,
                        num_channels_per_sm,
                        num_smem_bytes,
                        num_gpu_timeout_cycles,
                        do_cpu_sync,
                        zero_copy,
                        zc_recv_x_ptr, zc_recv_sf_ptr, zc_recv_topk_idx_ptr, zc_recv_topk_weights_ptr,
                        zc_recv_src_metadata_ptr,
                        zero_copy ? zc.sf_token_stride : 0, zero_copy ? zc.sf_hidden_stride : 0,
                        zc_count_matrix_ptr, zero_copy ? zc.count_row_stride : 0, zero_copy ? zc.count_matrix_rows : 0,
                        zero_copy ? zero_copy_slot : 0,
                        resolve_dynamic_take(),
                        acquire_flags,
                        comm_stream);

        // Received token counters
        int num_recv_tokens = 0, num_expanded_tokens = 0;
        int counter_scaleup_rank_idx = 0, counter_local_expert_idx = 0;
        std::vector<int> num_recv_tokens_per_expert_list;

        // Assign these values according to modes
        if (do_cpu_sync) {
            // With CPU sync, the exact counts
            const auto start_cpu_time = std::chrono::high_resolution_clock::now();
            while (true) {
                bool ready = true;

                // Read number of received tokens from each scaleup rank
                while (counter_scaleup_rank_idx < topo.num_scaleup_ranks and ready) {
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_rank_count_ptr<false>()[counter_scaleup_rank_idx]);
                    if ((ready = math::is_decoded_positive_ready(count))) {
                        num_recv_tokens += count;
                        ++ counter_scaleup_rank_idx;
                    }
                }

                // Read expert counts
                while (counter_local_expert_idx < num_local_experts and ready) {
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_expert_count_ptr<false>()[counter_local_expert_idx]);
                    if ((ready = math::is_decoded_positive_ready(count))) {
                        num_recv_tokens_per_expert_list.push_back(count);
                        num_expanded_tokens += count;
                        ++ counter_local_expert_idx;
                    }
                }

                // Ready and do next steps
                const auto get_buffer_info = [&]() {
                    std::stringstream ss;
                    ss << "CPU side received count (scaleup: " << topo.scaleup_rank_idx << "): ";
                    for (int i = 0; i < topo.num_scaleup_ranks; ++ i)
                        ss << host_workspace_layout.get_scaleup_rank_count_ptr<false>()[i] << " ";
                    ss << "# ";
                    for (int i = 0; i < num_local_experts; ++ i)
                        ss << host_workspace_layout.get_scaleup_expert_count_ptr<false>()[i] << " ";
                    return ss.str();
                };
                if (ready) {
                    if (get_env<int>("NIXL_EP_BUFFER_DEBUG"))
                        printf("%s\n", get_buffer_info().c_str());
                    break;
                }

                // Timeout checks
                const auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - start_cpu_time).count() > num_cpu_timeout_secs)
                    throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());
            }
        } else {
            // Without CPU sync, allocate for the worst case of the declared maximum world: every source token of every
            // rank in every local slot it may select (at most `num_topk`), plus the alignment slack of the fixed local slots
            num_recv_tokens = num_max_tokens_per_rank * max_num_ranks;
            num_expanded_tokens = num_recv_tokens * std::min(num_topk, num_local_experts);
            num_expanded_tokens += (expert_alignment - 1) * num_local_experts;
            num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
        }

        // Allocate received tensors
        // `recv_src_metadata` includes source token indices and buffer slot indices
        // Zero copy: views of the slot (`from_blob`, no deleter: the slot rule protects them), sliced to the count
        const auto num_allocated_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
        torch::Tensor recv_x, recv_src_metadata;
        auto recv_sf = std::optional<torch::Tensor>();
        auto recv_topk_idx = std::optional<torch::Tensor>();
        auto recv_topk_weights = std::optional<torch::Tensor>();

        // Optional tensors
        void* recv_sf_ptr = nullptr;
        topk_idx_t* recv_topk_idx_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
        if (zero_copy) {
            EP_HOST_ASSERT(num_allocated_tokens <= zc.num_rows and num_recv_tokens <= zc.num_compact_rows and "The receive count exceeds the zero-copy slot");
            recv_x = torch::from_blob(zc_recv_x_ptr, {zc.num_rows, static_cast<int64_t>(num_cols)}, x.options())
                .slice(0, 0, num_allocated_tokens);
            recv_src_metadata = torch::from_blob(zc_recv_src_metadata_ptr, {zc.num_compact_rows, static_cast<int64_t>(num_topk + 2)}, int_options)
                .slice(0, 0, num_recv_tokens);
            if (sf.has_value()) {
                recv_sf_token_stride = zc.sf_token_stride, recv_sf_hidden_stride = zc.sf_hidden_stride;
                recv_sf = torch::from_blob(zc_recv_sf_ptr, {zc.num_rows, static_cast<int64_t>(num_sf_packs)},
                                           {static_cast<int64_t>(recv_sf_token_stride), static_cast<int64_t>(recv_sf_hidden_stride)},
                                           sf->options())
                    .slice(0, 0, num_allocated_tokens);
                recv_sf_ptr = zc_recv_sf_ptr;
            }
            recv_topk_idx = torch::from_blob(zc_recv_topk_idx_ptr, {zc.num_compact_rows, static_cast<int64_t>(num_topk)}, topk_idx.options())
                .slice(0, 0, num_allocated_tokens);
            recv_topk_idx_ptr = zc_recv_topk_idx_ptr;
            if (topk_weights.has_value()) {
                recv_topk_weights = torch::from_blob(zc_recv_topk_weights_ptr, {zc.num_compact_rows, static_cast<int64_t>(num_topk)}, topk_weights->options())
                    .slice(0, 0, num_allocated_tokens);
                recv_topk_weights_ptr = zc_recv_topk_weights_ptr;
            }
        } else {
            recv_x = torch::empty({num_allocated_tokens, num_cols}, x.options());
            recv_src_metadata = torch::empty({num_recv_tokens, num_topk + 2},
                                             torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            if (sf.has_value()) {
                if (not use_tma_aligned_col_major_sf) {
                    recv_sf_token_stride = num_sf_packs, recv_sf_hidden_stride = 1;
                } else {
                    // TMA-aligned layout for the next GEMM input
                    recv_sf_token_stride = 1, recv_sf_hidden_stride = math::align(num_allocated_tokens, kNumAlignedSFPacks);
                }
                recv_sf = torch::empty_strided({num_allocated_tokens, num_sf_packs},
                                               {recv_sf_token_stride, recv_sf_hidden_stride},
                                               sf->options());
                recv_sf_ptr = recv_sf->data_ptr();
            }
            if (not do_expand) {
                recv_topk_idx = torch::empty({num_allocated_tokens, num_topk}, topk_idx.options());
                recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
            }
            if (topk_weights.has_value()) {
                recv_topk_weights = do_expand ?
                    torch::empty({num_allocated_tokens}, topk_weights->options()) :
                    torch::empty({num_allocated_tokens, num_topk}, topk_weights->options());
                recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
            }
        }

        // Process prefix sum, in expanding mode, it is also atomic counters
        if (do_expand) {
            // Slice the exclusive part and do atomic additions into inclusive
            psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);
        } else {
            // Slice the inclusive part (and will not be used in the epilogue)
            psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);
        }
        EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.size(0) == num_local_experts);

        // Launch copy kernels with full SMs
        stream_control_before_epilogue(previous_event_before_epilogue);
        // Without a CPU sync the count is the worst case: a negative value tells the epilogue to read the fixed slot
        // Zero copy: launched in both modes (a captured graph keeps it across a mode switch); it writes the slot in hybrid mode and returns at once in direct mode
        launch_dispatch_copy_epilogue(buffer, workspace,
                                      psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                                      psum_num_recv_tokens_per_expert.data_ptr<int>(),
                                      recv_x.data_ptr(), recv_sf_ptr,
                                      recv_topk_idx_ptr, recv_topk_weights_ptr,
                                      recv_src_metadata.data_ptr<int>(),
                                      channel_linked_list_ptr,
                                      num_unaligned_recv_tokens_per_expert_ptr,
                                      do_cpu_sync ? num_recv_tokens : -1, max_num_scaleup_ranks, num_max_tokens_per_rank,
                                      num_hidden_bytes,
                                      num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride,
                                      num_experts_per_rank, num_topk, expert_alignment,
                                      get_topology_ptr(),
                                      jit::device_runtime->get_num_sms(),
                                      jit::device_runtime->get_num_smem_bytes(),
                                      num_channels_per_sm,
                                      do_expand, do_zero_padding, zero_copy,
                                      comm_stream);

        // Stream control
        const auto event = stream_control_epilogue(
            {x, sf, topk_idx, topk_weights,
             recv_x, recv_sf, recv_topk_idx, recv_topk_weights,
             cumulative_local_expert_recv_stats,
             copied_topk_idx,
             psum_num_recv_tokens_per_scaleup_rank,
             psum_num_recv_tokens_per_expert,
             num_unaligned_recv_tokens_per_expert,
             recv_src_metadata,
             token_metadata_at_forward,
             channel_linked_list},
            compute_stream,
            allocate_on_comm_stream, async_with_compute_stream);

        return {recv_x, recv_sf,
                recv_topk_idx, recv_topk_weights,
                copied_topk_idx,
                num_recv_tokens, num_expanded_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_scaleup_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                token_metadata_at_forward,
                channel_linked_list,
                event};
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,
            const torch::Tensor& token_metadata_at_forward,
            const torch::Tensor& channel_linked_list,
            const bool& recv_count_is_exact,
            const int& num_max_tokens_per_rank,
            const int& num_sms,
            const std::optional<EventHandle>& previous_event,
            const std::optional<EventHandle>& previous_event_before_epilogue,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& use_expanded_layout) const {
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0);

        EP_HOST_ASSERT(topology_host.is_participant() and
                       "This rank is outside the live world of the topology descriptor (see `stage_topology`)");
        const auto& topo = topology_host;

        // Check data
        EP_HOST_ASSERT(num_experts_per_rank > 0 and "Combine needs `num_experts_per_rank` at buffer construction");
        const auto [num_tokens, hidden] = get_shape<2>(x);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);

        // Check tensors at dispatch
        const auto [num_combined_tokens, num_topk] = get_shape<2>(combined_topk_idx);
        const auto [num_psum_slots] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
        EP_HOST_ASSERT(combined_topk_idx.is_cuda() and combined_topk_idx.is_contiguous());
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(num_psum_slots == max_num_scaleup_ranks + 1);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);

        // Check metadata
        // For reduction mode, `num_tokens_` means the number of unexpanded tokens
        const auto [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);
        EP_HOST_ASSERT(num_reduced_tokens == (use_expanded_layout ? num_reduced_tokens : num_tokens));
        EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);
        EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);

        // Check optional tensors
        if (topk_weights.has_value()) {
            if (use_expanded_layout) {
                const auto [num_tokens__] = get_shape<1>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__);
            } else {
                const auto [num_tokens__, num_topk__] = get_shape<2>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk__);
            }
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
        }

        const auto bias_opts = std::vector({bias_0, bias_1});
        void* bias_ptrs[2] = {nullptr, nullptr};
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
                bias_ptrs[i] = bias.data_ptr();
            }
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);

        // Check buffer size
        const auto [num_shared_bytes, num_dispatch_send_bytes, num_combine_send_bytes] = get_buffer_arena(
            num_max_tokens_per_rank, hidden, num_topk, allow_multiple_reduction, max_num_scaleout_ranks, max_num_scaleup_ranks);
        EP_HOST_ASSERT(num_shared_bytes + num_dispatch_send_bytes + num_combine_send_bytes <= num_buffer_bytes);

        // The forwarding metadata of the dispatch handle
        // The token metadata during forward
        const auto [num_channels_, d1, d2] = get_shape<3>(token_metadata_at_forward);
        const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels_);
        const int num_channels = num_channels_;
        int* token_metadata_at_forward_ptr = token_metadata_at_forward.data_ptr<int>();
        EP_HOST_ASSERT(d1 == topo.num_scaleout_ranks * num_max_tokens_per_channel + 1);
        EP_HOST_ASSERT(d2 == 2 + num_topk * 2);
        EP_HOST_ASSERT(token_metadata_at_forward.is_cuda() and token_metadata_at_forward.is_contiguous());
        EP_HOST_ASSERT(token_metadata_at_forward.scalar_type() == torch::kInt);

        // Per-scaleup-peer-per-channel linked list
        const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list);
        int* channel_linked_list_ptr = channel_linked_list.data_ptr<int>();
        EP_HOST_ASSERT(num_channels == num_channels__);
        EP_HOST_ASSERT(d1_ == topo.num_scaleout_ranks * num_max_tokens_per_channel + 1);
        EP_HOST_ASSERT(d2_ == topo.num_scaleup_ranks);
        EP_HOST_ASSERT(channel_linked_list.is_cuda() and channel_linked_list.is_contiguous());
        EP_HOST_ASSERT(channel_linked_list.scalar_type() == torch::kInt);

        // Push data into remote buffers
        // NOTES: we don't use `num_hidden_bytes` due to enable later quantization possibility
        EP_HOST_ASSERT(adaptive_num_channels_per_sm.has_value() and "Combine before any dispatch");
        EP_HOST_ASSERT(num_channels == *adaptive_num_channels_per_sm * topology_host.num_active_sms and
                       "The handle was produced with another active SM count");
        launch_combine(
            x.data_ptr(),
            topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
            token_metadata_at_forward_ptr,
            channel_linked_list_ptr,
            buffer, math::advance_ptr(buffer, num_buffer_bytes - num_combine_send_bytes), workspace,
            get_topology_ptr(),
            recv_count_is_exact ? num_reduced_tokens : -1, num_max_tokens_per_rank,
            hidden, num_experts_per_rank, num_topk,
            num_gpu_timeout_cycles,
            max_num_scaleout_ranks, max_num_scaleup_ranks,
            *adaptive_num_channels_per_sm,
            jit::device_runtime->get_num_smem_bytes(),
            use_expanded_layout, allow_multiple_reduction,
            resolve_dynamic_take(),
            comm_stream);
        // Allocate output tensors
        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        auto combined_topk_weights = std::optional<torch::Tensor>();
        float* combined_topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
            combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
        }

        // Combine pushed data: the reduce epilogue derives the rows to reduce from the topology descriptor
        // (`EPTopology`, its `mode` and rank counts) and takes the buffer base
        stream_control_before_epilogue(previous_event_before_epilogue);
        launch_combine_reduce_epilogue(combined_x.data_ptr(),
                                       combined_topk_weights_ptr,
                                       combined_topk_idx.data_ptr<topk_idx_t>(),
                                       num_combined_tokens, num_max_tokens_per_rank,
                                       hidden,
                                       num_experts_per_rank, num_topk,
                                       num_gpu_timeout_cycles,
                                       buffer, workspace,
                                       bias_ptrs[0], bias_ptrs[1],
                                       get_topology_ptr(),
                                       jit::device_runtime->get_num_sms(),
                                       jit::device_runtime->get_num_smem_bytes(),
                                       use_expanded_layout, allow_multiple_reduction,
                                       acquire_flags,
                                       comm_stream);

        // Stream control
        const auto event = stream_control_epilogue(
            {x, topk_weights, bias_0, bias_1,
             src_metadata,
             combined_topk_idx,
             combined_x, combined_topk_weights,
             psum_num_recv_tokens_per_scaleup_rank,
             token_metadata_at_forward,
             channel_linked_list},
            compute_stream,
            allocate_on_comm_stream, async_with_compute_stream);
        return {combined_x, combined_topk_weights, event};
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait);

    pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<int64_t, bool, bool, int, int, bool, int, int, int, int, int, std::string, int, int64_t, int, bool>())
        .def("destroy", &ElasticBuffer::destroy)
        .def("get_comm_stream", &ElasticBuffer::get_comm_stream)
        .def("get_physical_domain_size", &ElasticBuffer::get_physical_domain_size)
        .def("get_logical_domain_size", &ElasticBuffer::get_logical_domain_size)
        .def("stage_topology_masks", &ElasticBuffer::stage_topology_masks)
        .def("commit_topology", &ElasticBuffer::commit_topology)
        .def("set_topology", &ElasticBuffer::set_topology)
        .def("get_topology", &ElasticBuffer::get_topology)
        .def("get_topology_masks", &ElasticBuffer::get_topology_masks)
        .def("get_topology_ranks", &ElasticBuffer::get_topology_ranks)
        .def("is_participant", &ElasticBuffer::is_participant)
        .def("get_local_metadata", &ElasticBuffer::get_local_metadata)
        .def("get_local_peer_info", &ElasticBuffer::get_local_peer_info)
        .def("connect_ranks", &ElasticBuffer::connect_ranks)
        .def("disconnect_ranks", &ElasticBuffer::disconnect_ranks)
        .def("get_connected_ranks", &ElasticBuffer::get_connected_ranks)
        .def("read_profile", &ElasticBuffer::read_profile)
        .def("barrier", &ElasticBuffer::barrier)
        .def("configure_zero_copy_arena", &ElasticBuffer::configure_zero_copy_arena)
        .def("has_zero_copy", &ElasticBuffer::has_zero_copy)
        .def("get_zero_copy_num_slots", &ElasticBuffer::get_zero_copy_num_slots)
        .def("dispatch", &ElasticBuffer::dispatch)
        .def("combine", &ElasticBuffer::combine);
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);
    m.def("calculate_zero_copy_arena_size", &ElasticBuffer::calculate_zero_copy_arena_size);
    m.def("get_elastic_buffer_alignment", [=]() {
        return ElasticBuffer::kNumAlignmentBytes;
    });
}

}  // namespace nixl_ep_v2
