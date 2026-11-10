// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/topology.cuh>

#include "../kernels/backend/nixl.hpp"
#include "../kernels/elastic/api.hpp"
#include "../utils/event.hpp"
#include "../utils/format.hpp"
#include "utils.hpp"

namespace nixl_ep_v2::elastic {

using namespace adaptive;
class ElasticBuffer {
    // Buffer bytes (excludes workspace)
    // Memory layout: [[Workspace] GPU buffer]
    int64_t num_buffer_bytes;
    void* buffer;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // Workspace
    // Control words reset at rest except combine readiness; barrier counters retain their epochs.
    void *workspace;
    void *host_workspace, *mapped_host_workspace;
    std::shared_ptr<layout::WorkspaceLayout> workspace_layout;
    torch::Tensor profile_storage;

    // CUDA streams
    at::cuda::CUDAStream comm_stream;

    // Whether to allow multiple reductions
    bool allow_multiple_reduction;

    // Fixed dispatch capacity; both operations derive their source arena offsets from it.
    const bool use_fp8_dispatch;

    // Whether to prefer overlapping communication with compute
    bool prefer_overlap_with_compute;

    // Timeout settings
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;

    // The transport: this rank's NIXL agent, its registered region (workspace + GPU buffer) and the peers' views
    std::shared_ptr<nixl::NIXLEPContext> nixl_context;

    // The declared maximum world and this rank's coordinates in it; the staged topology descriptor (`EPTopology`) awaits `commit_topology`
    int max_num_scaleout_ranks = 0, max_num_scaleup_ranks = 0, max_num_ranks = 0;
    int scaleout_rank_idx = 0, scaleup_rank_idx = 0;
    // Physical routing slots per rank; independent of the model's logical expert count.
    int num_experts_per_rank = 0;
    // Adaptive kernels: device-side topology descriptor (+ host mirror)
    torch::Tensor topology_tensor;
    topology::EPTopology topology_host{}, staged_topology{};
    bool topology_staged = false;

public:
    // The 2 MB unit of the workspace, the GPU buffer and `calculate_buffer_size`
    static constexpr int64_t kNumAlignmentBytes = 2097152;

    ElasticBuffer(const int64_t& num_buffer_bytes,
                  const bool& allow_multiple_reduction,
                  const bool& use_fp8_dispatch,
                  const bool& prefer_overlap_with_compute,
                  const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
                  const bool& explicitly_destroy,
                  const int& max_num_scaleout_ranks, const int& max_num_scaleup_ranks,
                  const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                  const int& num_nixl_channels, const std::string& agent_name_prefix,
                  const int& num_experts_per_rank):
        num_buffer_bytes(num_buffer_bytes),
        explicitly_destroy(explicitly_destroy),
        comm_stream(get_global_comm_stream()),
        allow_multiple_reduction(allow_multiple_reduction),
        use_fp8_dispatch(use_fp8_dispatch),
        prefer_overlap_with_compute(prefer_overlap_with_compute),
        max_num_scaleout_ranks(max_num_scaleout_ranks), max_num_scaleup_ranks(max_num_scaleup_ranks), max_num_ranks(max_num_scaleout_ranks * max_num_scaleup_ranks),
        scaleout_rank_idx(scaleout_rank_idx), scaleup_rank_idx(scaleup_rank_idx), num_experts_per_rank(num_experts_per_rank) {
        // Check buffer bytes alignment (2 MB)
        EP_HOST_ASSERT(num_buffer_bytes > 0 and num_buffer_bytes % kNumAlignmentBytes == 0);

        // Workspace is aligned to 2 MB so that it sits cleanly at the front of the symmetric memory
        const auto num_workspace_bytes = math::align<int64_t>(
            layout::WorkspaceLayout::get_num_bytes(), kNumAlignmentBytes);

        // The declared maximum world and this rank's coordinates in it. Every rank of a world declares the same maximum world:
        // the world-rank numbering (`topology::world_rank`) and the handle shapes depend on it, and no exchange compares it
        // (`connect_ranks` checks a peer's world-rank range and region size only). The dispatch capacity flag must also match,
        // including when num_buffer_bytes is supplied explicitly, because it determines the source arena offsets.
        EP_HOST_ASSERT(0 < max_num_scaleout_ranks and max_num_scaleout_ranks <= topology::kNumMaxScaleoutRanksCap and
                       0 < max_num_scaleup_ranks and max_num_scaleup_ranks <= topology::kNumMaxScaleupRanksCap and "Bad declared maximum world");
        EP_HOST_ASSERT(0 <= scaleout_rank_idx and scaleout_rank_idx < max_num_scaleout_ranks and
                       0 <= scaleup_rank_idx and scaleup_rank_idx < max_num_scaleup_ranks and
                       "This rank's coordinates are outside the declared maximum world");

        // All declared slots must fit, including currently inactive ranks. Zero is for barrier-only buffers.
        EP_HOST_ASSERT(0 <= num_experts_per_rank and
                       "Invalid number of physical expert slots per rank");
        EP_HOST_ASSERT(static_cast<int64_t>(max_num_ranks) * num_experts_per_rank <= layout::WorkspaceLayout::kNumMaxExperts and
                       "The declared maximum world's physical expert slots exceed workspace capacity");

        // Create the transport: one symmetric region per rank, registered with this rank's agent
        // Symmetric memory layout: [[Workspace] GPU buffer]
        const int world_rank = topology::world_rank(scaleout_rank_idx, scaleup_rank_idx, max_num_scaleup_ranks);
        this->nixl_context = std::make_shared<nixl::NIXLEPContext>(
            agent_name_prefix + "_" + std::to_string(world_rank), world_rank, max_num_ranks,
            max_num_scaleup_ranks, num_workspace_bytes + num_buffer_bytes, num_nixl_channels);
        workspace = this->nixl_context->region_base();

        // Topology descriptor (`EPTopology`); its `mode` follows the active domains: the Python layer derives it for every world it stages
        topology_tensor = torch::empty({static_cast<int64_t>(sizeof(topology::EPTopology))},
                                       torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

        // Timeout
        this->num_cpu_timeout_secs = num_cpu_timeout_secs;
        this->num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs);
        this->num_gpu_timeout_cycles *= jit::device_runtime->get_clock_rate();
        // Compile during setup so the first barrier, including destruction, never triggers compilation.
        build_barrier(max_num_scaleup_ranks, num_gpu_timeout_cycles);

        // Assign workspaces and buffers
        workspace_layout = std::make_shared<layout::WorkspaceLayout>(workspace);
        buffer = static_cast<uint8_t*>(workspace) + num_workspace_bytes;
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, num_workspace_bytes));

        if (jit::Compiler::profiling_enabled()) {
            const int num_blocks = std::max(topology::get_num_launch_sms(), jit::device_runtime->get_num_sms());
            const int num_warps = jit::device_runtime->get_max_warps_per_block();
            profile_storage = torch::empty({profile::kNumPhases, num_blocks, num_warps},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            CUDA_RUNTIME_CHECK(cudaMemset(profile_storage.data_ptr(), 0, profile_storage.nbytes()));
            const profile::Header profile_header{profile_storage.data_ptr<int64_t>(), num_blocks, num_warps};
            CUDA_RUNTIME_CHECK(cudaMemcpy(workspace_layout->get_profile_header(), &profile_header,
                                          sizeof(profile_header), cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        }

        // The initial topology descriptor (`EPTopology`): this rank alone (one active domain: direct, as the rule says; nothing
        // launches on it), every launched block active
        topology::ScaleupMask self = {};
        self.words[scaleup_rank_idx / 64] = 1ull << (scaleup_rank_idx % 64);
        stage_topology(topology::kModeDirect, 1u << scaleout_rank_idx, self, topology::get_num_launch_sms());
        // No live topology exists yet: an upload failure unwinds the constructor's owned resources.
        publish_topology(at::cuda::getCurrentCUDAStream());

        // Allocate host workspaces
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
        std::memset(host_workspace, 0, layout::WorkspaceLayout::get_num_bytes());

        // No barrier here: the Python constructor connects the cohort (`connect_ranks`), commits its world (`commit_topology`) and ends with the bootstrap Store barrier
    }

    ~ElasticBuffer() noexcept(false) {
        if (not explicitly_destroy)
            destroy();

        if (not destroyed) {
            printf("`destroy()` is not called before NIXL EP v2 elastic buffer destruction, which can leak resources.\n");
            fflush(stdout);
        }
    }

    void destroy() {
        EP_HOST_ASSERT(not destroyed);

        // Finish all works on all GPUs
        barrier(true, true);

        // Deallocate host workspaces
        CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));
        profile_storage = torch::Tensor();

        // Destroy the transport
        nixl_context->finalize();

        // Cannot use anymore
        destroyed = true;
    }

    torch::Stream get_comm_stream() const {
        return comm_stream;
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
        stage_topology(mode, scaleout_mask, scaleup_mask, num_active_sms);
    }

private:
    // Validate the world through a local descriptor, then hold it for `commit_topology` (a refused world leaves the staged one untouched)
    void stage_topology(const int& mode, const uint32_t& scaleout_mask, const topology::ScaleupMask& scaleup_mask,
                        const int& num_active_sms) {
        EP_HOST_ASSERT(mode == topology::kModeDirect or mode == topology::kModeHybrid);
        EP_HOST_ASSERT(scaleout_mask != 0 and scaleup_mask.popc() > 0 and "No rank is active");
        const int num_scaleout_ranks = 32 - __builtin_clz(scaleout_mask);
        const int num_scaleup_ranks = scaleup_mask.extent();
        EP_HOST_ASSERT(num_scaleout_ranks <= max_num_scaleout_ranks and num_scaleup_ranks <= max_num_scaleup_ranks and
                       "The world exceeds the declared maximum (`max_num_scaleout_ranks` / `max_num_scaleup_ranks`)");
        topology::EPTopology next{};
        next.mode = static_cast<topology::Mode>(mode);
        next.num_scaleout_ranks = num_scaleout_ranks;
        next.num_scaleup_ranks = num_scaleup_ranks;
        next.scaleout_rank_idx = scaleout_rank_idx;
        next.scaleup_rank_idx = scaleup_rank_idx;
        next.num_active_sms = num_active_sms;
        next.scaleout_mask = scaleout_mask;
        next.scaleup_mask = scaleup_mask;
        next.transport = nixl_context->get_device_context_ptr();
        EP_HOST_ASSERT((mode == topology::kModeHybrid or next.num_active_scaleout_ranks() == 1) and
                       "Direct mode needs exactly one active NVLink domain");
        EP_HOST_ASSERT(2 <= num_active_sms and num_active_sms <= topology::get_num_launch_sms() and num_active_sms % 2 == 0 and
                       "`num_active_sms` must be even, between 2 and the launch grid (`get_adaptive_num_launch_sms()`, NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS)");

        // An active rank's kernels address every active peer (a rank that is not active launches nothing and may stage anything)
        if (next.is_active())
            for (int d = 0; d < next.num_scaleout_ranks; ++ d) for (int k = 0; k < next.num_scaleup_ranks; ++ k) {
                const int wr = topology::world_rank(d, k, max_num_scaleup_ranks);
                if (next.is_active(d, k) and not nixl_context->is_connected(wr))
                    EP_HOST_UNREACHABLE(fmt::format("Rank {} stages a world with active world rank {} (domain {}, position {}) that is not connected: call `connect_ranks` first", nixl_context->world_rank, wr, d, k));
            }
        staged_topology = next;
        topology_staged = true;
    }

    void publish_topology(const at::cuda::CUDAStream& stream) {
        // One stream-ordered write of the staged topology descriptor (`EPTopology`) on the current stream, which the
        // communication stream then waits for (a pageable source: CUDA takes the bytes now and refuses it under capture)
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(topology_tensor.data_ptr(), &staged_topology, sizeof(topology::EPTopology),
                                           cudaMemcpyHostToDevice, stream));
        stream_wait(comm_stream, stream);

        topology_host = staged_topology;
        topology_staged = false;
    }

public:
    void commit_topology() {
        EP_HOST_ASSERT((topology_staged or nixl_context->has_staged_views()) and
                       "Nothing staged: call `stage_topology`, `connect_ranks` or `disconnect_ranks` first");
        EP_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None and
                       "`commit_topology` inside a CUDA-graph capture: commit between launches");
        // Peers changed but the world did not (e.g. dropping the peers a previous commit excluded): recommit it
        if (not topology_staged)
            staged_topology = topology_host;
        const auto stream = at::cuda::getCurrentCUDAStream();

        // Peers changed since the last commit: swap the transport's views in first (a device sync inside)
        if (nixl_context->has_staged_views()) {
            nixl_context->commit(stream);
        }

        // The descriptor may already have changed if the stream dependency fails; do not expose a stale host mirror.
        nixl::mutate_transport_or_abort(nixl_context->world_rank, "topology commit", [&] { publish_topology(stream); });
    }

    std::tuple<int, int> get_topology() const {
        return {static_cast<int>(topology_host.mode), topology_host.num_active_sms};
    }

    const topology::EPTopology* get_topology_ptr() const {
        return static_cast<const topology::EPTopology*>(topology_tensor.data_ptr());
    }

    // The masks of the active domains and positions of the committed topology descriptor (`EPTopology`), the position mask as its 64-bit words
    std::tuple<uint32_t, std::vector<uint64_t>> get_topology_masks() const {
        std::vector<uint64_t> words(topology_host.scaleup_mask.words, topology_host.scaleup_mask.words + topology::kNumScaleupMaskWords);
        return {topology_host.scaleout_mask, words};
    }

    // Whether this rank is active in the committed world (a rank that is not active must not call `dispatch`/`combine`)
    bool is_active() const {
        return topology_host.is_active();
    }

    // Whether a world is staged and not committed (`set_num_active_sms` refuses to overwrite it)
    bool has_staged_topology() const {
        return topology_staged;
    }

private:
    // The committed topology descriptor of an active rank: `dispatch` / `combine` refuse a rank outside the committed world
    const topology::EPTopology& active_rank_topology() const {
        topology_host.check_launch();
        return topology_host;
    }

public:
    // ---- Peers ----
    // Join: publish the metadata and peer info, `connect_ranks`, then stage and commit. Leave: commit a world without the rank, then `disconnect_ranks`
    pybind11::bytes get_local_metadata() const {
        return pybind11::bytes(nixl_context->get_local_metadata());
    }

    // `(region base, device, world rank, allocation bytes)`
    using peer_info_t = std::tuple<uint64_t, int, int, uint64_t>;

    peer_info_t get_local_peer_info() const {
        const auto info = nixl_context->get_local_peer_info();
        return {info.region_base, info.device_id, info.world_rank, info.num_region_bytes};
    }

    void connect_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_mds,
                       const std::vector<peer_info_t>& peer_infos) {
        EP_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None and
                       "`connect_ranks` inside a CUDA-graph capture: prepare peers outside capture");
        std::vector<nixl::NIXLPeerInfo> infos;
        for (const auto& [base, device, world_rank, num_bytes]: peer_infos)
            infos.push_back({base, device, world_rank, num_bytes});
        // Native preparation may wait for joining peers while another Python thread replays the committed graph.
        pybind11::gil_scoped_release release;
        // A connected pair counts its barriers from zero on both sides (the peer runs this for us too): zero our words for
        // the world rank now, before our hello goes out, since the peer can write to us only once it has our hello and committed
        const auto stream = at::cuda::getStreamFromPool();
        const auto& wl = *workspace_layout;
        for (const int wr: world_ranks) {
            if (nixl_context->is_connected(wr))
                EP_HOST_UNREACHABLE(fmt::format("Rank {}: world rank {} is this rank or already connected", nixl_context->world_rank, wr));
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(wl.get_peer_barrier_signal_ptr(wr), 0, sizeof(int64_t), stream));
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(wl.get_peer_barrier_expected_ptr(wr), 0, sizeof(int64_t), stream));
        }
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        nixl_context->connect_ranks(world_ranks, remote_mds, infos);
    }

    void disconnect_ranks(const std::vector<int>& world_ranks) {
        // An active rank's kernels address every active peer: commit a world without the rank first (a rank that is not active may drop anyone);
        // a staged world that holds the rank would reach the kernels with an unconnected active peer at its commit: stage one without it first
        if (topology_host.is_active()) {
            for (const int wr: world_ranks) {
                if (topology_host.is_active(wr / max_num_scaleup_ranks, wr % max_num_scaleup_ranks))
                    EP_HOST_UNREACHABLE(fmt::format("Rank {} disconnects world rank {}, which is active: commit a world without it first", nixl_context->world_rank, wr));
            }
        }
        if (topology_staged and staged_topology.is_active()) {
            for (const int wr: world_ranks) {
                if (staged_topology.is_active(wr / max_num_scaleup_ranks, wr % max_num_scaleup_ranks))
                    EP_HOST_UNREACHABLE(fmt::format("Rank {} disconnects world rank {}, which is active in the staged world: stage a world without it first", nixl_context->world_rank, wr));
            }
        }
        nixl_context->disconnect_ranks(world_ranks);
    }

    std::vector<int> get_connected_ranks() const {
        return nixl_context->get_connected_ranks();
    }

    // One isolated pair or graph replay between reads; synchronization and reset are diagnostic-only.
    std::tuple<torch::Tensor, std::vector<std::string>> read_profile() const {
        EP_HOST_ASSERT(not destroyed and "`read_profile` after buffer destruction: create a new buffer");
        EP_HOST_ASSERT(profile_storage.defined() and
                       "Profiling is disabled: set NIXL_EP_ADAPTIVE_PROFILE=1 before importing NIXL EP v2 and creating the buffer");
        EP_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None and
                       "`read_profile` inside CUDA-graph capture: read after the captured graph has replayed");
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        auto values = torch::empty(profile_storage.sizes(), torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        CUDA_RUNTIME_CHECK(cudaMemcpy(values.data_ptr(), profile_storage.data_ptr(), profile_storage.nbytes(), cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(cudaMemset(profile_storage.data_ptr(), 0, profile_storage.nbytes()));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        return {values, std::vector<std::string>(profile::kPhaseNames, profile::kPhaseNames + profile::kNumPhases)};
    }

    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync) const {
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto stream = use_comm_stream ? comm_stream : compute_stream;
        if (use_comm_stream)
            stream_wait(comm_stream, compute_stream);

        // Wait all streams to finish on this GPU
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Launch GPU barrier
        launch_barrier(get_topology_ptr(), max_num_scaleup_ranks, num_gpu_timeout_cycles, stream);

        // Let CPU wait
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Compute stream should also wait for the barrier
        if (use_comm_stream)
            stream_wait(compute_stream, comm_stream);
    }

    torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                    const bool& allocate_on_comm_stream) const {
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
            // Keep communication-stream allocations alive for subsequent compute-stream consumers.
            if (allocate_on_comm_stream)
                for (auto& t: tensors) if (t.has_value())
                    t->record_stream(compute_stream);
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
        const auto token_layout = get_combine_token_layout(hidden);

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

    static int get_max_num_fp8_sf_packs(const int& hidden) {
        return math::ceil_div(hidden, 32);
    }

    // How the token buffer is carved: the dispatch and the combine overlay their RECEIVE regions on `[0, num_shared_bytes)`
    // (sized for the larger of the two); the scale-out SEND buffers of the two kernels sit at the END of the GPU buffer, each
    // in a region of its own, because the NIC keeps reading the sources of RDMA puts after the kernel that issued them has ended
    static std::tuple<int64_t, int64_t, int64_t> get_buffer_arena_sizes(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                                                        const bool& allow_multiple_reduction, const bool& use_fp8_dispatch,
                                                                        const int& num_scaleout_ranks, const int& num_scaleup_ranks) {
        const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
        const auto num_sf_packs = use_fp8_dispatch ? get_max_num_fp8_sf_packs(hidden) : 0;
        const auto [num_dispatch_recv_bytes, num_dispatch_send_bytes] = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden * elem_size, num_sf_packs, num_topk, num_scaleout_ranks, num_scaleup_ranks);
        const auto [num_combine_recv_bytes, num_combine_send_bytes] = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk, num_scaleout_ranks, num_scaleup_ranks, allow_multiple_reduction);
        return {std::max(num_dispatch_recv_bytes, num_combine_recv_bytes), num_dispatch_send_bytes, num_combine_send_bytes};
    }

    // The two send buffers' offsets inside `buffer` (the carve-up above at the declared maximum world)
    struct BufferArena {
        int64_t dispatch_send_offset, combine_send_offset;
    };

    BufferArena get_buffer_arena(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk) const {
        const auto [num_shared_bytes, num_dispatch_send_bytes, num_combine_send_bytes] = get_buffer_arena_sizes(
            num_max_tokens_per_rank, hidden, num_topk, allow_multiple_reduction, use_fp8_dispatch, max_num_scaleout_ranks, max_num_scaleup_ranks);
        EP_HOST_ASSERT(num_shared_bytes + num_dispatch_send_bytes + num_combine_send_bytes <= num_buffer_bytes);
        return {num_buffer_bytes - num_dispatch_send_bytes - num_combine_send_bytes, num_buffer_bytes - num_combine_send_bytes};
    }

    // Buffer size for a declared maximum world (the committed extent's sub-layouts are prefixes of the maximum ones)
    static int64_t calculate_buffer_size(const int& max_num_scaleout_ranks, const int& max_num_scaleup_ranks,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk,
                                         const bool& use_fp8_dispatch,
                                         const bool& allow_multiple_reduction) {
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

        // The worst case SF bytes must be less than the main part
        EP_HOST_ASSERT(get_max_num_fp8_sf_packs(hidden) * sizeof(sf_pack_t) <= hidden);

        // Reserve for the maximum supported top-k.
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Shared receive regions plus the two send buffers (see `get_buffer_arena_sizes`), aligned to 2 MB
        const auto [num_shared_bytes, num_dispatch_send_bytes, num_combine_send_bytes] = get_buffer_arena_sizes(
            num_max_tokens_per_rank, hidden, num_topk, allow_multiple_reduction, use_fp8_dispatch, max_num_scaleout_ranks, max_num_scaleup_ranks);
        return math::align(num_shared_bytes + num_dispatch_send_bytes + num_combine_send_bytes, kNumAlignmentBytes);
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               int,
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
             const std::optional<EventHandle>& previous_event,
             const std::optional<EventHandle>& previous_event_before_epilogue,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_handle_copy, const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf) const {
        // Zero padding only makes sense with expand mode
        EP_HOST_ASSERT(not do_zero_padding or do_expand);

        const auto& topo = active_rank_topology();

        // Check data tensor
        const auto [num_tokens, num_cols] = get_shape<2>(x);
        const bool nvfp4 = x.scalar_type() == torch::kByte;
        const int hidden = nvfp4 ? 2 * num_cols : num_cols;
        const auto num_hidden_bytes = num_cols * static_cast<int>(x.element_size());
        EP_HOST_ASSERT((not nvfp4 or (sf.has_value() and hidden % 64 == 0)) and
                       "NVFP4 requires packed uint8 codes and block scales, with logical hidden divisible by 64");
        EP_HOST_ASSERT(not (nvfp4 and use_tma_aligned_col_major_sf) and "NVFP4 scales must use row-major layout");
        EP_HOST_ASSERT(num_experts_per_rank > 0 and "Dispatch requires num_experts_per_rank at buffer construction");
        const int num_local_experts = num_experts_per_rank;
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT((not use_fp8_dispatch or x.element_size() == sizeof(__nv_fp8_e4m3)) and
                       "An FP8-capacity buffer requires one-byte dispatch activations; construct with use_fp8_dispatch=False for BF16");
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

        // Capacities of the handle tensors: the maximum world, fixed local slots, the whole launch grid
        const int num_launch_channels = get_num_launch_channels();

        // Check SF stuffs
        int num_sf_packs = 0;
        void* sf_ptr = nullptr;
        int sf_token_stride = 0, sf_hidden_stride = 0;
        if (sf.has_value()) {
            // SF is FP32, packed UE8M0x4, or packed NVFP4 block-scale bytes.
            const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());
            EP_HOST_ASSERT(num_tokens == num_tokens_);
            EP_HOST_ASSERT(sf->is_cuda());
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
            EP_HOST_ASSERT((not nvfp4 or (num_sf_packs_ == hidden / 64 and sf->is_contiguous())) and
                           "NVFP4 block scales require contiguous int32 packs of shape [tokens, hidden / 64]");
            num_sf_packs = num_sf_packs_;
            sf_ptr = sf->data_ptr();
            sf_token_stride = sf->stride(0);
            sf_hidden_stride = sf->stride(1);
        }
        EP_HOST_ASSERT((use_fp8_dispatch ? num_sf_packs <= get_max_num_fp8_sf_packs(hidden) : num_sf_packs * sizeof(sf_pack_t) <= hidden) and
                       "Too many dispatch SF packs for the selected capacity; use at most ceil(hidden / 32) packs for an FP8-capacity buffer");

        EP_HOST_ASSERT((x.element_size() == 1 or (x.element_size() == sizeof(nv_bfloat16) and num_sf_packs == 0)) and
                       "Dispatch requires one-byte activations with optional SF, or two-byte activations without SF");

        // Check top-k stuffs
        const auto [num_tokens_, num_topk] = get_shape<2>(topk_idx);
        EP_HOST_ASSERT(num_tokens == num_tokens_);
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());

        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            const auto [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());
            EP_HOST_ASSERT(num_tokens == num_tokens__);
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // Expert receiving counter
        const bool is_capturing = c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None;
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            const auto [num_local_experts_] = get_shape<1>(cumulative_local_expert_recv_stats.value());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                           cumulative_local_expert_recv_stats->is_contiguous());
            if (num_local_experts_ < num_local_experts)
                EP_HOST_UNREACHABLE(fmt::format("`cumulative_local_expert_recv_stats` has {} entries: allocate at least {} for the fixed local expert slots",
                                               num_local_experts_, num_local_experts));
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        // Under stream capture the kernel does not run, so the host spin of a CPU sync would end in the CPU timeout: refuse it
        if (do_cpu_sync)
            EP_HOST_ASSERT(not is_capturing and
                           "`dispatch(do_cpu_sync=True)` inside a CUDA-graph capture: capture without the CPU sync");

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream);

        // Handle tensors: capacity storage (the maximum world, the whole launch grid) handed back as extent-shaped views
        const auto int_options = at::TensorOptions(torch::kCUDA).dtype(torch::kInt);
        const auto alloc_flat = [&](const int64_t& capacity, const std::vector<int64_t>& extent_shape) {
            const auto extent_numel = std::accumulate(extent_shape.begin(), extent_shape.end(), int64_t(1), std::multiplies<>());
            EP_HOST_ASSERT(extent_numel <= capacity);
            return torch::empty({capacity}, int_options).slice(0, 0, extent_numel).view(extent_shape);
        };

        // The number of received tokens per expert
        // This is useful for expanding mode
        // NOTES: for expand mode, the input is exclusive prefix sum, while for non-expand, it is inclusive
        auto psum_num_recv_tokens_per_expert = torch::empty({num_local_experts + 1}, int_options);

        // The unaligned (actual) number of received tokens per expert
        // Written by the dispatch kernel's notify warps, used by the epilogue for zero padding
        auto num_unaligned_recv_tokens_per_expert = torch::empty({num_local_experts}, int_options);
        int* num_unaligned_recv_tokens_per_expert_ptr = num_unaligned_recv_tokens_per_expert.data_ptr<int>();

        // The prefix sum tensor of number of received tokens from each rank
        // Will also be used in combine as the dispatch handle
        // `[max_num_scaleup_ranks + 1]`: the total goes into the fixed last slot; zeroed on the communication stream, which the kernel follows
        auto psum_num_recv_tokens_per_scaleup_rank = torch::empty({max_num_scaleup_ranks + 1}, int_options);
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(psum_num_recv_tokens_per_scaleup_rank.data_ptr(), 0,
                                           psum_num_recv_tokens_per_scaleup_rank.nbytes(), comm_stream));

        // Decide number of channels by shared memory consumption
        const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
        const int num_channels_per_sm = get_num_channels_per_sm(
            hidden, num_hidden_bytes, num_sf_packs, num_topk, max_num_ranks * num_experts_per_rank,
            max_num_ranks, num_smem_bytes, prefer_overlap_with_compute);
        if (get_env<int>("NIXL_EP_BUFFER_DEBUG"))
            printf("Elastic buffer uses %d channels per SM\n", num_channels_per_sm);
        const int num_channels = topo.num_active_sms * num_channels_per_sm;

        // Handles: the forwarding metadata and the linked list are allocated in both modes (the direct mode
        // leaves them unused), so that a captured graph can switch modes
        // Capacities: `num_channels * num_max_tokens_per_channel <= num_max_tokens_per_rank + num_channels` at the whole launch grid
        const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
        const int64_t max_channel_tokens = static_cast<int64_t>(num_max_tokens_per_rank) + num_launch_channels;
        const int64_t max_linked_list_len = max_num_scaleout_ranks * max_channel_tokens + num_launch_channels;

        // The token metadata during forward
        // `[i, j]` means: in channel i, the j-th forwarded token's metadata
        // Info contains:
        //   - Scaleout rank index and source token index in the original rank (0)
        //   - Whether the token is the last one in the chunk (1)
        //   - cached top-k scaleup peer indices (top-k)
        //   - each selections' destination slot indices (top-k)
        const auto num_max_forwarded_tokens = topo.num_scaleout_ranks * num_max_tokens_per_channel + 1;
        const auto num_forward_metadata_dims = 2 + num_topk * 2;
        auto token_metadata_at_forward = alloc_flat(
            max_linked_list_len * num_forward_metadata_dims,
            {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims});
        int* token_metadata_at_forward_ptr = token_metadata_at_forward.data_ptr<int>();

        // Per-scaleup-peer-per-channel linked list
        // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's index in the combine's input
        auto channel_linked_list = alloc_flat(
            // Index 0 of the list means the starting item
            max_linked_list_len * max_num_scaleup_ranks,
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
        const auto arena = get_buffer_arena(num_max_tokens_per_rank, hidden, num_topk);

        // Ready and clean host workspace for this round
        const auto host_workspace_layout = layout::WorkspaceLayout(host_workspace);
        std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(), topo.num_scaleup_ranks, 0);
        std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(), num_local_experts, 0);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Do dispatch into the buffers (with SM limitation)
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
                        buffer, math::advance_ptr(buffer, arena.dispatch_send_offset),
                        mapped_host_workspace,
                        get_topology_ptr(),
                        max_num_scaleout_ranks, max_num_scaleup_ranks,
                        num_channels_per_sm,
                        num_smem_bytes,
                        num_gpu_timeout_cycles,
                        do_cpu_sync,
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
                    ss << "CPU side received count (rank " << topo.scaleout_rank_idx << "/" << topo.scaleup_rank_idx << "): ";
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
            // Reserve every source token in each local slot it may select, plus fixed local alignment slack.
            num_recv_tokens = num_max_tokens_per_rank * max_num_ranks;
            num_expanded_tokens = num_recv_tokens * std::min(num_topk, num_local_experts);
            num_expanded_tokens += (expert_alignment - 1) * num_local_experts;
            num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
        }

        // Allocate received tensors
        // `recv_src_metadata` includes source token indices and buffer slot indices
        const auto num_allocated_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
        auto recv_x = torch::empty({num_allocated_tokens, num_cols}, x.options());
        auto recv_sf = std::optional<torch::Tensor>();
        auto recv_topk_idx = std::optional<torch::Tensor>();
        auto recv_topk_weights = std::optional<torch::Tensor>();
        auto recv_src_metadata = torch::empty({num_recv_tokens, num_topk + 2}, int_options);

        // Optional tensors
        void* recv_sf_ptr = nullptr;
        topk_idx_t* recv_topk_idx_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
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

        // Expanded mode atomically advances exclusive prefixes; compact mode exposes inclusive prefixes.
        const int prefix_start = do_expand ? 0 : 1;
        psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(
            0, prefix_start, prefix_start + num_local_experts);

        // Launch copy kernels with full SMs
        stream_control_before_epilogue(previous_event_before_epilogue);
        // Without a CPU sync the count is the worst case: a negative value tells the epilogue to read the fixed slot
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
                                      do_expand, do_zero_padding,
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
                num_recv_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_scaleup_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                token_metadata_at_forward,
                channel_linked_list,
                event};
    }

    std::tuple<torch::Tensor, std::optional<EventHandle>>
    combine(const torch::Tensor& x,
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
        const auto& topo = active_rank_topology();

        // Check data
        EP_HOST_ASSERT(num_experts_per_rank > 0 and "Combine requires num_experts_per_rank at buffer construction");
        const auto [num_tokens, hidden] = get_shape<2>(x);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(hidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 and
                       "Combine requires hidden divisible by 256");

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
        // For reduction mode, `num_reduced_tokens` means the number of unexpanded tokens
        const auto [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);
        EP_HOST_ASSERT(num_reduced_tokens == (use_expanded_layout ? num_reduced_tokens : num_tokens));
        EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);
        EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);

        // Check optional tensors
        const auto bias_opts = std::vector({bias_0, bias_1});
        void* bias_ptrs[2] = {nullptr, nullptr};
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.is_cuda() and bias.dim() == 2 and bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
                bias_ptrs[i] = bias.data_ptr();
            }
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream);

        // Check buffer size
        const auto arena = get_buffer_arena(num_max_tokens_per_rank, hidden, num_topk);

        // The forwarding metadata of the dispatch handle
        // The token metadata during forward
        const auto [num_channels_, d1, d2] = get_shape<3>(token_metadata_at_forward);
        EP_HOST_ASSERT(num_sms > 0 and num_sms == topo.num_active_sms and
                       "The handle was produced with another active SM count");
        EP_HOST_ASSERT(num_channels_ > 0 and num_channels_ % num_sms == 0 and
                       num_channels_ / num_sms <= kNumMaxChannelsPerSM and
                       "Invalid dispatch handle channel count");
        const int num_channels_per_sm = num_channels_ / num_sms;
        const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels_);
        const int num_channels = num_channels_;
        int* token_metadata_at_forward_ptr = token_metadata_at_forward.data_ptr<int>();
        EP_HOST_ASSERT(d1 == topo.num_scaleout_ranks * num_max_tokens_per_channel + 1);
        EP_HOST_ASSERT(d2 == 2 + num_topk * 2);
        EP_HOST_ASSERT(token_metadata_at_forward.is_cuda() and token_metadata_at_forward.is_contiguous());

        // Per-scaleup-peer-per-channel linked list
        const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list);
        int* channel_linked_list_ptr = channel_linked_list.data_ptr<int>();
        EP_HOST_ASSERT(num_channels == num_channels__);
        EP_HOST_ASSERT(d1_ == topo.num_scaleout_ranks * num_max_tokens_per_channel + 1);
        EP_HOST_ASSERT(d2_ == topo.num_scaleup_ranks);
        EP_HOST_ASSERT(channel_linked_list.is_cuda() and channel_linked_list.is_contiguous());

        // Push data into remote buffers
        launch_combine(
            x.data_ptr(),
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
            token_metadata_at_forward_ptr,
            channel_linked_list_ptr,
            buffer, math::advance_ptr(buffer, arena.combine_send_offset),
            get_topology_ptr(),
            recv_count_is_exact ? num_reduced_tokens : -1, num_combined_tokens, num_max_tokens_per_rank,
            hidden, num_topk,
            num_gpu_timeout_cycles,
            max_num_scaleup_ranks,
            num_channels_per_sm,
            jit::device_runtime->get_num_smem_bytes(),
            use_expanded_layout, allow_multiple_reduction,
            comm_stream);

        // Allocate output tensors
        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());

        // Combine pushed data
        stream_control_before_epilogue(previous_event_before_epilogue);
        launch_combine_reduce_epilogue(combined_x.data_ptr(),
                                       combined_topk_idx.data_ptr<topk_idx_t>(),
                                       num_combined_tokens, num_max_tokens_per_rank,
                                       hidden,
                                       num_experts_per_rank, num_topk,
                                       num_gpu_timeout_cycles,
                                       buffer,
                                       bias_ptrs[0], bias_ptrs[1],
                                       get_topology_ptr(),
                                       jit::device_runtime->get_num_sms(),
                                       jit::device_runtime->get_num_smem_bytes(),
                                       num_channels_per_sm,
                                       use_expanded_layout, allow_multiple_reduction,
                                       comm_stream);

        // Stream control
        const auto event = stream_control_epilogue(
            {x, bias_0, bias_1,
             src_metadata,
             combined_topk_idx,
             combined_x,
             psum_num_recv_tokens_per_scaleup_rank,
             token_metadata_at_forward,
             channel_linked_list},
            compute_stream,
            allocate_on_comm_stream, async_with_compute_stream);
        return {combined_x, event};
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait);

    pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<int64_t, bool, bool, bool, int, int, bool, int, int, int, int, int, std::string, int>())
        .def("destroy", &ElasticBuffer::destroy)
        .def("get_comm_stream", &ElasticBuffer::get_comm_stream)
        .def("stage_topology_masks", &ElasticBuffer::stage_topology_masks)
        .def("commit_topology", &ElasticBuffer::commit_topology)
        .def("get_topology", &ElasticBuffer::get_topology)
        .def("get_topology_masks", &ElasticBuffer::get_topology_masks)
        .def("is_active", &ElasticBuffer::is_active)
        .def("has_staged_topology", &ElasticBuffer::has_staged_topology)
        .def("get_local_metadata", &ElasticBuffer::get_local_metadata)
        .def("get_local_peer_info", &ElasticBuffer::get_local_peer_info)
        .def("connect_ranks", &ElasticBuffer::connect_ranks)
        .def("disconnect_ranks", &ElasticBuffer::disconnect_ranks)
        .def("get_connected_ranks", &ElasticBuffer::get_connected_ranks)
        .def("read_profile", &ElasticBuffer::read_profile)
        .def("barrier", &ElasticBuffer::barrier)
        .def("dispatch", &ElasticBuffer::dispatch)
        .def("combine", &ElasticBuffer::combine);
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);
}

}  // namespace nixl_ep_v2::elastic
