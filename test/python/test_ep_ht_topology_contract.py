# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _function_body(source: str, signature: str) -> str:
    match = re.search(rf"{re.escape(signature)}\s*\{{(?P<body>.*?)\n\}}", source, re.S)
    assert match is not None, f"missing function body for {signature}"
    return match.group("body")


def test_runtime_reports_active_nvl_and_rdma_topology() -> None:
    header = read_repo_file("examples/device/ep/csrc/nixl_ep.hpp")
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    assert "int get_num_nvl_ranks() const;" in header
    assert "int get_num_rdma_ranks() const;" in header
    assert "int Buffer::get_num_nvl_ranks() const" in source
    assert '.def("get_num_nvl_ranks", &nixl_ep::Buffer::get_num_nvl_ranks)' in source
    assert '.def("get_num_rdma_ranks", &nixl_ep::Buffer::get_num_rdma_ranks)' in source


def test_runtime_derives_rdma_rank_from_active_nvl_rank_count() -> None:
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    init_body = _function_body(
        source,
        "void Buffer::init(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes)",
    )

    assert "num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS)" in init_body
    assert "num_rdma_ranks = ceil_div<int>(num_ranks, num_nvl_ranks)" in init_body
    assert "rdma_rank = rank / num_nvl_ranks" in init_body
    assert "nvl_rank = rank % num_nvl_ranks" in init_body


def test_ht_runtime_initializes_active_rank_count_before_peer_connect() -> None:
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    init_body = _function_body(
        source,
        "void Buffer::init(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes)",
    )

    assert (
        "if (not low_latency_mode and num_nvl_bytes > 0)\n"
        "        this->num_ranks = num_ranks;"
    ) in init_body
    assert init_body.index("this->num_ranks = num_ranks") < init_body.index(
        "num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS)"
    )


def test_ht_peer_connect_preserves_initialized_active_rank_topology() -> None:
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    connect_body = _function_body(
        source,
        "void Buffer::connect_ranks(const std::vector<int>& remote_ranks_list, const std::optional<std::vector<nixl_blob_t>>& remote_mds,\n"
        "    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles, bool activate)",
    )

    assert (
        "if (not low_latency_mode and num_nvl_bytes > 0) {\n"
        "        EP_HOST_ASSERT(max_added_rank < num_ranks);\n"
        "    } else {\n"
        "        num_ranks = std::max(num_ranks, max_added_rank + 1);\n"
        "    }"
    ) in connect_body


def test_buffer_init_allows_named_single_node_ht_rdma_staging() -> None:
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    init_body = _function_body(
        source,
        "void Buffer::init(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes)",
    )

    assert "single_node_high_throughput" in init_body
    assert "(num_ranks == 4 or num_ranks == NUM_MAX_NVL_PEERS)" in init_body
    assert "not low_latency_mode" in init_body
    assert "num_nvl_bytes > 0" in init_body
    assert "num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode or single_node_high_throughput" in init_body
    assert "EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);" not in init_body


def test_config_rdma_size_hint_includes_four_rank_ht_target() -> None:
    config = read_repo_file("examples/device/ep/csrc/config.hpp")
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    assert "num_ranks < NUM_MAX_NVL_PEERS and num_ranks != 4" in config
    assert "num_ranks <= NUM_MAX_NVL_PEERS and num_ranks != 4" not in config
    assert "num_ranks == 4 or num_ranks % NUM_MAX_NVL_PEERS == 0" in config
    assert "num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS)" in config
    assert "num_rdma_ranks = ceil_div<int>(num_ranks, num_nvl_ranks)" in config
    assert "rdma_buffer_size_hint <= static_cast<size_t>(num_rdma_bytes)" in source


def test_named_single_node_ht_paths_are_not_reported_as_intranode_only() -> None:
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    body = _function_body(source, "bool Buffer::is_ht_available() const")

    assert "(num_nvl_ranks == 4 or num_nvl_ranks == NUM_MAX_NVL_PEERS)" in body
    assert "num_rdma_ranks == 1" in body
    assert "num_rdma_ranks > 1" in body
    assert "num_ranks > NUM_MAX_NVL_PEERS" not in body
    assert "not low_latency_mode" in body
    assert "num_nvl_bytes > 0" in body
    assert "num_rdma_bytes > 0" in body


def test_named_single_node_ht_layout_keeps_one_visible_rdma_group() -> None:
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    init_body = _function_body(
        source,
        "void Buffer::init(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes)",
    )
    availability_body = _function_body(source, "bool Buffer::is_ht_available() const")
    layout_body = _function_body(
        source,
        "Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,\n"
        "                            std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream)",
    )

    assert "(num_ranks == 4 or num_ranks == NUM_MAX_NVL_PEERS)" in init_body
    assert "(num_nvl_ranks == 4 or num_nvl_ranks == NUM_MAX_NVL_PEERS)" in availability_body
    assert "num_rdma_ranks = ceil_div<int>(num_ranks, num_nvl_ranks)" in init_body
    assert "torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA))" in layout_body


def test_layout_kernel_uses_active_nvl_rank_count_for_rdma_metadata() -> None:
    api = read_repo_file("examples/device/ep/csrc/kernels/api.cuh")
    layout = read_repo_file("examples/device/ep/csrc/kernels/layout.cu")
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    assert "int num_nvl_ranks," in api
    assert "int num_nvl_ranks," in layout
    assert "num_nvl_ranks," in source
    assert "/ num_nvl_ranks - rdma_rank_begin_idx" in layout
    assert "ceil_div<int>(rank_end_idx, num_nvl_ranks)" in layout
    assert "num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS" not in layout


def test_peer_wiring_uses_explicit_active_local_ranks_for_ipc_sync() -> None:
    header = read_repo_file("examples/device/ep/csrc/nixl_ep.hpp")
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    assert "std::vector<int> get_active_local_ranks() const;" in header
    assert "std::vector<int> Buffer::get_active_local_ranks() const" in source

    helper_body = _function_body(source, "std::vector<int> Buffer::get_active_local_ranks() const")
    assert "std::min(local_rank_begin + num_nvl_ranks, num_ranks)" in helper_body
    assert "std::min(local_rank_begin + num_nvl_ranks, max_num_ranks)" not in helper_body

    body = _function_body(
        source,
        "void Buffer::_ipc_handles_sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles = {})",
    )

    assert "EP_HOST_ASSERT(all_gathered_handles.size() == static_cast<size_t>(num_ranks));" in body
    assert "EP_HOST_ASSERT(all_gathered_handles.size() == max_num_ranks);" not in body
    assert "const auto active_local_ranks = get_active_local_ranks();" in body
    assert "local_idx < static_cast<int>(active_local_ranks.size())" in body
    assert "int remote_rank = active_local_ranks[local_idx]" in body
    assert "all_gathered_handles[remote_rank]" in body
    assert "offset + i" not in body


def test_four_rank_layout_allocates_one_visible_rdma_group_metadata_tensor() -> None:
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    body = _function_body(
        source,
        "Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,\n                            std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream)",
    )

    assert "if (is_ht_available())" in body
    assert "torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA))" in body
    assert "torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA))" in body
    assert "num_tokens, num_topk, num_ranks, num_nvl_ranks, num_experts" in body


def _declaration(source: str, name: str) -> str:
    match = re.search(rf"void {re.escape(name)}\([^;]+;", source, re.S)
    assert match is not None, f"missing declaration for {name}"
    return match.group(0)


def test_ht_api_and_runtime_calls_pass_active_nvl_rank_count() -> None:
    api = read_repo_file("examples/device/ep/csrc/kernels/api.cuh")
    source = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    for name in ("notify_dispatch", "dispatch", "cached_notify", "combine"):
        assert "int num_nvl_ranks" in _declaration(api, name)

    assert re.search(r"ht::notify_dispatch\([^;]*num_ranks,\s*num_nvl_ranks,\s*num_tokens_per_rdma_rank", source, re.S)
    assert re.search(r"ht::dispatch\([^;]*rank,\s*num_ranks,\s*num_nvl_ranks,\s*cached_mode", source, re.S)
    assert re.search(r"ht::combine\([^;]*rank,\s*num_ranks,\s*num_nvl_ranks,\s*comm_stream", source, re.S)
    assert len(re.findall(r"ht::cached_notify\([^;]*num_ranks,\s*num_nvl_ranks,\s*num_channels", source, re.S)) == 2


def test_ht_launch_specializes_rdma_from_active_nvl_rank_count() -> None:
    launch = read_repo_file("examples/device/ep/csrc/kernels/launch.cuh")
    source = read_repo_file("examples/device/ep/csrc/kernels/nixl_ep_ht.cu")

    assert "#define NUM_RDMA_RANKS_FOR(num_ranks, num_nvl_ranks)" in launch
    assert "#define SWITCH_NVL_AND_RDMA_RANKS(case_macro)" in launch
    assert "NUM_RDMA_RANKS_FOR(num_ranks, num_nvl_ranks)" in source
    assert "NUM_RDMA_RANKS_FOR(num_ranks)" not in source
    assert source.count("SWITCH_NVL_AND_RDMA_RANKS(") >= 3
    assert "get_nvl_clean_meta(hidden_int4,\n                                             num_scales,\n                                             num_topk,\n                                             num_topk,\n                                             num_rdma_ranks,\n                                             num_nvl_ranks" in source


def test_ht_notify_and_cached_notify_use_active_nvl_barriers_and_offsets() -> None:
    source = read_repo_file("examples/device/ep/csrc/kernels/nixl_ep_ht.cu")

    assert "template <bool kLowLatencyMode, int kNumNVLRanks, int kNumRDMARanks>" in source
    assert "template <bool kLowLatencyMode, int kNumNVLRanks, int kNumRDMARanks, int kNumTMABytesPerWarp>" in source
    assert "barrier_block<kNumNVLRanks, true>(barrier_signal_ptrs, nvl_rank, timeout_cycles)" in source
    assert "barrier_block<kNumNVLRanks>(barrier_signal_ptrs, nvl_rank, timeout_cycles)" in source
    assert "nixl_barrier_send_warp<kNumNVLRanks>(nixl_ctx, num_channels)" in source
    assert "load_token_in_nvl_rank_bits<kNumNVLRanks>" in source
    assert "rank / NUM_MAX_NVL_PEERS" not in source
    assert "rank % NUM_MAX_NVL_PEERS" not in source
    assert "barrier_block<NUM_MAX_NVL_PEERS" not in source


def test_cached_notify_kernel_uses_compile_time_rdma_rank_count() -> None:
    source = read_repo_file("examples/device/ep/csrc/kernels/nixl_ep_ht.cu")
    match = re.search(r"__global__ void cached_notify\(.*?\n\}\n\nvoid cached_notify\(", source, re.S)
    assert match is not None, "missing templated cached_notify kernel body"
    kernel_body = match.group(0)

    assert "dst_rdma_rank < kNumRDMARanks" in kernel_body
    assert "dst_rdma_rank < num_rdma_ranks" not in kernel_body


def test_ht_dispatch_and_combine_do_not_assign_roles_to_capacity_slots() -> None:
    source = read_repo_file("examples/device/ep/csrc/kernels/nixl_ep_ht.cu")

    assert "__launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + kNumNVLRanks) * 32), 1)" in source
    assert "EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + kNumNVLRanks)" in source
    assert "int kNumRDMAReceivers = kNumForwarders - kNumNVLRanks" in source
    assert "if (warp_id < kNumNVLRanks)" in source
    assert "% kNumNVLRanks" in source
    assert "forwarder_nvl_head[kNumForwarders][kNumNVLRanks]" in source
    assert "lane_id < kNumNVLRanks" in source
    assert "combine_token<kNumNVLRanks, false" in source


def test_ht_nvl_head_storage_keeps_capacity_stride_for_abi() -> None:
    source = read_repo_file("examples/device/ep/csrc/kernels/nixl_ep_ht.cu")
    runtime = read_repo_file("examples/device/ep/csrc/nixl_ep.cpp")

    assert "send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}" in runtime
    assert "combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS" in runtime
    assert "send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank" in source
    assert "send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head" in source
    assert "constexpr int num_bytes_per_token = sizeof(int) * NUM_MAX_NVL_PEERS" in source
    assert "combined_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS" in source
    assert "combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS" in source
    assert "combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id" in source
