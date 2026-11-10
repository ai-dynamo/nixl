# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import functools
import math
import os
import pickle
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
from .. import _C

# noinspection PyUnresolvedReferences
from .._C import EventHandle
from ..utils.envs import (
    check_torch_deterministic,
    get_logical_domain_size,
    get_nvlink_gbs,
    get_rdma_gbs,
)
from ..utils.event import EventOverlap
from ..utils.nixl_bootstrap import NixlBootstrap, resolve_bootstrap_store
from ..utils.semantic import align, value_or, weak_lru

_DEFAULT_SM_READ_GBS, _DEFAULT_SM_WRITE_GBS = 200, 50


class EPHandle:
    """
    Communication handle returned by `ElasticBuffer.dispatch`.
    It is consumed by `ElasticBuffer.combine` to reverse the token routing.

    Attributes:
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        num_experts: the number of all experts.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        num_sms: the active SM count recorded when the handle was created, checked by eager combine. A captured
            dispatch/combine graph reads the current count from the topology descriptor.
        topk_idx: dispatch top-k expert indices, `[num_tokens, num_topk]`; cloned when `do_handle_copy=True`.
        psum_num_recv_tokens_per_scaleup_rank: inclusive prefix sum of deduplicated received token counts
            per scaleup rank, shape `[max_num_scaleup_ranks + 1]`. A token is counted once per rank even if
            multiple of its top-k experts land on the same rank. The last element equals the total number
            of received tokens.
        psum_num_recv_tokens_per_expert: prefix sum of alignment-padded received token counts per local
            expert, shape `[num_local_experts]`. Each expert's count is padded to `expert_alignment`.
            In non-expand mode, this is the inclusive prefix sum. In expand mode, `psum[i]` equals
            the aligned cumulative count of experts before `i` plus the actual (unaligned) token count
            of expert `i` — so `psum[i] - align(psum[i-1], expert_alignment)` recovers the real
            count for expert `i`, and `align(psum[i], expert_alignment)` gives expert `i+1`'s
            starting offset.
        num_recv_tokens_per_expert_list: Python list of per-expert received token counts (CPU-side).
        num_unaligned_recv_tokens_per_expert: the actual (unaligned) number of tokens received per local
            expert, shape `[num_local_experts]` with `torch.int`, in both compact and expanded layouts.
        recv_src_metadata: source token indices and buffer slot indices.
        token_metadata_at_forward: per-channel forwarded token metadata (allocated in both modes, written in hybrid mode).
        channel_linked_list: per-channel per-scaleup-peer linked list (allocated in both modes, written in hybrid mode).
        num_recv_tokens: the total number of received tokens (the worst case of the declared maximum world unless
            `recv_count_is_exact`).
        recv_count_is_exact: whether `num_recv_tokens` is the real count (the dispatch had a CPU sync).
        num_recv_tokens_device: 0-d `torch.int` device view of the real count, valid without a CPU sync and in graphs.
    """

    def __init__(
        self,
        do_expand: bool,
        num_experts: int,
        expert_alignment: int,
        num_max_tokens_per_rank: int,
        num_sms: int,
        topk_idx: torch.Tensor,
        num_recv_tokens: int,
        num_recv_tokens_per_expert_list: list,
        psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
        psum_num_recv_tokens_per_expert: torch.Tensor,
        num_unaligned_recv_tokens_per_expert: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        token_metadata_at_forward: torch.Tensor,
        channel_linked_list: torch.Tensor,
        recv_count_is_exact: bool,
    ):
        assert topk_idx is not None

        self.do_expand = do_expand
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
        self.topk_idx = topk_idx
        self.psum_num_recv_tokens_per_scaleup_rank = (
            psum_num_recv_tokens_per_scaleup_rank
        )
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.recv_src_metadata = recv_src_metadata
        self.token_metadata_at_forward = token_metadata_at_forward
        self.channel_linked_list = channel_linked_list

        # May not be accurate without CPU sync
        self.num_recv_tokens = num_recv_tokens
        self.recv_count_is_exact = recv_count_is_exact

    @property
    def num_recv_tokens_device(self) -> torch.Tensor:
        """The real receive count: a 0-d view of the fixed last prefix-sum slot."""
        return self.psum_num_recv_tokens_per_scaleup_rank[-1]

    def deterministic_sort(
        self,
        recv_x: torch.Tensor,
        recv_sf: Optional[torch.Tensor],
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
    ):
        """
        Sort received tokens to guarantee deterministic dispatch output.
        The principle:
          - Non-expand mode: sort everything that depends on the receive order, including
            `recv_x`, `recv_sf`, `recv_topk_weights`, `recv_topk_idx`, and `self.recv_src_metadata`.
          - Expand mode: only sort the expanded arrays — `recv_x`, `recv_sf`, and `recv_topk_weights`.
            The slot pointers in `self.recv_src_metadata[:, 2:]` are updated to reflect the new positions, but `self.recv_src_metadata` itself is not permuted.
        """

        sort_keys = self.recv_src_metadata[:, 0]

        # Ignore trailing tokens by setting their `sort_keys` to max
        num_recv_tokens = (
            self.num_recv_tokens_device
            if not self.recv_count_is_exact
            else self.recv_src_metadata.shape[0]
        )
        if not self.recv_count_is_exact:
            oob_tokens_mask = (
                torch.arange(
                    0,
                    self.recv_src_metadata.shape[0],
                    device=self.recv_src_metadata.device,
                )
                >= num_recv_tokens
            )
            sort_keys = sort_keys.clone()
            sort_keys[oob_tokens_mask] = torch.iinfo(sort_keys.dtype).max
        orig_indices = torch.sort(sort_keys).indices

        def get_reverse_permutation(perm: torch.Tensor) -> torch.Tensor:
            assert perm.dim() == 1
            result = torch.empty_like(perm)
            result[perm] = torch.arange(
                0, perm.shape[0], dtype=perm.dtype, device=perm.device
            )
            return result

        def permute(tensor: Optional[torch.Tensor], orig_indices: torch.Tensor):
            if tensor is not None:
                tmp = tensor[orig_indices]
                tensor.copy_(tmp)

        if not self.do_expand:
            # Non-expand mode
            permute(recv_x, orig_indices)
            permute(recv_sf, orig_indices)
            permute(recv_topk_weights, orig_indices)
            permute(recv_topk_idx, orig_indices)
            permute(self.recv_src_metadata, orig_indices)

            valid_mask = (self.channel_linked_list >= 0) & (
                self.channel_linked_list < num_recv_tokens
            )
            to_indices = get_reverse_permutation(orig_indices)
            self.channel_linked_list[valid_mask] = to_indices[
                self.channel_linked_list[valid_mask]
            ].to(self.channel_linked_list.dtype)

        else:
            # In expand mode, `recv_x`, `recv_sf`, and `recv_topk_weights` are grouped by expert ID, possibly with padding (expert alignment). We permute tokens within each expert and update `self.recv_src_metadata[:, 2:]` accordingly.

            # Now we're going to construct the sorting key, which is:
            #  - `expert_idx*src_token_global_index_max_x2 + (-src_token_global_index_max) + src_token_global_idx`, for valid tokens
            #  - `expert_idx * src_token_global_index_max_x2`, for padding slots
            # This guarantees a two-key sort: first by expert, then by order within each expert.
            # Valid tokens precede padding tokens, and valid tokens are sorted by `src_token_global_idx`.
            src_token_global_index_max_x2 = 10000000000  # 1e10
            tensor_dim0_after_expand = recv_x.shape[0]

            expert_token_idx_start = (
                self.psum_num_recv_tokens_per_expert
                - self.num_unaligned_recv_tokens_per_expert
            )
            token_idx2expert_idx = torch.bucketize(
                torch.arange(tensor_dim0_after_expand, device="cuda"),
                expert_token_idx_start[1:],
                right=True,
                out_int32=False,
            )
            sort_keys_for_expanded_tensors = (
                token_idx2expert_idx * src_token_global_index_max_x2
            )

            slots = self.recv_src_metadata[:, 2:]  # [num_recv_tokens, topk]
            src_global_idx = self.recv_src_metadata[:, 0]
            valid_mask = slots >= 0
            if not self.recv_count_is_exact:
                valid_mask[oob_tokens_mask] = False
            sort_keys_for_expanded_tensors.scatter_add_(
                0,
                slots[valid_mask],
                -src_token_global_index_max_x2 // 2
                + src_global_idx.unsqueeze(1)
                .expand_as(slots)[valid_mask]
                .to(torch.int64),
            )

            orig_indices_for_expanded_tensors = torch.sort(
                sort_keys_for_expanded_tensors, stable=True
            ).indices.to(torch.int32)
            permute(recv_x, orig_indices_for_expanded_tensors)
            permute(recv_sf, orig_indices_for_expanded_tensors)
            permute(recv_topk_weights, orig_indices_for_expanded_tensors)

            to_indices_for_expanded_tensors = get_reverse_permutation(
                orig_indices_for_expanded_tensors
            )
            self.recv_src_metadata[:, 2:][valid_mask] = to_indices_for_expanded_tensors[
                self.recv_src_metadata[:, 2:][valid_mask]
            ]


# The `mode` field of the topology descriptor (`EPTopology`): direct = 0, hybrid = 1 (the index in `_MODE_NAMES`)
_MODE_NAMES = ("direct", "hybrid")


def _mode_code(mode: str) -> int:
    assert mode in _MODE_NAMES, f"mode must be direct or hybrid, got {mode!r}"
    return _MODE_NAMES.index(mode)


class ElasticBuffer:
    """
    The elastic communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine, using NVLink and/or RDMA)

    Attributes:
        rank_idx: this rank's cohort rank, its index in the constructing cohort.
        num_ranks: the number of ranks in the group.
        store: the explicit `torch.distributed.Store` given to the constructor (the default of `publish` and `connect_ranks`).
        allow_multiple_reduction: whether to allow multiple reductions in combine. If disabled,
            only one reduction will be done in the combine epilogue for best precision,
            but it may increase data transfer size.
        prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            If enabled, we tend to use fewer SMs.
        num_bytes: the total buffer size in bytes.
        num_max_tokens_per_rank: the default maximum tokens per rank.
        max_num_scaleout_ranks: the declared maximum number of scaleout ranks (NVLink domains) the buffer is sized for.
        max_num_scaleup_ranks: the declared maximum number of scaleup ranks (positions per domain).
        scaleout_rank_idx: this rank's NVLink domain index in the declared maximum world.
        scaleup_rank_idx: this rank's position inside its NVLink domain.
        world_rank: this rank's world rank, its cell `scaleout_rank_idx * max_num_scaleup_ranks + scaleup_rank_idx` of the
            declared maximum world (the transport's peer index).
        num_launch_sms: the fixed launch grid of the adaptive kernels (24, or `NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS`, even,
            at most 64); `num_sms` is the active block count, even and at most this.
        runtime: the C++ runtime.

    Elasticity: the active ranks are a product subset of the declared maximum world (active NVLink domains x active
    positions, `masks_of`). `connect_ranks` / `disconnect_ranks` stage peers, `stage_topology` + `commit_topology`
    rewrite the topology descriptor (`EPTopology`) the kernels read, a rank that is not active sits the launches
    out (`is_active`). Construction coordinates the constructing cohort through a Store (see the constructor).
    Dispatch/combine and their graph replays require at least two active GPUs; smaller cohorts may bootstrap or tear down.
    Every peer or topology change (`connect_ranks`, `disconnect_ranks`, `commit_topology`, `set_num_active_sms`)
    is collective, runs between launches and never inside a CUDA-graph capture, and never between a
    dispatch and its combine.
    Order topology commits and graph replays on the same CUDA stream, or establish the dependencies explicitly.
    Before membership or transport-view changes, every rank of the old world, including departing ranks, must finish
    its queued CUDA work, call `barrier(with_cpu_sync=True)` to drain the transport, and rendezvous on the host before
    any peer departs. The update methods do not establish that completion boundary.
    Transport mutation/activation failures are fatal; invalid inputs and mapping refusals leave active views intact.

    Three numberings: the cohort rank (`rank_idx`, this rank's index in the constructing cohort), the world rank
    (`world_rank`, its cell in the declared maximum world, the transport's peer index) and the rank index (the kernels'
    `rank_idx` over the committed extent, `scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx`).
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        # Provide `num_bytes` (excludes workspace)
        num_bytes: Optional[int] = None,
        # Or provide MoE settings (BF16 by default)
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        # Configs
        deterministic: bool = False,
        allow_multiple_reduction: bool = True,
        prefer_overlap_with_compute: bool = True,
        max_num_scaleout_ranks: int = 0,
        max_num_scaleup_ranks: int = 0,
        num_nixl_channels: int = 0,
        scaleout_rank_idx: Optional[int] = None,
        scaleup_rank_idx: Optional[int] = None,
        store: Optional[dist.Store] = None,
        rank_idx: Optional[int] = None,
        num_ranks: Optional[int] = None,
        num_group_scaleup_ranks: Optional[int] = None,
        bootstrap_id: Optional[str] = None,
        store_timeout_secs: int = 300,
        num_cpu_timeout_secs: int = 300,
        num_gpu_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
    ):
        """
        Initialize the elastic communication buffer.
        The kernel mode follows the active NVLink domains at construction and each subsequent topology change.

        Arguments:
            group: the process group whose ranks form the constructing cohort. Unless both `store` and
                `bootstrap_id` are supplied, a private TCPStore is created and its address is broadcast over the group.
                Use `None` with `rank_idx`, `num_ranks`, `store` and `bootstrap_id`.
            num_bytes: the total buffer size in bytes (excludes workspace), if set, overrides MoE-based calculation.
                Must be aligned to 2 MB.
            num_max_tokens_per_rank: the maximum number of tokens per rank, used for buffer size calculation.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: reserve smaller dispatch arenas for FP8 inputs only; False reserves BF16 capacity and also
                accepts FP8 inputs. Must match across peers, including with explicit `num_bytes`; it does not cast inputs.
            deterministic: whether to sort received tokens deterministically (eager execution only).
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            num_cpu_timeout_secs: CPU-side timeout in seconds for CPU sync.
            num_gpu_timeout_secs: GPU-side timeout in seconds for GPU operations.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
            max_num_scaleout_ranks: the maximum number of scaleout ranks (NVLink domains) the buffer, the handles and
                the kernels are sized for; 0 = the cohort's domain count. The active ranks are any product subset of it.
            max_num_scaleup_ranks: likewise for the positions per domain, at most 128 (`kNumMaxScaleupRanksCap`).
            num_nixl_channels: UCX device channels per peer of the NIXL agent; 0 = `NIXL_EP_NIXL_NUM_CHANNELS` or 64 (UCX's maximum).
            scaleout_rank_idx: this rank's NVLink domain in the maximum world (default: from the cohort's domain count and size).
            scaleup_rank_idx: this rank's position inside its domain (default: from the cohort's domain count and size).
            store: an explicit `torch.distributed.Store`: the default of `publish` and `connect_ranks`, and with
                `bootstrap_id` the Store the cohort bootstraps through (else the group's first rank serves a cohort-private TCPStore).
            rank_idx: this rank's index in the constructing cohort (required without a group; not the world rank).
            num_ranks: the size of the constructing cohort (required without a group; a joiner uses a one-rank cohort).
            num_group_scaleup_ranks: the cohort's declared NVLink-domain size (must divide `num_ranks`; contiguous blocks
                validated as one domain each; `num_ranks` = one domain); omitted: inferred from the GPU evidence.
                NVML proves the links are NVLink, not PCIe P2P. Commit separately requires CUDA-IPC mappings within
                each domain and RDMA between domains; each declared domain must be a complete CUDA-IPC clique.
            bootstrap_id: the cohort's namespace in `store` (required with an explicit bootstrap Store; distinct per
                concurrent cohort, fresh after a partial construction or a change of the cohort's membership).
            store_timeout_secs: how long the bootstrap waits for the cohort at each of its Store steps.
        """
        if group is not None:
            if rank_idx not in (None, group.rank()) or num_ranks not in (
                None,
                group.size(),
            ):
                raise ValueError(
                    "rank_idx / num_ranks do not match the supplied process group"
                )
            rank_idx, num_ranks = group.rank(), group.size()
        if (
            type(rank_idx) is not int
            or type(num_ranks) is not int
            or not 0 <= rank_idx < num_ranks
        ):
            raise ValueError(
                "Without a process group, rank_idx and a positive num_ranks are required"
            )
        self.rank_idx, self.num_ranks = rank_idx, num_ranks
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.deterministic = deterministic
        self.store = store
        # Automatic/manual selection follows the topology's stage/commit transaction.
        self._auto_sm_config: Optional[Tuple[int, int]] = None
        self._staged_auto_sm_config: Optional[Tuple[int, int]] = None

        # Coordinate the cohort through a Store: the domain count and size, the session token, the peer metadata, the barrier
        bootstrap_store, namespace = resolve_bootstrap_store(
            group,
            store,
            bootstrap_id,
            "elastic-buffer",
            timeout_seconds=store_timeout_secs,
        )
        bootstrap = NixlBootstrap(
            bootstrap_store,
            self.rank_idx,
            self.num_ranks,
            namespace=namespace,
            num_scaleup_ranks=num_group_scaleup_ranks,
            timeout_seconds=store_timeout_secs,
        )
        num_group_scaleout_ranks = bootstrap.num_scaleout_ranks
        group_scaleup_ranks: int = bootstrap.num_scaleup_ranks

        # The declared maximum world and this rank's coordinates in it: explicit values win, else the cohort's domain count and size
        max_num_scaleout_ranks = max_num_scaleout_ranks or num_group_scaleout_ranks
        max_num_scaleup_ranks = max_num_scaleup_ranks or group_scaleup_ranks
        if scaleout_rank_idx is None:
            scaleout_rank_idx = self.rank_idx // group_scaleup_ranks
        if scaleup_rank_idx is None:
            scaleup_rank_idx = self.rank_idx % group_scaleup_ranks

        # Calculate buffer size (already 2 MB-aligned from hint functions / calculate_elastic_buffer_size)
        if num_bytes is None:
            # NOTES: we allow `num_topk == 0`, as the buffer size can also be calculated by number of ranks (maybe bigger though)
            num_bytes = _C.calculate_elastic_buffer_size(
                max_num_scaleout_ranks,
                max_num_scaleup_ranks,
                num_max_tokens_per_rank,
                hidden,
                num_topk,
                use_fp8_dispatch,
                allow_multiple_reduction,
            )

        if int(os.environ.get("NIXL_EP_BUFFER_DEBUG", 0)):
            print(
                f"Initializing EP elastic buffer with {num_bytes} bytes "
                f"at cohort rank {self.rank_idx}/{self.num_ranks}"
            )
        self.num_bytes = num_bytes

        self.num_max_tokens_per_rank = num_max_tokens_per_rank

        # Avoid matching replacement processes to closed UD wire-up endpoints. This process-wide default can be
        # overridden before construction; see the README deployment guidance.
        os.environ.setdefault("UCX_UD_LINGER_TIMEOUT", "1s")

        # NIXL: the UCX device channels of the agent (a warp is a channel; the transport refuses more than UCX's 64)
        num_nixl_channels = num_nixl_channels or int(
            os.getenv("NIXL_EP_NIXL_NUM_CHANNELS", 64)
        )

        self.explicitly_destroy = explicitly_destroy
        self.runtime = _C.ElasticBuffer(
            num_bytes,
            allow_multiple_reduction,
            use_fp8_dispatch,
            prefer_overlap_with_compute,
            num_cpu_timeout_secs,
            num_gpu_timeout_secs,
            self.explicitly_destroy,
            max_num_scaleout_ranks,
            max_num_scaleup_ranks,
            scaleout_rank_idx,
            scaleup_rank_idx,
            num_nixl_channels,
            bootstrap.session_token,
        )

        # Adaptive kernels: the fixed launch grid; the active SM count is set per topology via `set_num_active_sms`
        self.num_launch_sms = _C.get_adaptive_num_launch_sms()
        self.max_num_scaleout_ranks, self.max_num_scaleup_ranks = (
            max_num_scaleout_ranks,
            max_num_scaleup_ranks,
        )

        self.scaleout_rank_idx, self.scaleup_rank_idx = (
            scaleout_rank_idx,
            scaleup_rank_idx,
        )
        self.world_rank = self.world_rank_of(scaleout_rank_idx, scaleup_rank_idx)

        # Connect the whole cohort and make it the active ranks (the constructor is collective over the cohort); the gather
        # returns every member's `(metadata, peer info)` by world rank
        infos = bootstrap.all_gather_peer_infos(
            (self.get_local_metadata(), self.get_local_peer_info())
        )
        peers = [wr for wr in infos if wr != self.world_rank]
        self.connect_ranks(
            peers, [infos[wr][0] for wr in peers], [infos[wr][1] for wr in peers]
        )
        self.stage_topology(
            num_active_sms=self.num_launch_sms, active_ranks=sorted(infos)
        )
        self.commit_topology()

        # Call a barrier to ensure initialization visibility for all peers
        torch.cuda.synchronize()
        bootstrap.barrier("ready")

    def destroy(self) -> None:
        """Destroy the runtime; requires `explicitly_destroy=True` at construction."""
        assert self.explicitly_destroy

        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None  # Cannot use anymore

    @staticmethod
    def get_buffer_size_hint(
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        allow_multiple_reduction: bool = True,
        max_num_scaleout_ranks: int = 0,
        max_num_scaleup_ranks: int = 0,
        num_group_scaleup_ranks: Optional[int] = None,
        store: Optional[dist.Store] = None,
        bootstrap_id: Optional[str] = None,
        store_timeout_secs: int = 300,
    ) -> int:
        """
        Get a recommended buffer size (in bytes) for the given MoE settings, without constructing the buffer.
        The returned value is aligned to 2 MB. A missing maximum is discovered from the group's domain count and size (a Store exchange,
        collective over `group`); the constructor repeats the exchange. To reuse the discovered domain count and size, call
        `nixl.runtime.ep.get_logical_domain_size` once and pass its `num_scaleup_ranks` as `num_group_scaleup_ranks` with both
        maxima: the constructor then validates declared blocks instead of inferring.

        Arguments:
            group: the process group whose domain count and size fill in a missing maximum (`None` with both maxima given).
            num_max_tokens_per_rank: the maximum number of tokens per rank.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: size dispatch arenas for FP8 inputs only; pass the same value to the constructor.
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            max_num_scaleout_ranks: the declared maximum world to size for (0 = the group's domain count).
            max_num_scaleup_ranks: likewise (0 = the group's domain size).
            num_group_scaleup_ranks: the group's declared domain size, as the constructor's argument of the same name.
            store: with `bootstrap_id`, the Store used for discovery; otherwise a private TCPStore whose address is broadcast over the group.
            bootstrap_id: the namespace in `store`.
            store_timeout_secs: the Store timeout of the discovery.

        Returns:
            size: the recommended buffer size in bytes (2 MB-aligned).
        """
        if max_num_scaleout_ranks == 0 or max_num_scaleup_ranks == 0:
            if group is None:
                raise ValueError(
                    "get_buffer_size_hint without a process group needs both max_num_scaleout_ranks and max_num_scaleup_ranks"
                )
            group_domains, group_positions = get_logical_domain_size(
                group,
                num_scaleup_ranks=num_group_scaleup_ranks,
                store=store,
                bootstrap_id=bootstrap_id,
                store_timeout_secs=store_timeout_secs,
            )
            max_num_scaleout_ranks = max_num_scaleout_ranks or group_domains
            max_num_scaleup_ranks = max_num_scaleup_ranks or group_positions
        # NOTES: calculate_elastic_buffer_size already returns 2 MB-aligned values
        return _C.calculate_elastic_buffer_size(
            max_num_scaleout_ranks,
            max_num_scaleup_ranks,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            use_fp8_dispatch,
            allow_multiple_reduction,
        )

    def read_profile(self) -> Dict[str, torch.Tensor]:
        """Read and clear diagnostic timestamps; requires NIXL_EP_ADAPTIVE_PROFILE=1 before library initialization.

        Returns a mapping from native phase names to CPU int64 tensors indexed by [block, warp].
        Values are device global-timer nanoseconds; zero means that writer did not reach that phase.
        This call synchronizes the device and completes the reset before returning; it cannot run during capture.
        Discard warm-up samples with one read, then run one isolated dispatch/combine pair or graph replay per read.
        Instrumented phase boundaries are diagnostic samples, not uninstrumented kernel completion times.
        """
        timestamps, names = self.runtime.read_profile()
        return {name: timestamps[index] for index, name in enumerate(names)}

    def barrier(
        self, use_comm_stream: bool = True, with_cpu_sync: bool = False
    ) -> None:
        """
        Drain outstanding transport operations and barrier across the active ranks, optionally synchronizing the CPU.

        Arguments:
            use_comm_stream: whether to use the communication stream (otherwise uses the current compute stream).
            with_cpu_sync: whether to also call `cudaDeviceSynchronize` before and after the barrier.
        """
        self.runtime.barrier(use_comm_stream, with_cpu_sync)

    @staticmethod
    def capture() -> EventHandle:
        """Capture a CUDA event on the current stream."""
        return EventHandle()

    def get_comm_stream(self) -> torch.Stream:
        """Return the buffer's communication stream."""
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(
            stream_id=ts.stream_id,
            device_index=ts.device_index,
            device_type=ts.device_type,
        )

    def get_theoretical_num_sms(
        self,
        num_experts: int,
        num_topk: int,
        rdma_gbs: float = 0,
        nvlink_gbs: float = 0,
        # TODO: use different values for other architectures
        sm_read_gbs: float = _DEFAULT_SM_READ_GBS,
        sm_write_gbs: float = _DEFAULT_SM_WRITE_GBS,
    ) -> int:
        """
        Estimate the optimal number of SMs for dispatch/combine kernels based on bandwidth modeling.
        The result is cached per set of active ranks and argument set. This assumes a balanced gate distribution.
        This query does not change the buffer's automatic/manual SM policy.
        The raw estimate may exceed `num_launch_sms`; automatic selection clamps it to that launch limit.
        Traffic uses active domain/position counts; experts per rank follow the extent product. One active domain
        has no RDMA traffic even above domain index 0. Fewer active experts than `num_topk` requires explicit `num_sms`.
        Every active rank's `dispatch(num_sms=0)` commits the same active SM count.

        Arguments:
            num_experts: the number of all experts.
            num_topk: the number of top-k experts per token.
            rdma_gbs: the RDMA bandwidth in GB/s (0 for auto-detect).
            nvlink_gbs: the NVLink bandwidth in GB/s (0 for auto-detect).
            sm_read_gbs: the per-SM HBM read bandwidth in GB/s.
            sm_write_gbs: the per-SM HBM write bandwidth in GB/s.

        Returns:
            num_sms: the raw model estimate (even, at least 2), before the launch-grid limit.
        """
        return self._get_theoretical_num_sms_for_topology(
            num_experts,
            num_topk,
            *self.get_topology_masks(),
            rdma_gbs=rdma_gbs,
            nvlink_gbs=nvlink_gbs,
            sm_read_gbs=sm_read_gbs,
            sm_write_gbs=sm_write_gbs,
        )

    def _get_theoretical_num_sms_for_topology(
        self,
        num_experts: int,
        num_topk: int,
        scaleout_mask: int,
        scaleup_mask: int,
        rdma_gbs: float = 0,
        nvlink_gbs: float = 0,
        sm_read_gbs: float = _DEFAULT_SM_READ_GBS,
        sm_write_gbs: float = _DEFAULT_SM_WRITE_GBS,
    ) -> int:
        num_active_scaleout_ranks, num_active_scaleup_ranks = bin(scaleout_mask).count(
            "1"
        ), bin(scaleup_mask).count("1")
        num_active_ranks = num_active_scaleout_ranks * num_active_scaleup_ranks
        # The active ranks' expert count: their number times the experts per rank of the requested extent (the kernels
        # divide `num_experts` over the extent product, so an extent with holes owns fewer experts than the total)
        num_scaleout_extent, num_scaleup_extent = (
            scaleout_mask.bit_length(),
            scaleup_mask.bit_length(),
        )
        assert (
            num_experts % (num_scaleout_extent * num_scaleup_extent) == 0
        ), f"{num_experts} experts do not divide by the topology extent {num_scaleout_extent}x{num_scaleup_extent}"
        num_active_experts = (
            num_experts // (num_scaleout_extent * num_scaleup_extent) * num_active_ranks
        )
        assert (
            num_active_experts >= num_topk
        ), f"the active ranks own {num_active_experts} experts, fewer than num_topk={num_topk}: the traffic model has no estimate, pass num_sms"
        return self._get_theoretical_num_sms(
            num_active_experts,
            num_topk,
            num_active_scaleout_ranks,
            num_active_scaleup_ranks,
            rdma_gbs,
            nvlink_gbs,
            sm_read_gbs,
            sm_write_gbs,
        )

    @weak_lru(maxsize=None)
    def _get_theoretical_num_sms(
        self,
        num_experts: int,
        num_topk: int,
        num_active_scaleout_ranks: int,
        num_active_scaleup_ranks: int,
        rdma_gbs: float,
        nvlink_gbs: float,
        sm_read_gbs: float,
        sm_write_gbs: float,
    ) -> int:
        # TODO: support `do_expand` and `allow_multiple_reduction`

        # The `1` in this function means scale-up traffic
        # i.e. the HBM read volume of the dispatch copy epilogue, equals to "the number of tokens" * "num_expected_topk" * "data size per token"
        num_active_ranks = num_active_scaleout_ranks * num_active_scaleup_ranks

        # Get bandwidth
        if rdma_gbs == 0 and num_active_scaleout_ranks > 1:
            rdma_gbs = get_rdma_gbs()
        if nvlink_gbs == 0 and num_active_scaleup_ranks > 1:
            nvlink_gbs = get_nvlink_gbs()

        # Initial count
        # NOTES: we don't count HBM traffic
        sm_read: float = 0
        sm_write: float = 0
        rdma_traffic: float = 0
        nvlink_traffic: float = 0

        def get_expected_topk(num_groups: int) -> float:
            assert num_experts % num_groups == 0
            return num_groups * (
                1
                - math.comb(num_experts - num_experts // num_groups, num_topk)
                / math.comb(num_experts, num_topk)
            )

        # Expected top-k scale-out ranks
        num_expected_scaleout_topk = (
            get_expected_topk(num_active_scaleout_ranks)
            if num_active_scaleout_ranks > 1
            else 0
        )

        # Expected top-k scale-up ranks
        num_expected_topk = get_expected_topk(num_active_ranks)

        # Read tokens
        sm_read += 1 / num_expected_topk

        # NOTES: we don't consider the skip-send-buffer cases (all selections fall in the local)
        if num_active_scaleout_ranks > 1:
            # Scaleup warps: write send buffer
            sm_write += 1 / num_expected_topk

            # Scaleout traffic
            sm_write += (1 / num_expected_topk) * (
                num_expected_scaleout_topk / num_active_scaleout_ranks
            )  # Local bypass
            rdma_traffic += (1 / num_expected_topk) * (
                num_expected_scaleout_topk * (1 - 1 / num_active_scaleout_ranks)
            )

            # Forward warps
            sm_read += num_expected_scaleout_topk / num_expected_topk
            sm_write += 1  # Issue scaleup
            nvlink_traffic += 1 - (1 / num_active_scaleup_ranks)
        else:
            # One active domain: `num_active_ranks == num_active_scaleup_ranks`, no send buffer and no scale-out traffic
            # Issue NVLink
            sm_write += 1.0

            # NVLink traffic, except local bypass
            nvlink_traffic += 1 - 1 / num_active_scaleup_ranks

        # Found the bounded one
        if num_active_scaleout_ranks > 1 and (
            num_active_scaleup_ranks == 1
            or (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs)
        ):
            bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
        else:
            bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs

        # Calculate SM count
        # NOTES: will try to use more SMs if not overlap with compute
        num_device_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_sms = num_device_sms  # No traffic, e.g., EP=1
        if bounded_traffic > 0:
            num_sms = max(
                bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
                bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
            )
        num_sms = align(max(2, math.ceil(num_sms * 1.25)), 2)
        num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)
        num_sms = min(num_sms, num_device_sms)

        # Summary
        if int(os.environ.get("NIXL_EP_BUFFER_DEBUG", 0)):
            print(
                f"EP SM approximation: "
                f"{sm_read=}, {sm_write=}, {rdma_traffic=}, {nvlink_traffic=}, "
                f"{rdma_gbs=}, {nvlink_gbs=}, "
                f"{num_expected_scaleout_topk=}, {num_expected_topk=}, "
                f"{bounded_traffic=}, {bounded_gbs=}, {num_sms=}"
            )
        return num_sms

    def get_local_metadata(self) -> bytes:
        """This rank's NIXL agent metadata, which a peer loads to address this rank (`connect_ranks`)."""
        return self.runtime.get_local_metadata()

    def get_local_peer_info(self) -> Tuple[int, int, int, int]:
        """This region's `(base address, CUDA device index, world rank, allocation bytes)`."""
        return self.runtime.get_local_peer_info()

    def publish(
        self, store: Optional[dist.Store] = None, key_prefix: str = "EP_NIXL"
    ) -> None:
        """
        Publish this rank's transport metadata to a `torch.distributed.Store` under `{key_prefix}/{world_rank}`, for ranks
        outside this cohort to `connect_ranks` to it (same Store and `key_prefix` on both sides).
        On process replacement, use a fresh shared `key_prefix`, or rendezvous after all targets' current processes
        finish publishing before connecting. Store keys survive departed processes; `bootstrap_id` does not scope them.

        Arguments:
            store: the publication Store; defaults to the constructor's explicit `store` (never the bootstrap Store).
            key_prefix: the key prefix the readers pass to `connect_ranks`.
        """
        store = value_or(store, self.store)
        if store is None:
            raise ValueError(
                "Peer publication requires an explicit Store; pass store to ElasticBuffer or publish(store=...)"
            )
        store.set(
            f"{key_prefix}/{self.world_rank}",
            pickle.dumps((self.get_local_metadata(), self.get_local_peer_info())),
        )

    def connect_ranks(
        self,
        world_ranks: List[int],
        remote_mds: Optional[List[bytes]] = None,
        peer_infos: Optional[List[Tuple[int, int, int, int]]] = None,
        store: Optional[dist.Store] = None,
        key_prefix: str = "EP_NIXL",
        timeout_secs: float = 120.0,
    ) -> None:
        """
        Load peers' transport metadata and stage new memory views (made live by the next `commit_topology`), from
        `remote_mds` and `peer_infos` as published by the peers, or from a `store` they `publish`ed to.
        The per-peer barrier words are reset and the stream synchronised.

        Arguments:
            world_ranks: world ranks of the peers (not this rank, not connected yet).
            remote_mds: the peers' `get_local_metadata()`, aligned with `world_ranks`.
            peer_infos: the peers' `get_local_peer_info()`, aligned with `world_ranks`.
            store: read `{key_prefix}/{world_rank}` from this store instead (default: the constructor's explicit `store`).
            key_prefix: the publisher's key prefix; replacement freshness is the caller's responsibility (see `publish`).
            timeout_secs: how long to wait for each peer's key in `store` (the wire-up has its own `NIXL_EP_NIXL_CONNECT_TIMEOUT_SECS`).
        """
        if (remote_mds is None) != (peer_infos is None):
            raise ValueError(
                "remote_mds and peer_infos are given together (the peers' metadata and peer info), or neither (Store discovery)"
            )
        if remote_mds is None or peer_infos is None:
            store = value_or(store, self.store)
            if store is None:
                raise ValueError(
                    "Peer discovery requires an explicit Store; pass store to ElasticBuffer or "
                    "connect_ranks(store=...), or pass remote_mds and peer_infos"
                )
            remote_mds, peer_infos = [], []
            for world_rank in world_ranks:
                key = f"{key_prefix}/{world_rank}"
                store.wait([key], timedelta(seconds=timeout_secs))
                md, info = pickle.loads(store.get(key))
                remote_mds.append(md)
                peer_infos.append(info)
        self.runtime.connect_ranks(
            list(world_ranks), list(remote_mds), [tuple(info) for info in peer_infos]
        )

    def disconnect_ranks(self, world_ranks: List[int]) -> None:
        """
        Forget peers whose process goes away (staged; dropped at the next `commit_topology`). Commit a topology without
        the peer first (a staged, uncommitted topology must not hold it either). A rank that stays alive but sits a world
        out is left out of `active_ranks` instead.
        """
        self.runtime.disconnect_ranks(list(world_ranks))

    def get_connected_ranks(self) -> List[int]:
        """World ranks of the connected peers (this rank excluded)."""
        return list(self.runtime.get_connected_ranks())

    def world_rank_of(self, scaleout_rank_idx: int, scaleup_rank_idx: int) -> int:
        """The world rank of coordinates `(scaleout_rank_idx, scaleup_rank_idx)` in the declared maximum world."""
        return scaleout_rank_idx * self.max_num_scaleup_ranks + scaleup_rank_idx

    def masks_of(self, active_ranks: Sequence[int]) -> Tuple[int, int]:
        """The `(scaleout_mask, scaleup_mask)` of world ranks forming a product of domains and positions."""
        active = sorted(set(int(r) for r in active_ranks))
        domains = sorted(set(r // self.max_num_scaleup_ranks for r in active))
        positions = sorted(set(r % self.max_num_scaleup_ranks for r in active))
        product = sorted(self.world_rank_of(d, k) for d in domains for k in positions)
        assert (
            product == active
        ), f"The active ranks must be a product of domains and positions; {domains} x {positions} != {active}"
        return sum(1 << d for d in domains), sum(1 << k for k in positions)

    @staticmethod
    def mode_of(scaleout_mask: int) -> str:
        """Exactly one active NVLink domain selects direct mode, regardless of its index."""
        return "direct" if bin(scaleout_mask).count("1") == 1 else "hybrid"

    def _stage(
        self,
        *,
        mode: Optional[str] = None,
        num_active_sms: int = 0,
        active_ranks: Optional[Sequence[int]] = None,
    ) -> None:
        scaleout_mask, scaleup_mask = (
            self.masks_of(active_ranks)
            if active_ranks is not None
            else self.get_topology_masks()
        )
        auto_sm_config = None if num_active_sms else self._auto_sm_config
        if not num_active_sms:
            num_active_sms = self.get_topology()[1]
            # Inactive ranks and bootstrap/teardown worlds do not launch dispatch/combine.
            if (
                auto_sm_config is not None
                and bin(scaleout_mask).count("1") * bin(scaleup_mask).count("1")
                >= _C.min_num_active_ranks
                and scaleout_mask & (1 << self.scaleout_rank_idx)
                and scaleup_mask & (1 << self.scaleup_rank_idx)
            ):
                num_active_sms = self._clamp_active_sms(
                    self._get_theoretical_num_sms_for_topology(
                        *auto_sm_config, scaleout_mask, scaleup_mask
                    )
                )
        if mode is None:
            mode = self.mode_of(scaleout_mask)
        # The position mask crosses the extension as 64-bit words (a whole NVL72 domain needs more than 64 bits)
        words = [
            (scaleup_mask >> (64 * i)) & ((1 << 64) - 1)
            for i in range(max(1, (scaleup_mask.bit_length() + 63) // 64))
        ]
        self.runtime.stage_topology_masks(
            _mode_code(mode), scaleout_mask, words, num_active_sms
        )
        self._staged_auto_sm_config = auto_sm_config

    def stage_topology(
        self, *, num_active_sms: int = 0, active_ranks: Optional[Sequence[int]] = None
    ) -> None:
        """
        Prepare a new topology descriptor (`EPTopology`) on the host, validated here and made live by `commit_topology`.
        The mode is not an input: it follows `mode_of` on the staged domains.
        A `set_num_active_sms` before the commit (a `dispatch` whose `num_sms` differs from the
        committed count calls the former) is refused, as it would recommit the committed world over the staged one:
        commit first, or stage after it.

        Arguments:
            num_active_sms: the active blocks of the fixed grid (even, at most `num_launch_sms`). A nonzero value pins
                manual mode; 0 retunes from the last successful automatic dispatch, or keeps the count in manual mode.
            active_ranks: the world ranks of the active ranks, a product set (`masks_of`) of connected ranks; `None`
                keeps the committed active ranks.
        """
        self._stage(num_active_sms=num_active_sms, active_ranks=active_ranks)

    def commit_topology(self) -> None:
        """
        Make the staged topology descriptor (`EPTopology`) live: one stream-ordered device write on the current stream
        (a device synchronisation first when peers were connected or disconnected).
        """
        has_staged_topology = self.runtime.has_staged_topology()
        self.runtime.commit_topology()
        if has_staged_topology:
            self._auto_sm_config = self._staged_auto_sm_config
        self._staged_auto_sm_config = None

    def set_active_ranks(
        self, active_ranks: Sequence[int], num_active_sms: int = 0
    ) -> None:
        """`stage_topology(active_ranks=...)` + `commit_topology()` in one step; the mode follows the new world."""
        self.stage_topology(num_active_sms=num_active_sms, active_ranks=active_ranks)
        self.commit_topology()

    def get_topology_masks(self) -> Tuple[int, int]:
        """Committed `(scaleout_mask, scaleup_mask)`: bit d / bit k denotes an active domain / position."""
        scaleout_mask, words = self.runtime.get_topology_masks()
        return scaleout_mask, sum(int(w) << (64 * i) for i, w in enumerate(words))

    def get_active_ranks(self) -> List[int]:
        """The world ranks of the committed active ranks, sorted."""
        scaleout_mask, scaleup_mask = self.get_topology_masks()
        domains = [
            d for d in range(self.max_num_scaleout_ranks) if (scaleout_mask >> d) & 1
        ]
        positions = [
            k for k in range(self.max_num_scaleup_ranks) if (scaleup_mask >> k) & 1
        ]
        return sorted(self.world_rank_of(d, k) for d in domains for k in positions)

    def get_topology_extent(self) -> Tuple[int, int]:
        """Committed extents: highest active domain index + 1 and highest active position + 1."""
        scaleout_mask, scaleup_mask = self.get_topology_masks()
        return scaleout_mask.bit_length(), scaleup_mask.bit_length()

    @property
    def is_active(self) -> bool:
        """Whether this rank belongs to the committed world, including singleton bootstrap worlds."""
        return self.runtime.is_active()

    def set_num_active_sms(self, num_active_sms: int) -> None:
        """
        Recommit the committed topology descriptor with a new active SM count (even, at most `num_launch_sms`); its world
        and mode are untouched. Pins manual mode until a successful `dispatch(num_sms=0)`.
        Refused while a staged world awaits `commit_topology`: commit it first, or stage after this call.
        """
        self._set_num_active_sms(num_active_sms)

    def _set_num_active_sms(
        self, num_active_sms: int, *, preserve_auto: bool = False
    ) -> None:
        assert (
            num_active_sms
        ), 'set_num_active_sms needs the count (0 is not "keep the current value" here)'
        assert (
            not self.runtime.has_staged_topology()
        ), "A staged world awaits commit_topology(): commit it before changing the active SM count, or stage after it"
        mode, _ = self.get_topology()
        self._stage(mode=mode, num_active_sms=num_active_sms)
        if preserve_auto:
            self._staged_auto_sm_config = self._auto_sm_config
        self.commit_topology()

    def get_topology(self) -> Tuple[str, int]:
        """The committed `(mode, num_active_sms)`."""
        mode, num_active_sms = self.runtime.get_topology()
        return _MODE_NAMES[mode], num_active_sms

    def _clamp_active_sms(self, num_sms: int) -> int:
        return max(2, min(align(num_sms, 2), self.num_launch_sms))

    def _adaptive_active_sms(self, num_sms: int) -> int:
        # The grid of the adaptive kernels is fixed, so `num_sms` only decides how many blocks do work: clamp it to
        # the launch grid (even, at least 2) and update the topology descriptor (`EPTopology`, its `num_active_sms`) if it changed
        num_active_sms = self._clamp_active_sms(num_sms)
        _, current = self.get_topology()
        if current != num_active_sms:
            assert (
                not torch.cuda.is_current_stream_capturing()
            ), f"`dispatch(num_sms={num_sms})` would change the committed active SM count ({current}) inside a CUDA-graph capture: warm up eagerly with the same `num_sms`"
            self._set_num_active_sms(
                num_active_sms, preserve_auto=True
            )  # a failed dispatch keeps the previous policy
        return num_active_sms

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        num_experts: int,
        topk_weights: Optional[torch.Tensor] = None,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: int = 1,
        num_sms: int = 0,
        previous_event: Optional[EventHandle] = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        do_handle_copy: bool = True,
        do_cpu_sync: bool = True,
        do_expand: bool = False,
        do_zero_padding: bool = False,
        use_tma_aligned_col_major_sf: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        EPHandle,
        EventOverlap,
    ]:
        """
        Dispatch tokens to different ranks. Supports both single-node and multi-node settings.
            The SM count is automatically determined if not specified.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be
                `[num_tokens, hidden]`, and type must be `torch.bfloat16`; for the second type (FP8 mode),
                the first element of the tuple must be `[num_tokens, hidden]` with type `torch.float8_e4m3fn`,
                the second is the scale factors.
            topk_idx: `[num_tokens, num_topk]` with `nixl.runtime.ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
            num_experts: the number of all experts, no longer optional because cached dispatch was removed. Upstream
                required it for a fresh dispatch and could obtain it from a cached handle. A multiple of the committed
                extent product (`get_topology_extent()`):
                each world rank inside the extent owns `num_experts / product` experts, so the active ranks own fewer
                than `num_experts` when the extent has holes.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            cumulative_local_expert_recv_stats: `[num_local_experts]` with `torch.int`, a cumulative expert count
                tensor for statistics, useful for online EP load balance monitoring.
                For capture, reserve `num_experts // 2` entries for shrink/expand replay on at least two active GPUs.
                Replay updates the current local-expert prefix without changing the allocation or recapturing.
            num_max_tokens_per_rank: the maximum number of tokens per rank (the constructor's value if not given).
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_sms: the number of SMs to use (0 for automatic via `get_theoretical_num_sms`). A successful automatic
                dispatch remembers `num_experts`/top-k for later topology updates; a nonzero value pins manual mode.
                The policy is shared by all graphs using this buffer; the latest successful dispatch selects it.
                This is the active block count of the fixed grid, clamped to `num_launch_sms` (even, at least 2).
                A value other than the committed active SM count rewrites the topology descriptor (`EPTopology`) eagerly;
                under a CUDA-graph capture it must equal the committed active SM count (warm up eagerly with the same value).
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            previous_event_before_epilogue: the event to wait before actually executing the copy epilogue.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.
            do_handle_copy: whether to clone `topk_idx` in the returned handle (to prevent user modification).
            do_cpu_sync: whether to synchronize with CPU to get exact received token counts.
            do_expand: whether to use the expanding layout (one slot per expert per token).
            do_zero_padding: whether to zero out the alignment padding slots in the expanded output.
                Only valid when `do_expand` is True. Ensures alignment gaps between experts are zeroed.
            use_tma_aligned_col_major_sf: whether to use TMA-aligned column-major layout for scale factors.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`
            recv_topk_idx: received expert indices
            recv_topk_weights: received expert weights (`None` if `topk_weights` was not provided).
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        check_torch_deterministic()

        # Automatic decide SM count
        num_topk = topk_idx.shape[1]
        auto_sm_config = (num_experts, num_topk) if num_sms == 0 else None
        assert (
            auto_sm_config == self._auto_sm_config
            or not self.runtime.has_staged_topology()
        ), "A staged world awaits commit_topology(): commit it before changing the automatic/manual SM policy"
        num_sms = (
            self.get_theoretical_num_sms(num_experts, num_topk)
            if num_sms == 0
            else num_sms
        )
        # Fixed grid: `num_sms` becomes the active block count of the topology descriptor
        num_sms = self._adaptive_active_sms(num_sms)

        # Unpack SF
        x, sf = x if isinstance(x, tuple) else (x, None)

        if num_max_tokens_per_rank is None:
            num_max_tokens_per_rank = self.num_max_tokens_per_rank

        (
            recv_x,
            recv_sf,
            recv_topk_idx,
            recv_topk_weights,
            cloned_topk_idx,
            num_recv_tokens,
            num_recv_tokens_per_expert_list,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            num_unaligned_recv_tokens_per_expert,
            recv_src_metadata,
            token_metadata_at_forward,
            channel_linked_list,
            event,
        ) = self.runtime.dispatch(
            x,
            sf,
            topk_idx,
            topk_weights,
            cumulative_local_expert_recv_stats,
            num_max_tokens_per_rank,
            num_experts,
            expert_alignment,
            previous_event,
            previous_event_before_epilogue,
            async_with_compute_stream,
            allocate_on_comm_stream,
            do_handle_copy,
            do_cpu_sync,
            do_expand,
            do_zero_padding,
            use_tma_aligned_col_major_sf,
        )

        handle = EPHandle(
            do_expand,
            num_experts,
            expert_alignment,
            num_max_tokens_per_rank,
            num_sms,
            cloned_topk_idx if do_handle_copy else topk_idx,
            num_recv_tokens,
            num_recv_tokens_per_expert_list,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            num_unaligned_recv_tokens_per_expert,
            recv_src_metadata,
            token_metadata_at_forward,
            channel_linked_list,
            do_cpu_sync,
        )

        event_overlap = EventOverlap(event)

        # Deterministic epilogue
        # NOTES: when we change the metadata layout, the epilogue should also be changed
        if self.deterministic:
            epilogue = functools.partial(
                handle.deterministic_sort,
                recv_x,
                recv_sf,
                recv_topk_idx,
                recv_topk_weights,
            )
            (
                event_overlap.register_hook_after_wait(epilogue)
                if async_with_compute_stream
                else epilogue()
            )

        recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x

        self._auto_sm_config = auto_sm_config
        return recv_x, recv_topk_idx, recv_topk_weights, handle, event_overlap

    @staticmethod
    def _unpack_bias(
        bias: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        elif bias is not None:
            raise TypeError("bias must be a Tensor, a pair of Tensors, or None")
        return bias_0, bias_1

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        previous_event: EventHandle = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, EventOverlap]:
        """
        Combine (reduce) tokens from different ranks back to their original ranks.
        Supports both single-node and multi-node settings.
        Dispatch and combine use the topology descriptor's active block count throughout the pair.
        Eager combine checks the handle's recorded count; captured kernels read the current descriptor.
        Upstream's optional top-k weight return, used for routing-weight gradients in dispatch backward, is unsupported.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16` and `hidden` divisible by 256, the tokens to send for
                reducing to their original ranks. This combine alignment also applies after FP8 dispatch.
            handle: a must-set communication handle, you can obtain this from the `dispatch` function.
            bias: 0, 1 or 2 `[num_combined_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            previous_event_before_epilogue: the event to wait before actually executing the reduce epilogue.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        check_torch_deterministic()

        bias_0, bias_1 = ElasticBuffer._unpack_bias(bias)
        combined_x, event = self.runtime.combine(
            x,
            bias_0,
            bias_1,
            handle.recv_src_metadata,
            handle.topk_idx,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.token_metadata_at_forward,
            handle.channel_linked_list,
            handle.recv_count_is_exact,
            handle.num_experts,
            handle.num_max_tokens_per_rank,
            handle.num_sms,
            previous_event,
            previous_event_before_epilogue,
            async_with_compute_stream,
            allocate_on_comm_stream,
            handle.do_expand,
        )
        return combined_x, EventOverlap(event)
