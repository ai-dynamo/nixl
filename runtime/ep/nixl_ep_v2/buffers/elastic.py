# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import math
import pickle
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

# noinspection PyUnresolvedReferences
from .. import _C
# noinspection PyUnresolvedReferences
from .._C import EventHandle

from ..utils.event import EventOverlap
from ..utils.math import align
from ..utils.semantic import value_or, weak_lru
from ..utils.envs import (
    check_torch_deterministic, get_logical_domain_size,
    get_nvlink_gbs, get_rdma_gbs
)
from ..utils.nixl_bootstrap import NixlBootstrap, resolve_bootstrap_store


class EPHandle:
    """
    Communication handle returned by `ElasticBuffer.dispatch`.
    It is consumed by `ElasticBuffer.combine` to reverse the token routing.

    Attributes:
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        num_sms: the SM count used during dispatch (reused in combine).
        topk_idx: cloned top-k expert indices from dispatch, `[num_tokens, num_topk]`.
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
            expert, shape `[num_local_experts]` with `torch.int`. Only populated in expand mode.
        recv_src_metadata: source token indices and buffer slot indices.
        token_metadata_at_forward: per-channel forwarded token metadata (allocated in both modes, written in hybrid mode).
        channel_linked_list: per-channel per-scaleup-peer linked list (allocated in both modes, written in hybrid mode).
        num_recv_tokens: the total number of received tokens (the worst case of the declared maximum world unless
            `recv_count_is_exact`).
        recv_count_is_exact: whether `num_recv_tokens` is the real count (the dispatch had a CPU sync).
        num_recv_tokens_device: 0-d `torch.int` device view of the real count, valid without a CPU sync and in graphs.
    """

    def __init__(self,
                 do_expand: bool,
                 expert_alignment: int,
                 num_max_tokens_per_rank: int,
                 num_sms: int,
                 topk_idx: torch.Tensor,
                 num_recv_tokens: int,
                 num_expanded_tokens: int,
                 num_recv_tokens_per_expert_list: list,
                 psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
                 psum_num_recv_tokens_per_expert: torch.Tensor,
                 num_unaligned_recv_tokens_per_expert: torch.Tensor,
                 recv_src_metadata: torch.Tensor,
                 token_metadata_at_forward: torch.Tensor,
                 channel_linked_list: torch.Tensor, recv_count_is_exact: bool):
        # NOTES: remember to copy the original users' input to prevent uncasual modifications on them
        assert topk_idx is not None

        self.do_expand = do_expand
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
        self.topk_idx = topk_idx
        self.psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.recv_src_metadata = recv_src_metadata
        self.token_metadata_at_forward = token_metadata_at_forward
        self.channel_linked_list = channel_linked_list

        # May not be accurate without CPU sync
        self.num_recv_tokens = num_recv_tokens
        self.num_expanded_tokens = num_expanded_tokens
        self.recv_count_is_exact = recv_count_is_exact
        # The routing metadata as received, kept by `deterministic_sort` (which permutes `recv_src_metadata`)
        self.cached_recv_src_metadata_before_sort = None

    @property
    def num_recv_tokens_device(self) -> torch.Tensor:
        """
        The real receive count as a 0-d `torch.int` device tensor: the fixed last slot of `psum_num_recv_tokens_per_scaleup_rank`.
        """
        return self.psum_num_recv_tokens_per_scaleup_rank[-1]

    def deterministic_sort(self,
                           do_cpu_sync: bool,
                           recv_x: torch.Tensor,
                           recv_sf: Optional[torch.Tensor],
                           recv_topk_idx: torch.Tensor,
                           recv_topk_weights: torch.Tensor,
                           channel_linked_list: Optional[torch.Tensor]):
        """
        Sort received tokens to guarantee deterministic dispatch output.
        The principle:
          - Non-expand mode: sort everything that depends on the receive order, including
            `recv_x`, `recv_sf`, `recv_topk_weights`, `recv_topk_idx`, and `self.recv_src_metadata`.
          - Expand mode: only sort the expanded arrays — `recv_x`, `recv_sf`, and `recv_topk_weights`.
            The slot pointers in `self.recv_src_metadata[:, 2:]` are updated to reflect the new positions, but `self.recv_src_metadata` itself is not permuted.
        """

        # Preserve the fresh routing metadata while sorting the received tensors.
        self.cached_recv_src_metadata_before_sort = self.recv_src_metadata.clone()
        sort_keys = self.cached_recv_src_metadata_before_sort[:, 0]

        # Ignore trailing tokens by setting their `sort_keys` to max
        # NOTES: every step below is capture-safe (no boolean-mask indexing, which needs a host sync), so the sort can
        # be captured into a CUDA graph and replayed with another receive count
        num_recv_tokens = self.num_recv_tokens_device if not do_cpu_sync else self.recv_src_metadata.shape[0]
        if not do_cpu_sync:
            oob_tokens_mask = torch.arange(0, self.recv_src_metadata.shape[0], device=self.recv_src_metadata.device) >= num_recv_tokens
            sort_keys = sort_keys.clone()
            sort_keys.masked_fill_(oob_tokens_mask, torch.iinfo(sort_keys.dtype).max)
        orig_indices = torch.sort(sort_keys).indices

        def get_reverse_permutation(perm: torch.Tensor) -> torch.Tensor:
            assert perm.dim() == 1
            result = torch.empty_like(perm)
            result[perm] = torch.arange(0, perm.shape[0], dtype=perm.dtype, device=perm.device)
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

            if channel_linked_list is not None and orig_indices.numel() > 0:
                # The public view has the capture-time shape; replay can use the full backing allocation.
                linked_list = channel_linked_list.as_strided(
                    (channel_linked_list.untyped_storage().nbytes() // channel_linked_list.element_size(),),
                    (1,), storage_offset=0)
                valid_mask = (linked_list >= 0) & (linked_list < num_recv_tokens)
                to_indices = get_reverse_permutation(orig_indices)
                safe_indices = torch.where(valid_mask, linked_list, 0).to(torch.int64)
                linked_list.copy_(torch.where(valid_mask, to_indices[safe_indices].to(linked_list.dtype), linked_list))

        else:
            # In expand mode, `recv_x`, `recv_sf`, and `recv_topk_weights` are grouped by expert ID, possibly with padding (expert alignment). We permute tokens within each expert and update `self.recv_src_metadata[:, 2:]` accordingly.

            # Now we're going to construct the sorting key, which is:
            #  - `expert_idx*src_token_global_index_max_x2 + (-src_token_global_index_max) + src_token_global_idx`, for valid tokens
            #  - `expert_idx * src_token_global_index_max_x2`, for padding slots
            # This guarantees a two-key sort: first by expert, then by order within each expert.
            # Valid tokens precede padding tokens, and valid tokens are sorted by `src_token_global_idx`.
            src_token_global_index_max_x2 = 10000000000    # 1e10
            tensor_dim0_after_expand = recv_x.shape[0]
            if tensor_dim0_after_expand == 0:
                return

            expert_token_idx_start = self.psum_num_recv_tokens_per_expert - self.num_unaligned_recv_tokens_per_expert
            token_idx2expert_idx = torch.bucketize(torch.arange(tensor_dim0_after_expand, device='cuda'),
                                                   expert_token_idx_start[1:], right=True, out_int32=False)
            sort_keys_for_expanded_tensors = token_idx2expert_idx * src_token_global_index_max_x2

            slots = self.cached_recv_src_metadata_before_sort[:, 2:]    # [num_recv_tokens, topk]
            src_global_idx = self.cached_recv_src_metadata_before_sort[:, 0]
            valid_mask = slots >= 0
            if not do_cpu_sync:
                valid_mask = valid_mask & ~oob_tokens_mask.unsqueeze(1)
            # Invalid selections (-1 slots, trailing rows) scatter a zero offset into slot 0 instead of being filtered
            # by a boolean index, so the sort stays capture-safe
            safe_slots = torch.where(valid_mask, slots, 0).to(torch.int64)
            key_offsets = torch.where(valid_mask, -src_token_global_index_max_x2//2 + src_global_idx.unsqueeze(1).to(torch.int64), 0)
            sort_keys_for_expanded_tensors.scatter_add_(0, safe_slots.reshape(-1), key_offsets.reshape(-1))

            orig_indices_for_expanded_tensors = torch.sort(sort_keys_for_expanded_tensors, stable=True).indices.to(torch.int32)
            permute(recv_x, orig_indices_for_expanded_tensors)
            permute(recv_sf, orig_indices_for_expanded_tensors)
            permute(recv_topk_weights, orig_indices_for_expanded_tensors)

            to_indices_for_expanded_tensors = get_reverse_permutation(orig_indices_for_expanded_tensors)
            self.recv_src_metadata[:, 2:].copy_(torch.where(valid_mask, to_indices_for_expanded_tensors[safe_slots], slots))


class ElasticBuffer:
    """
    The elastic communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine, using NVLink and/or RDMA)

    Attributes:
        group: the communication group.
        rank_idx: the rank index.
        num_ranks: the number of ranks in the group.
        store: the explicit `torch.distributed.Store` given to the constructor (the default of `publish` and `connect_ranks`).
        allow_multiple_reduction: whether to allow multiple reductions in combine. If disabled,
            only one reduction will be done in the combine epilogue for best precision,
            but it may increase data transfer size.
        prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            If enabled, we tend to use fewer SMs.
        num_bytes: the total buffer size in bytes.
        num_max_tokens_per_rank: the default maximum tokens per rank.
        num_scaleout_ranks: the number of scaleout ranks of the declared maximum world (the live world: `get_active_ranks`).
        num_scaleup_ranks: the number of scaleup ranks.
        max_num_scaleout_ranks: the declared maximum number of scaleout ranks (NVLink domains) the buffer is sized for.
        max_num_scaleup_ranks: the declared maximum number of scaleup ranks (positions per domain).
        scaleout_rank_idx: the scaleout rank index of this rank.
        scaleup_rank_idx: the scaleup rank index of this rank.
        rank_id: this rank's dense id, `scaleout_rank_idx * max_num_scaleup_ranks + scaleup_rank_idx`.
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
        runtime: the C++ runtime.
        has_zero_copy, num_zero_copy_slots: the zero-copy output arena (`num_zero_copy_slots > 0`) and its slot count.

    Elasticity: the live world is a product subset of the declared maximum world (live NVLink domains x live
    positions, `masks_of`). `connect_ranks` / `disconnect_ranks` stage peers, `stage_topology` + `commit_topology`
    rewrite the topology descriptor (`EPTopology`) the kernels read, a rank outside the live world sits the launches
    out (`is_participant`). Construction coordinates the constructing cohort through a Store (see the constructor).
    """

    def __init__(self,
                 group: Optional[dist.ProcessGroup] = None,
                 # Provide `num_bytes` (excludes workspace)
                 num_bytes: Optional[int] = None,
                 # Or provide MoE settings (BF16 by default)
                 num_max_tokens_per_rank: int = 0,
                 hidden: int = 0,
                 num_topk: int = 0,
                 use_fp8_dispatch: bool = False,
                 num_experts_per_rank: int = 0,
                 # Configs
                 deterministic: bool = False,
                 allow_multiple_reduction: bool = True,
                 prefer_overlap_with_compute: bool = True,
                 max_num_scaleout_ranks: int = 0, max_num_scaleup_ranks: int = 0, num_nixl_channels: int = 0,
                 scaleout_rank_idx: Optional[int] = None, scaleup_rank_idx: Optional[int] = None,
                 store: Optional[dist.Store] = None, connect_group: bool = True,
                 rank_idx: Optional[int] = None, num_ranks: Optional[int] = None,
                 num_group_scaleup_ranks: Optional[int] = None, bootstrap_id: Optional[str] = None,
                 store_timeout_secs: int = 300,
                 num_cpu_timeout_secs: int = 300, num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False,
                 num_zero_copy_slots: int = 0,
                 zc_use_fp8_dispatch: bool = False,
                 zc_use_nvfp4_dispatch: bool = False,
                 dynamic_take: Optional[bool] = False,
                 acquire_flags: bool = False):
        """
        Initialize the elastic communication buffer.
        The kernel mode (the `mode` field of the topology descriptor, `EPTopology`) is decided by the library, never by the
        caller: direct iff exactly one NVLink domain is live, whatever its index, hybrid otherwise (`mode_of`). It is derived
        for the cohort at construction and for every world a later `stage_topology` / `set_active_ranks` stages. The only
        overrides are for tests and A/B runs: `stage_topology_forced` / `set_topology_forced`, and `NIXL_EP_ADAPTIVE_MODE=hybrid`
        on a one-domain world.

        Arguments:
            group: the process group whose ranks form the constructing cohort (its rank, its size and its own scoped
                Store are used, no collective), or `None` with `rank_idx`, `num_ranks`, `store` and `bootstrap_id`.
            num_bytes: the total buffer size in bytes (excludes workspace), if set, overrides MoE-based calculation.
                Must be aligned to 2 MB (``get_elastic_buffer_alignment()``).
            num_max_tokens_per_rank: the maximum number of tokens per rank, used for buffer size calculation.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            num_experts_per_rank: the fixed number of expert slots on every rank, in every live world: the expert count
                of a world is its rank count times this, and slot `e` belongs to rank `e // num_experts_per_rank` of the
                live world's packed numbering. Required (positive) for `dispatch` / `combine`; 0 for a buffer that only
                runs other operations.
            deterministic: whether to use deterministic routing algorithms.
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            num_cpu_timeout_secs: CPU-side timeout in seconds for CPU sync.
            num_gpu_timeout_secs: GPU-side timeout in seconds for GPU operations.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
            max_num_scaleout_ranks: the maximum number of scaleout ranks (NVLink domains) the buffer, the handles and
                the kernels are sized for; 0 = the cohort's validated layout. The live world is any product subset of it.
            max_num_scaleup_ranks: likewise for the positions per domain, at most 128 (`kNumMaxScaleupRanksCap`).
            num_nixl_channels: UCX device channels per peer of the NIXL agent; 0 = `NIXL_EP_NIXL_NUM_CHANNELS` or 64 (UCX's maximum).
            scaleout_rank_idx: this rank's NVLink domain in the maximum world (default: the cohort's validated layout).
            scaleup_rank_idx: this rank's position inside its domain (default: the cohort's validated layout).
            store: an explicit `torch.distributed.Store`: the default of `publish` and `connect_ranks`, and with
                `bootstrap_id` the Store the cohort bootstraps through (else the group's first rank serves a cohort-private TCPStore).
            connect_group: connect the whole cohort and commit a topology with all of it live (default); with `False`
                this rank alone is live (in the mode the host derives for one domain) until `connect_ranks` +
                `stage_topology` + `commit_topology`.
            rank_idx: this rank's index in the constructing cohort (required without a group; not the dense rank id).
            num_ranks: the size of the constructing cohort (required without a group; a joiner uses a one-rank cohort).
            num_group_scaleup_ranks: the cohort's declared NVLink-domain size (must divide `num_ranks`; contiguous blocks
                validated as one domain each; `num_ranks` = one domain); omitted: inferred from the GPU evidence.
            bootstrap_id: the cohort's namespace in `store` (required with an explicit bootstrap Store; distinct per
                concurrent cohort, fresh after a partial construction or a change of the cohort's membership).
            store_timeout_secs: how long the bootstrap waits for the cohort at each of its Store steps.
            num_zero_copy_slots: slots of the zero-copy output arena (0 = none, at most 8; the same on every rank), a
                direct-mode feature (`dispatch(zero_copy=True)` is refused while the topology is hybrid). A slot is
                overwritten by the next dispatch into it: issue that dispatch after the slot's last reader (the GEMM,
                and the combine through the handle); the returned tensors are views of the slot.
            zc_use_fp8_dispatch: with a zero-copy arena, lay the arena out for the FP8 wire (rows of `hidden` bytes plus their
                scaling factors); the arena is BF16 by default. A zero-copy setting only: the staged dispatch takes its wire from
                the payload, and `use_fp8_dispatch` above stays the review tip's size argument, which never touches the arena.
            zc_use_nvfp4_dispatch: with a zero-copy arena, lay the arena out for the NVFP4 wire (`hidden // 2` code bytes and
                `hidden // 64` scale packs per row, `hidden % 64 == 0`); the two `zc_` flags exclude each other (also without an
                arena, where neither lays anything out: `dispatch` accepts any wire's payload on any buffer).
            dynamic_take: the dynamic token take of the direct-mode kernels, OFF by default (`False`): the direct paths keep
                their static token partition unless the caller asks for the take. `True` turns it on (measured faster or equal
                at every batch size and active SM count on 1x16, where the static shares meet the per-SM egress arbitration
                tiers); `None` = automatic, on in direct mode. The hybrid paths' division is pinned by the notify count protocol
                and ignores the setting. `NIXL_EP_DYNAMIC_TAKE` (`auto`, `1`, `0`) overrides the argument for A/B runs.
            acquire_flags: the arrival-flag protocol of the scale-out receivers (the forwarder and the combine's reduce epilogue),
                OFF by default: relaxed polls of a record's per-packet flags followed by one acquire-release fence before the record
                is read, the review tip's protocol. `True` polls with acquire loads and drops the fence (the forwarder keeps only
                the proxy fence its bulk load needs): the dispatch at 4x8 measured -10 to -13 % at 16-128 tokens, -5 % at 1024,
                -2 % at 4096, the combine unchanged. Compile-time in the kernels (one instantiation per setting). `NIXL_EP_ACQUIRE_FLAGS`
                (`1`, `0`) overrides the argument for A/B runs.
        """
        assert num_zero_copy_slots >= 0, 'num_zero_copy_slots must not be negative'
        if num_zero_copy_slots > 0:
            assert not deterministic, 'Zero copy is not available with deterministic mode'
            assert num_max_tokens_per_rank > 0 and hidden > 0 and num_topk > 0 and num_experts_per_rank > 0, 'num_zero_copy_slots > 0 needs the MoE sizes'

        # Some useful utilities
        self.group = group
        if group is not None:
            if rank_idx not in (None, group.rank()) or num_ranks not in (None, group.size()):
                raise ValueError('rank_idx / num_ranks do not match the supplied process group')
            rank_idx, num_ranks = group.rank(), group.size()
        if type(rank_idx) is not int or type(num_ranks) is not int or not 0 <= rank_idx < num_ranks:
            raise ValueError('Without a process group, rank_idx and a positive num_ranks are required')
        self.rank_idx, self.num_ranks = rank_idx, num_ranks
        assert num_experts_per_rank >= 0, 'num_experts_per_rank must not be negative'
        self.num_experts_per_rank = num_experts_per_rank
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.deterministic = deterministic
        self.store = store

        # Coordinate the cohort through a Store: the NVLink-domain layout, the session token, the peer metadata, the barrier
        bootstrap_store, namespace = resolve_bootstrap_store(group, store, bootstrap_id, 'elastic-buffer', timeout_seconds=store_timeout_secs)
        self._nixl_bootstrap = NixlBootstrap(bootstrap_store, self.rank_idx, self.num_ranks, namespace=namespace,
                                             num_scaleup_ranks=num_group_scaleup_ranks, timeout_seconds=store_timeout_secs)
        num_group_scaleout_ranks = self._nixl_bootstrap.num_scaleout_ranks
        num_group_scaleup_ranks = self._nixl_bootstrap.num_scaleup_ranks

        # The declared maximum world and this rank's coordinates in it: explicit values win, else the cohort's layout
        max_num_scaleout_ranks = max_num_scaleout_ranks or num_group_scaleout_ranks
        max_num_scaleup_ranks = max_num_scaleup_ranks or num_group_scaleup_ranks
        scaleout_rank_idx = value_or(scaleout_rank_idx, self.rank_idx // num_group_scaleup_ranks)
        scaleup_rank_idx = value_or(scaleup_rank_idx, self.rank_idx % num_group_scaleup_ranks)

        # Calculate buffer size (already 2 MB-aligned from hint functions / calculate_elastic_buffer_size)
        if num_bytes is None:
            # NOTES: we allow `num_topk == 0`, as the buffer size can also be calculated by number of ranks (maybe bigger though)
            num_bytes = _C.calculate_elastic_buffer_size(
                max_num_scaleout_ranks, max_num_scaleup_ranks,
                num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
                allow_multiple_reduction)

        # The arena's wire: bytes per row and scale-factor packs per row (what the kernels move), from the constructor's `zc_` flags
        assert not (zc_use_fp8_dispatch and zc_use_nvfp4_dispatch), 'zc_use_fp8_dispatch and zc_use_nvfp4_dispatch exclude each other'
        if zc_use_nvfp4_dispatch:
            assert num_zero_copy_slots == 0 or hidden % 64 == 0, 'The NVFP4 wire needs hidden % 64 == 0'
            zero_copy_num_hidden_bytes, zero_copy_num_sf_packs = hidden // 2, hidden // 64
        elif zc_use_fp8_dispatch:
            zero_copy_num_hidden_bytes, zero_copy_num_sf_packs = hidden, align(hidden, 128) // 128
        else:
            zero_copy_num_hidden_bytes, zero_copy_num_sf_packs = hidden * 2, 0
        num_zero_copy_bytes = _C.calculate_zero_copy_arena_size(
            max_num_scaleout_ranks, max_num_scaleup_ranks,
            num_max_tokens_per_rank, zero_copy_num_hidden_bytes, zero_copy_num_sf_packs,
            num_topk, num_experts_per_rank, 1, False, False, num_zero_copy_slots) if num_zero_copy_slots > 0 else 0

        if os.environ.get('NIXL_EP_BUFFER_DEBUG', 0):
            arena = f'(plus a {num_zero_copy_bytes}-byte zero-copy arena of {num_zero_copy_slots} slot(s)) ' if num_zero_copy_slots > 0 else ''
            print(f'Initializing EP elastic buffer with {num_bytes} bytes {arena}'
                  f'at cohort rank {self.rank_idx}/{self.num_ranks}')
        self.num_bytes = num_bytes

        # Store default values
        self.num_max_tokens_per_rank = num_max_tokens_per_rank

        # The arrival flags of the RDMA records (`TokenLayout`) need strict PCIe ordering on the receiving region and one
        # flag per packet, so the path MTU must be the 4096 bytes of `kNumChunkBytes` (UCX would otherwise take the port's)
        assert os.environ.setdefault('UCX_IB_PCI_RELAXED_ORDERING', 'n') == 'n', \
            'UCX_IB_PCI_RELAXED_ORDERING must be n for the arrival flags of the RDMA records'
        assert os.environ.setdefault('UCX_IB_PATH_MTU', '4096') == '4096', \
            'UCX_IB_PATH_MTU must be 4096: an RDMA record carries one arrival flag per 4096-byte packet'

        # UCX only flags a closed UD endpoint (the wire-up lane between hosts) as disconnected and keeps it in the
        # interface's connection-match table for UCX_UD_LINGER_TIMEOUT (UCX default 5 minutes); a connection request
        # from the same interface address (LID, QPN) and connection sequence number is matched to it and REUSES it
        # (`uct_ud_ep_rx_creq`, "stale EP reuse"). A replacement process on a departed peer's host that is handed the
        # departed queue pair then wires up against a dead endpoint and never connects (4 of 18 cross-host rejoins;
        # 0 of 22 with 1 s). One second covers the in-flight packets the linger exists for; no process is replaced faster
        os.environ.setdefault('UCX_UD_LINGER_TIMEOUT', '1s')

        # NIXL: the UCX device channels of the agent (a warp is a channel; UCX's worker address carries at most 64)
        num_nixl_channels = num_nixl_channels or int(os.getenv('NIXL_EP_NIXL_NUM_CHANNELS', 64))
        assert 1 <= num_nixl_channels <= 64, f'UCX connects at most 64 device channels per peer, got {num_nixl_channels}'

        # Dynamic token take of the direct paths: False (default) off, True on, None automatic; the environment overrides for A/B runs
        env_take = os.getenv('NIXL_EP_DYNAMIC_TAKE')
        if env_take is not None:
            dynamic_take = None if env_take.strip().lower() == 'auto' else bool(int(env_take))
        self.dynamic_take = dynamic_take
        dynamic_take_mode = 2 if dynamic_take is None else int(bool(dynamic_take))

        # Arrival-flag protocol: False (default) relaxed polls + acq_rel fence, True acquire polls; the environment overrides for A/B runs
        env_acquire = os.getenv('NIXL_EP_ACQUIRE_FLAGS')
        if env_acquire is not None:
            acquire_flags = bool(int(env_acquire))
        self.acquire_flags = bool(acquire_flags)

        # Create CPP handle
        self.explicitly_destroy = explicitly_destroy
        self.runtime = _C.ElasticBuffer(num_bytes,
                                        allow_multiple_reduction,
                                        prefer_overlap_with_compute,
                                        num_cpu_timeout_secs, num_gpu_timeout_secs,
                                        self.explicitly_destroy,
                                        max_num_scaleout_ranks, max_num_scaleup_ranks,
                                        scaleout_rank_idx, scaleup_rank_idx,
                                        num_nixl_channels, self._nixl_bootstrap.session_token,
                                        num_experts_per_rank, num_zero_copy_bytes, dynamic_take_mode, self.acquire_flags)

        if num_zero_copy_slots > 0:
            self.runtime.configure_zero_copy_arena(num_zero_copy_slots, num_max_tokens_per_rank, hidden,
                                                   zero_copy_num_hidden_bytes, zero_copy_num_sf_packs, num_topk,
                                                   1, False, False)

        # Adaptive kernels: the fixed launch grid; the active SM count is set per topology (`dispatch(num_sms=...)` reaches
        # it through `set_num_active_sms`)
        self.num_launch_sms = _C.get_adaptive_num_launch_sms()
        self.max_num_scaleout_ranks, self.max_num_scaleup_ranks = max_num_scaleout_ranks, max_num_scaleup_ranks

        # Logical rank indices
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()
        self.scaleout_rank_idx, self.scaleup_rank_idx = scaleout_rank_idx, scaleup_rank_idx
        self.rank_id = self.rank_id_of(scaleout_rank_idx, scaleup_rank_idx)
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()

        # Connect the whole cohort and make it the live world (the constructor is collective over the cohort)
        if connect_group:
            infos = self._gather_group_peer_infos()
            peers = [rid for rid in infos if rid != self.rank_id]
            self.connect_ranks(peers, [infos[rid][0] for rid in peers], [infos[rid][1] for rid in peers])
            # The mode follows the cohort's domains (`mode_of`): direct iff the cohort is one NVLink domain
            self.stage_topology(num_active_sms=self.num_launch_sms, active_ranks=sorted(infos))
            self.commit_topology()

        # Call a barrier to ensure initialization visibility for all peers
        torch.cuda.synchronize()
        self._nixl_bootstrap.barrier('ready')
        torch.cuda.synchronize()

    def _gather_group_peer_infos(self) -> Dict[int, Tuple[bytes, Tuple[int, int, int, int]]]:
        # Every cohort member's `(metadata, peer info)` by dense rank id; explicit coordinates must keep the physical domains
        mine = (self.rank_id, self.get_local_metadata(), self.get_local_peer_info())
        infos = self._nixl_bootstrap.all_gather_peer_infos(mine)
        pairs = {(rid // self.max_num_scaleup_ranks, rank // self._nixl_bootstrap.num_scaleup_ranks)
                 for rank, rid in enumerate(infos)}
        if len({d for d, _ in pairs}) != len(pairs) or len({p for _, p in pairs}) != len(pairs):
            raise ValueError('Explicit rank coordinates disagree with the validated NVLink domain layout')
        return infos

    def destroy(self) -> None:
        """
        Destroy the C++ runtime and release resources. Requires `explicitly_destroy=True` at construction.
        """
        assert self.explicitly_destroy

        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None  # Cannot use anymore

    @property
    def has_zero_copy(self) -> bool:
        """Whether this buffer has a zero-copy output arena (constructed with `num_zero_copy_slots > 0`)."""
        return self.runtime.has_zero_copy()

    @property
    def num_zero_copy_slots(self) -> int:
        """The number of slots of the zero-copy arena (0 without one)."""
        return self.runtime.get_zero_copy_num_slots()

    @staticmethod
    def get_buffer_size_hint(group: dist.ProcessGroup,
                             num_max_tokens_per_rank: int, hidden: int,
                             num_topk: int = 0, use_fp8_dispatch: bool = False,
                             allow_multiple_reduction: bool = True,
                             max_num_scaleout_ranks: int = 0, max_num_scaleup_ranks: int = 0,
                             num_group_scaleup_ranks: Optional[int] = None, store: Optional[dist.Store] = None,
                             bootstrap_id: Optional[str] = None, store_timeout_secs: int = 300) -> int:
        """
        Get a recommended buffer size (in bytes) for the given MoE settings, without constructing the buffer.
        The returned value is aligned to 2 MB. A missing maximum is discovered from the group's layout (a Store exchange).

        Arguments:
            group: the process group whose layout fills in a missing maximum (`None` with both maxima given).
            num_max_tokens_per_rank: the maximum number of tokens per rank.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to use FP8 for dispatch.
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            max_num_scaleout_ranks: the declared maximum world to size for (0 = the group's domain count).
            max_num_scaleup_ranks: likewise (0 = the group's domain size).
            num_group_scaleup_ranks: the group's declared domain size, as the constructor's argument of the same name.
            store: with `bootstrap_id`, the Store the group's layout is discovered through (else the group's own Store).
            bootstrap_id: the namespace in `store`.
            store_timeout_secs: the Store timeout of the discovery.

        Returns:
            size: the recommended buffer size in bytes (2 MB-aligned).
        """
        if max_num_scaleout_ranks == 0 or max_num_scaleup_ranks == 0:
            group_domains, group_positions = get_logical_domain_size(group, num_scaleup_ranks=num_group_scaleup_ranks,
                store=store, bootstrap_id=bootstrap_id, store_timeout_secs=store_timeout_secs)
            max_num_scaleout_ranks = max_num_scaleout_ranks or group_domains
            max_num_scaleup_ranks = max_num_scaleup_ranks or group_positions
        # NOTES: calculate_elastic_buffer_size already returns 2 MB-aligned values
        return _C.calculate_elastic_buffer_size(
            max_num_scaleout_ranks, max_num_scaleup_ranks,
            num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
            allow_multiple_reduction)

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
        """
        Perform a GPU-level barrier across all ranks, optionally with CPU synchronization.

        Arguments:
            use_comm_stream: whether to use the communication stream (otherwise uses the current compute stream).
            with_cpu_sync: whether to also call `cudaDeviceSynchronize` before and after the barrier.
        """
        self.runtime.barrier(use_comm_stream, with_cpu_sync)

    @staticmethod
    def capture() -> EventHandle:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event_handle: the captured event handle.
        """
        return EventHandle()

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

    def get_physical_domain_size(self) -> Tuple[int, int]:
        """
        Get the physical domain sizes (RDMA ranks and NVLink ranks) of the declared maximum world.

        Returns:
            num_rdma_ranks: the number of physical RDMA ranks.
            num_nvlink_ranks: the number of physical NVLink ranks.
        """
        return self.runtime.get_physical_domain_size()

    def get_logical_domain_size(self) -> Tuple[int, int]:
        """
        Get the logical domain sizes (scaleout ranks and scaleup ranks) of the declared maximum world.

        Returns:
            num_scaleout_ranks: the number of logical scaleout ranks.
            num_scaleup_ranks: the number of logical scaleup ranks.
        """
        return self.runtime.get_logical_domain_size()

    def get_theoretical_num_sms(self, num_topk: int,
                                num_scaleout_topk: int = 0,
                                rdma_gbs: float = 0, nvlink_gbs: float = 0,
                                # TODO: use different values for other architectures
                                sm_read_gbs: float = 200, sm_write_gbs: float = 50) -> int:
        """
        Estimate the optimal number of SMs for dispatch/combine kernels based on bandwidth modeling.
        The result is cached per live world. This assumes a balanced gate distribution.
        The traffic model is of the live world: the live domain and position counts of the committed topology descriptor
        (not its bounds: a single live domain above index 0 has a scale-out bound above 1 but no RDMA traffic), with an
        expert count of the live rank count times `num_experts_per_rank`.

        Arguments:
            num_topk: the number of top-k experts per token.
            num_scaleout_topk: reserved for balanced gate (must be 0 currently).
            rdma_gbs: the RDMA bandwidth in GB/s (0 for auto-detect).
            nvlink_gbs: the NVLink bandwidth in GB/s (0 for auto-detect).
            sm_read_gbs: the per-SM HBM read bandwidth in GB/s.
            sm_write_gbs: the per-SM HBM write bandwidth in GB/s.

        Returns:
            num_sms: the recommended SM count (even, at least 4).
        """
        scaleout_mask, scaleup_mask = self.get_topology_masks()
        num_scaleout_ranks, num_scaleup_ranks = bin(scaleout_mask).count('1'), bin(scaleup_mask).count('1')
        assert self.num_experts_per_rank > 0, 'Set num_experts_per_rank when constructing an EP buffer'
        return self._get_theoretical_num_sms(num_topk, num_scaleout_ranks, num_scaleup_ranks,
                                             num_scaleout_topk, rdma_gbs, nvlink_gbs, sm_read_gbs, sm_write_gbs)

    @weak_lru(maxsize=None)
    def _get_theoretical_num_sms(self, num_topk: int,
                                 num_scaleout_ranks: int, num_scaleup_ranks: int,
                                 num_scaleout_topk: int, rdma_gbs: float, nvlink_gbs: float,
                                 sm_read_gbs: float, sm_write_gbs: float) -> int:
        num_ranks = num_scaleout_ranks * num_scaleup_ranks
        # Fixed expert slots per rank: the expert count of the live world
        num_experts = num_ranks * self.num_experts_per_rank
        # TODO: support `do_expand` and `allow_multiple_reduction`

        # The `1` in this function means scale-up traffic
        # i.e. the HBM read volume of the dispatch copy epilogue, equals to "the number of tokens" * "num_expected_topk" * "data size per token"
        # NOTES: this is for balanced gate
        # For V3.0's group-limited gate, please do not use this function
        # TODO: support this
        assert num_scaleout_topk == 0

        # Get bandwidth
        # NOTES: a single live domain has no RDMA traffic to model, so the RDMA probe is skipped
        if rdma_gbs == 0 and num_scaleout_ranks > 1:
            rdma_gbs = get_rdma_gbs()
        if nvlink_gbs == 0:
            nvlink_gbs = get_nvlink_gbs()

        # Initial count
        # NOTES: we don't count HBM traffic
        sm_read, sm_write = 0, 0
        rdma_traffic, nvlink_traffic = 0, 0

        def get_expected_topk(num_groups: int) -> float:
            assert num_experts % num_groups == 0
            # A fixed-width top-k is padded with -1 when the live slot count is below it
            effective_topk = min(num_topk, num_experts)
            return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, effective_topk) / math.comb(num_experts, effective_topk))

        # The traffic model follows the live domain count, which is what decides the mode (hybrid across NVLink domains,
        # direct inside one); the `NIXL_EP_ADAPTIVE_MODE` test override on one domain does not change it

        # Expected top-k scale-out ranks
        num_expected_scaleout_topk = get_expected_topk(num_scaleout_ranks) if num_scaleout_ranks > 1 else 0

        # Expected top-k scale-up ranks
        num_expected_topk = get_expected_topk(num_ranks)

        # Read tokens
        sm_read += 1 / num_expected_topk

        # NOTES: we don't consider the skip-send-buffer cases (all selections fall in the local)
        if num_scaleout_ranks > 1:
            # Scaleup warps: write send buffer
            sm_write += 1 / num_expected_topk

            # Scaleout traffic
            sm_write += (1 / num_expected_topk) * (num_expected_scaleout_topk / num_scaleout_ranks)  # Local bypass
            rdma_traffic += (1 / num_expected_topk) * (num_expected_scaleout_topk * (1 - 1 / num_scaleout_ranks))

            # Forward warps
            sm_read += num_expected_scaleout_topk / num_expected_topk
            sm_write += 1  # Issue scaleup
            nvlink_traffic += 1 - (1 / num_scaleup_ranks)
        else:
            # Issue NVLink (one live domain: every peer is an NVLink peer, no send buffer)
            sm_write += num_scaleup_ranks / num_ranks

            # NVLink traffic
            nvlink_traffic += num_scaleup_ranks / num_ranks * (1 - 1 / num_scaleup_ranks)  # Except local bypass

        # Found the bounded one
        if num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):
            bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
        else:
            bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs

        # Calculate SM count
        # NOTES: will try to use more SMs if not overlap with compute
        num_device_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
        num_sms = num_device_sms  # No traffic, e.g., EP=1
        if bounded_traffic > 0:
            num_sms = max(
                bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
                bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
            )
        num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)
        num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)
        num_sms = min(num_sms, num_device_sms)

        # Summary
        if os.environ.get('NIXL_EP_BUFFER_DEBUG', 0):
            print(f'EP SM approximation: '
                  f'{sm_read=}, {sm_write=}, {rdma_traffic=}, {nvlink_traffic=}, '
                  f'{rdma_gbs=}, {nvlink_gbs=}, '
                  f'{num_expected_scaleout_topk=}, {num_expected_topk=}, '
                  f'{bounded_traffic=}, {bounded_gbs=}, {num_sms=}')
        return num_sms

    def get_local_metadata(self) -> bytes:
        """This rank's NIXL agent metadata, which a peer loads to address this rank (`connect_ranks`)."""
        return self.runtime.get_local_metadata()

    def get_local_peer_info(self) -> Tuple[int, int, int, int]:
        """
        `(region base address, CUDA device index, rank id, allocation bytes)` of this rank's symmetric region.
        """
        return self.runtime.get_local_peer_info()

    def publish(self, store: Optional[dist.Store] = None, key_prefix: str = 'EP_NIXL') -> None:
        """
        Publish this rank's transport metadata to a `torch.distributed.Store` under `{key_prefix}/{rank_id}`, for ranks
        outside this cohort to `connect_ranks` to it (same Store and `key_prefix` on both sides).

        Arguments:
            store: the publication Store; defaults to the constructor's explicit `store` (never the bootstrap Store).
            key_prefix: the key prefix the readers pass to `connect_ranks`.
        """
        store = value_or(store, self.store)
        if store is None:
            raise ValueError('Peer publication requires an explicit Store; pass store to ElasticBuffer or publish(store=...)')
        store.set(f'{key_prefix}/{self.rank_id}', pickle.dumps((self.get_local_metadata(), self.get_local_peer_info())))

    def connect_ranks(self, rank_ids: List[int], remote_mds: Optional[List[bytes]] = None,
                      peer_infos: Optional[List[Tuple[int, int, int, int]]] = None,
                      store: Optional[dist.Store] = None, key_prefix: str = 'EP_NIXL',
                      timeout_secs: float = 120.0) -> None:
        """
        Load peers' transport metadata and stage new memory views (made live by the next `commit_topology`), from
        `remote_mds` and `peer_infos` as published by the peers, or from a `store` they `publish`ed to. Runs between
        launches, never inside a CUDA-graph capture (the per-peer barrier words are reset and the stream synchronised).

        Arguments:
            rank_ids: dense rank ids of the peers (not this rank, not connected yet).
            remote_mds: the peers' `get_local_metadata()`, aligned with `rank_ids`.
            peer_infos: the peers' `get_local_peer_info()`, aligned with `rank_ids`.
            store: read `{key_prefix}/{rank_id}` from this store instead (default: the constructor's explicit `store`).
            key_prefix: the publisher's key prefix.
            timeout_secs: how long to wait for each peer's key in `store` (the wire-up has its own `NIXL_EP_NIXL_CONNECT_TIMEOUT_SECS`).
        """
        if len(rank_ids) == 0:
            return
        if remote_mds is None or peer_infos is None:
            store = value_or(store, self.store)
            if store is None:
                raise ValueError('Peer discovery requires an explicit Store; pass store to ElasticBuffer or '
                                 'connect_ranks(store=...), or pass remote_mds and peer_infos')
            remote_mds, peer_infos = [], []
            for rid in rank_ids:
                key = f'{key_prefix}/{rid}'
                store.wait([key], timedelta(seconds=timeout_secs))
                md, info = pickle.loads(store.get(key))
                remote_mds.append(md), peer_infos.append(info)
        self.runtime.connect_ranks(list(rank_ids), list(remote_mds), [tuple(info) for info in peer_infos])

    def disconnect_ranks(self, rank_ids: List[int]) -> None:
        """
        Forget peers whose process goes away (staged; dropped at the next `commit_topology`). Commit a topology without
        the peer first. A rank that stays alive but sits a world out is left out of `active_ranks` instead.
        """
        self.runtime.disconnect_ranks(list(rank_ids))

    def get_connected_ranks(self) -> List[int]:
        """Dense rank ids of the connected peers (this rank excluded)."""
        return list(self.runtime.get_connected_ranks())

    def rank_id_of(self, scaleout_rank_idx: int, scaleup_rank_idx: int) -> int:
        """The dense rank id of coordinates `(scaleout_rank_idx, scaleup_rank_idx)` in the declared maximum world."""
        return scaleout_rank_idx * self.max_num_scaleup_ranks + scaleup_rank_idx

    def masks_of(self, active_ranks: Sequence[int]) -> Tuple[int, int]:
        """
        The `(scaleout_mask, scaleup_mask)` of a set of dense rank ids, which must be a product of domains and positions.
        """
        active = sorted(set(int(r) for r in active_ranks))
        assert len(active) > 0, 'The live world is empty'
        domains = sorted(set(r // self.max_num_scaleup_ranks for r in active))
        positions = sorted(set(r % self.max_num_scaleup_ranks for r in active))
        product = sorted(self.rank_id_of(d, k) for d in domains for k in positions)
        assert product == active, \
            f'The live world must be a product of domains and positions; {domains} x {positions} != {active}'
        return sum(1 << d for d in domains), sum(1 << k for k in positions)

    @staticmethod
    def mode_of(scaleout_mask: int) -> str:
        """
        The kernel mode of a live world whose NVLink domains are `scaleout_mask` (bit d = domain d live): `direct` when
        exactly one domain is live, whatever its index (every live peer over NVLink), `hybrid` otherwise (RDMA between
        domains, forwarding inside them). The rule every `stage_topology` applies; never a caller input. `NIXL_EP_ADAPTIVE_MODE=hybrid`
        forces the hybrid kernel on one domain (tests, A/B runs); several live domains ignore the variable.
        """
        if bin(scaleout_mask).count('1') != 1:
            return 'hybrid'
        forced = os.environ.get('NIXL_EP_ADAPTIVE_MODE', 'direct')
        assert forced in ('direct', 'hybrid'), f'Unknown NIXL_EP_ADAPTIVE_MODE {forced}'
        return 'hybrid' if forced == 'hybrid' else 'direct'

    def _stage(self, mode: Optional[str], num_active_sms: int, active_ranks: Optional[Sequence[int]]) -> None:
        num_active_sms = num_active_sms or self.get_topology()[1]
        scaleout_mask, scaleup_mask = self.masks_of(active_ranks) if active_ranks is not None else self.get_topology_masks()
        if mode is None:
            mode = self.mode_of(scaleout_mask)
        # The position mask crosses the extension as 64-bit words (a whole NVL72 domain needs more than 64 bits)
        words = [(scaleup_mask >> (64 * i)) & ((1 << 64) - 1)
                 for i in range(max(1, (scaleup_mask.bit_length() + 63) // 64))]
        self.runtime.stage_topology_masks(1 if mode == 'hybrid' else 0, scaleout_mask, words, num_active_sms)

    def stage_topology(self, *, num_active_sms: int = 0, active_ranks: Optional[Sequence[int]] = None) -> None:
        """
        Prepare a new topology descriptor (`EPTopology`) on the host, validated here and made live by `commit_topology`.
        The mode is not an input: it follows the staged domains (`mode_of`: direct iff exactly one NVLink domain is live).
        A `set_num_active_sms` in between (a `dispatch` whose `num_sms` differs from the committed count calls it)
        recommits the committed world and discards the staged one: stage after it.

        Arguments:
            num_active_sms: the active blocks of the fixed grid (even, at most `num_launch_sms`; 0 = keep the current value).
            active_ranks: the dense rank ids of the live world, a product set (`masks_of`) of connected ranks; `None`
                keeps the committed live world.
        """
        self._stage(None, num_active_sms, active_ranks)

    def stage_topology_forced(self, mode: str, num_active_sms: int = 0, active_ranks: Optional[Sequence[int]] = None) -> None:
        """
        TESTS AND A/B RUNS ONLY: `stage_topology` with an explicit mode instead of the rule (`hybrid` is legal everywhere,
        `direct` needs exactly one live NVLink domain). It holds until the next `stage_topology`, which derives again,
        and survives `set_num_active_sms`.
        """
        assert mode in ('direct', 'hybrid'), f'mode must be direct or hybrid, got {mode!r}'
        self._stage(mode, num_active_sms, active_ranks)

    def commit_topology(self) -> None:
        """
        Make the staged topology descriptor (`EPTopology`) live: one stream-ordered device write on the current stream
        (a device synchronisation first when peers were connected or disconnected). Every participating rank commits
        the same topology descriptor (`EPTopology`) between two launches, never inside a CUDA-graph capture and never
        between a dispatch and its combine.
        With a zero-copy arena a change of the live world resets the arena's call-tagged state, so every rank of the
        group commits every world change, participant or not, while idle and under a CPU barrier after the commit.
        """
        self.runtime.commit_topology()

    def set_active_ranks(self, active_ranks: Sequence[int], num_active_sms: int = 0) -> None:
        """`stage_topology(active_ranks=...)` + `commit_topology()` in one step; the mode follows the new world."""
        self.stage_topology(num_active_sms=num_active_sms, active_ranks=active_ranks)
        self.commit_topology()

    def get_topology_masks(self) -> Tuple[int, int]:
        """
        The `(scaleout_mask, scaleup_mask)` of the committed topology descriptor (`EPTopology`): bit d / bit k = live.
        """
        scaleout_mask, words = self.runtime.get_topology_masks()
        return scaleout_mask, sum(int(w) << (64 * i) for i, w in enumerate(words))

    def get_active_ranks(self) -> List[int]:
        """The dense rank ids of the committed live world, sorted."""
        scaleout_mask, scaleup_mask = self.get_topology_masks()
        domains = [d for d in range(self.max_num_scaleout_ranks) if (scaleout_mask >> d) & 1]
        positions = [k for k in range(self.max_num_scaleup_ranks) if (scaleup_mask >> k) & 1]
        return sorted(self.rank_id_of(d, k) for d in domains for k in positions)

    def get_topology_ranks(self) -> Tuple[int, int]:
        """
        The bounds `(num_scaleout_ranks, num_scaleup_ranks)` of the committed topology descriptor (`EPTopology`).
        """
        return self.runtime.get_topology_ranks()

    @property
    def is_participant(self) -> bool:
        """
        Whether this rank is inside the live world of the committed topology descriptor (`EPTopology`); a rank outside
        it must not call `dispatch`/`combine` or replay a graph containing them.
        """
        return self.runtime.is_participant()

    def set_num_active_sms(self, num_active_sms: int) -> None:
        """
        Recommit the committed topology descriptor with a new active SM count (even, at most `num_launch_sms`); its world
        and its mode (derived, or forced by a test) are untouched. Collective: every rank, the same count, between launches
        and never inside a CUDA-graph capture. `dispatch(num_sms=...)` reaches it through `_adaptive_active_sms`.
        """
        mode, _ = self.get_topology()
        self.runtime.set_topology(1 if mode == 'hybrid' else 0, num_active_sms)

    def set_topology_forced(self, mode: str, num_active_sms: int = 0) -> None:
        """
        TESTS AND A/B RUNS ONLY: the committed world in an explicit mode, committed at once (the graph-retention flip);
        a pending staged world is discarded, as with `set_num_active_sms`.
        """
        assert mode in ('direct', 'hybrid'), f'mode must be direct or hybrid, got {mode!r}'
        self.runtime.set_topology(1 if mode == 'hybrid' else 0, num_active_sms or self.get_topology()[1])

    def get_topology(self) -> Tuple[str, int]:
        """
        The current (mode, num_active_sms) of the topology descriptor.
        """
        mode, num_active_sms = self.runtime.get_topology()
        return ('hybrid' if mode == 1 else 'direct'), num_active_sms

    @property
    def combine_is_hybrid(self) -> bool:
        """
        Whether combine reduces within the scale-up domain first and then across scale-out ranks (the hybrid
        recipe), as opposed to within a rank and then globally (the direct recipe). Decides the reference
        reduction order in the tests; follows the `mode` field of the topology descriptor (`EPTopology`).
        """
        return self.get_topology()[0] == 'hybrid'

    def _adaptive_active_sms(self, num_sms: int) -> int:
        # The grid of the adaptive kernels is fixed, so `num_sms` only decides how many blocks do work: clamp it to
        # the launch grid (even, at least 2) and update the topology descriptor (`EPTopology`, its `num_active_sms`) if it changed
        num_active_sms = max(2, min(align(num_sms, 2), self.num_launch_sms))
        _, current = self.get_topology()
        if current != num_active_sms:
            assert not torch.cuda.is_current_stream_capturing(), \
                f'`dispatch(num_sms={num_sms})` would change the committed active SM count ({current}) inside a CUDA-graph capture: warm up eagerly with the same `num_sms`'
            self.set_num_active_sms(num_active_sms)   # the mode is never touched by a dispatch
        return num_active_sms

    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: torch.Tensor,
                 topk_weights: Optional[torch.Tensor] = None,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 num_max_tokens_per_rank: Optional[int] = None,
                 expert_alignment: Optional[int] = None,
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
                 zero_copy: bool = False,
                 zero_copy_slot: int = 0) \
            -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     Optional[torch.Tensor], Optional[torch.Tensor],
                     EPHandle, EventOverlap]:
        """
        Dispatch tokens to different ranks. Supports both single-node and multi-node settings.
            The SM count is automatically determined if not specified.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be
                `[num_tokens, hidden]`, and type must be `torch.bfloat16`; for the second type (FP8 mode),
                the first element of the tuple must be `[num_tokens, hidden]` with type `torch.float8_e4m3fn`,
                the second is the scale factors. NVFP4 mode: the tuple `(x_fp4, x_sf)` with `x_fp4` `[num_tokens, hidden // 2]`
                `torch.uint8` (two e2m1 codes per byte, the low nibble the even element) and `x_sf` `[num_tokens, hidden // 16]`
                `torch.float8_e4m3fn` or `torch.uint8` (one block scale per 16 elements, row-major), both contiguous,
                `hidden % 64 == 0`. The dispatch moves the codes and the block scales as bytes and applies no scale: the
                per-tensor global scale travels nowhere, so every rank of a dispatch must quantise with the same per-layer
                global scale (the checkpoint's static input scale, identical on all ranks; the dispatch cannot check it).
                The received scales are linear per token (`[n, hidden // 16]` row-major): a GEMM that wants the swizzled
                block-scale layout interleaves them after the expert grouping.
                A `torch.uint8` `x` is always read as NVFP4 codes and needs the scales; there is no byte wire (a uint8
                payload without scales, or with FP8-shaped scales, is refused), and an FP8 payload must be
                `torch.float8_e4m3fn`.
            topk_idx: `[num_tokens, num_topk]` with `nixl.runtime.ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            cumulative_local_expert_recv_stats: `[num_experts_per_rank]` with `torch.int`, a cumulative expert count
                tensor for statistics, useful for online EP load balance monitoring.
            num_max_tokens_per_rank: the maximum number of tokens per rank (the constructor's value if not given).
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_sms: the number of SMs to use (0 for automatic via `get_theoretical_num_sms`). With the adaptive
                kernels this is the active block count of the fixed grid, clamped to `num_launch_sms` (even, at least 2).
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
            zero_copy: the outputs in slot `zero_copy_slot` of the arena, written in place over NVLink: a DIRECT-mode feature,
                refused (a host assertion, before any collective step) while the topology is hybrid; compact; a collective
                argument. The returned tensors and the handle's `recv_src_metadata` are views of the slot (the constructor's
                slot rule). A graph captured in direct mode and replayed after the world grew to two domains cannot be refused:
                the kernels then run the staged path and the copy epilogue writes the slot; recapture without zero copy.
            zero_copy_slot: which slot, `0 <= zero_copy_slot < num_zero_copy_slots`.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x` (NVFP4: `recv_fp4` `[n, hidden // 2]` uint8 and
                `recv_sf` `[n, hidden // 16]` float8_e4m3fn, row-major)
            recv_topk_idx: received expert indices
            recv_topk_weights: received expert weights (`None` if `topk_weights` was not provided).
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        check_torch_deterministic()

        if zero_copy:
            assert self.has_zero_copy, 'Zero copy needs an arena: construct with `num_zero_copy_slots > 0`'
            assert self.get_topology()[0] == 'direct', \
                'Zero copy is a direct-mode feature (one NVLink domain): not available while the topology is hybrid'
            assert not self.deterministic, 'Zero copy is not available with deterministic mode'
            assert not do_expand and not use_tma_aligned_col_major_sf, 'Zero copy is the compact layout with row-major scaling factors'
            assert 0 <= zero_copy_slot < self.num_zero_copy_slots, f'zero_copy_slot {zero_copy_slot} is outside the arena of {self.num_zero_copy_slots} slot(s)'

        # Fixed expert slots per rank: the kernels derive the expert count from the live world
        assert self.num_experts_per_rank > 0, 'Set num_experts_per_rank when constructing an EP buffer'

        # Automatic decide SM count
        num_topk = topk_idx.shape[1]
        num_sms = self.get_theoretical_num_sms(num_topk) if num_sms == 0 else num_sms
        # Fixed grid: `num_sms` becomes the active block count of the topology descriptor
        num_sms = self._adaptive_active_sms(num_sms)

        # Unpack SF
        x, sf = x if isinstance(x, tuple) else (x, None)

        # NVFP4: the e4m3 block scales travel as int32 packs of four (the kernels' SF unit), a free view of the row-major tensor
        nvfp4 = sf is not None and x.dtype == torch.uint8
        if nvfp4:
            assert sf.dtype in (torch.float8_e4m3fn, torch.uint8) and sf.shape == (x.shape[0], x.shape[1] // 8) \
                and x.shape[1] % 32 == 0 and sf.is_contiguous(), \
                f'NVFP4 dispatch takes (codes [tokens, hidden // 2] uint8, scales [tokens, hidden // 16] float8_e4m3fn or uint8), ' \
                f'contiguous, hidden % 64 == 0; got {tuple(x.shape)} {x.dtype} and {tuple(sf.shape)} {sf.dtype}'
            sf = sf.view(torch.int32)

        # Some default values
        num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, self.num_max_tokens_per_rank)
        expert_alignment = value_or(expert_alignment, 1)

        # Do dispatch
        (recv_x, recv_sf,
         recv_topk_idx, recv_topk_weights,
         cloned_topk_idx,
         num_recv_tokens, num_expanded_tokens,
         num_recv_tokens_per_expert_list,
         psum_num_recv_tokens_per_scaleup_rank,
         psum_num_recv_tokens_per_expert,
         num_unaligned_recv_tokens_per_expert,
         recv_src_metadata,
         token_metadata_at_forward,
         channel_linked_list,
         event) = self.runtime.dispatch(x, sf, topk_idx, topk_weights,
                                        cumulative_local_expert_recv_stats,
                                        num_max_tokens_per_rank,
                                        expert_alignment,
                                        num_sms,
                                        previous_event,
                                        previous_event_before_epilogue,
                                        async_with_compute_stream, allocate_on_comm_stream,
                                        do_handle_copy, do_cpu_sync, do_expand,
                                        do_zero_padding,
                                        use_tma_aligned_col_major_sf,
                                        zero_copy, zero_copy_slot)

        # Create handle
        handle = EPHandle(do_expand,
                          expert_alignment,
                          num_max_tokens_per_rank,
                          num_sms,
                          cloned_topk_idx if do_handle_copy else topk_idx,
                          num_recv_tokens, num_expanded_tokens,
                          num_recv_tokens_per_expert_list,
                          psum_num_recv_tokens_per_scaleup_rank,
                          psum_num_recv_tokens_per_expert,
                          num_unaligned_recv_tokens_per_expert,
                          recv_src_metadata,
                          token_metadata_at_forward,
                          channel_linked_list, recv_count_is_exact=do_cpu_sync)

        # Create event
        event_overlap = EventOverlap(event)

        # Deterministic epilogue
        # NOTES: when we change the metadata layout, the epilogue should also be changed
        if self.deterministic:
            epilogue = functools.partial(
                handle.deterministic_sort,
                do_cpu_sync,
                recv_x, recv_sf, recv_topk_idx, recv_topk_weights, channel_linked_list
            )
            event_overlap.register_hook_after_wait(epilogue) if async_with_compute_stream else epilogue()

        # Repack SF (NVFP4: the e4m3 view of the received packs; it shares the storage the deterministic sort permutes)
        if nvfp4:
            recv_sf = recv_sf.view(torch.float8_e4m3fn)
        recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x

        # Return
        return recv_x, recv_topk_idx, recv_topk_weights, handle, event_overlap

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    def combine(self,
                x: torch.Tensor,
                handle: EPHandle,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                num_sms: int = 0,
                previous_event: EventHandle = None,
                previous_event_before_epilogue: Optional[EventHandle] = None,
                async_with_compute_stream: bool = False,
                allocate_on_comm_stream: bool = False) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens from different ranks back to their original ranks.
        Supports both single-node and multi-node settings.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the `dispatch` function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float` for non-expand mode, or
                `[num_tokens]` 1D for expand mode. The tokens' top-k weights for reducing to
                its original ranks.
            bias: 0, 1 or 2 `[num_combined_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            num_sms: the number of SMs to use (0 to reuse the SM count from the dispatch handle).
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            previous_event_before_epilogue: the event to wait before actually executing the reduce epilogue.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            combined_topk_weights: the reduced top-k weights, with shape `[num_combined_tokens, num_topk]` and type `torch.float`.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        check_torch_deterministic()

        # Automatic decide SM count
        num_sms = handle.num_sms if num_sms == 0 else num_sms

        # Fixed expert slots per rank: the kernels derive the expert count from the live world (not the dispatch's)
        assert self.num_experts_per_rank > 0, 'Set num_experts_per_rank when constructing an EP buffer'

        bias_0, bias_1 = ElasticBuffer._unpack_bias(bias)
        combined_x, combined_topk_weights, event = \
            self.runtime.combine(x, topk_weights,
                                 bias_0, bias_1,
                                 handle.recv_src_metadata,
                                 handle.topk_idx,
                                 handle.psum_num_recv_tokens_per_scaleup_rank,
                                 handle.token_metadata_at_forward,
                                 handle.channel_linked_list,
                                 handle.recv_count_is_exact,
                                 handle.num_max_tokens_per_rank,
                                 num_sms,
                                 previous_event,
                                 previous_event_before_epilogue,
                                 async_with_compute_stream,
                                 allocate_on_comm_stream,
                                 handle.do_expand)
        return combined_x, combined_topk_weights, EventOverlap(event)
