# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0

import functools
import os
import uuid
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
from .. import _C

# noinspection PyUnresolvedReferences
from .._C import EventHandle
from ..utils.envs import check_torch_deterministic
from ..utils.event import EventOverlap
from ..utils.nixl_bootstrap import exchange_records
from ..utils.nixl_fabric import inspect_fabric_device
from ..utils.nixl_topology import (
    _MODE_NAMES,
    _mode_code,
    build_rank_mapping,
    masks_of,
    mode_of,
)


class EPHandle:
    """
    Communication handle returned by `ElasticBuffer.dispatch`.
    It is consumed by `ElasticBuffer.combine` to reverse the token routing.

    Attributes:
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        topk_idx: dispatch top-k stable expert IDs, `[num_tokens, num_topk]`; cloned when `do_handle_copy=True`.
        psum_num_recv_tokens_per_scaleup_rank: `[max_num_scaleup_ranks + 1]` inclusive prefix counts
            over the committed scaleup extent; unused entries before the fixed final total are zero.
            A token is counted once per rank even if multiple selected experts belong to that rank.
        psum_num_recv_tokens_per_expert: prefix sum of alignment-padded received token counts per local
            expert, shape `[num_experts_per_rank]`. Each expert's count is padded to `expert_alignment`.
            In non-expand mode, this is the inclusive prefix sum. In expand mode, `psum[i]` equals
            the aligned cumulative count of experts before `i` plus the actual (unaligned) token count
            of expert `i` — so `psum[i] - align(psum[i-1], expert_alignment)` recovers the real
            count for expert `i`, and `align(psum[i], expert_alignment)` gives expert `i+1`'s
            starting offset.
        num_recv_tokens_per_expert_list: CPU per-expert receive counts; empty without CPU synchronization.
        num_unaligned_recv_tokens_per_expert: the actual (unaligned) number of tokens received per local
            expert, shape `[num_experts_per_rank]` with `torch.int`, in both compact and expanded layouts.
        recv_src_metadata: source token indices and buffer slot indices.
        token_metadata_at_forward: per-channel forwarded token metadata (allocated in both modes, written in hybrid mode).
        channel_linked_list: per-channel per-scaleup-peer linked list (allocated in both modes, written in hybrid mode).
        num_recv_tokens: the actual receive count with CPU synchronization; otherwise the declared maximum world's capacity.
        num_recv_tokens_device: 0-d `torch.int` device view of the real count, valid without a CPU sync and in graphs.
    """

    def __init__(
        self,
        do_expand: bool,
        expert_alignment: int,
        num_max_tokens_per_rank: int,
        topk_idx: torch.Tensor,
        num_recv_tokens: int,
        num_recv_tokens_per_expert_list: list,
        psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
        psum_num_recv_tokens_per_expert: torch.Tensor,
        num_unaligned_recv_tokens_per_expert: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        token_metadata_at_forward: torch.Tensor,
        channel_linked_list: torch.Tensor,
    ):
        assert topk_idx is not None

        self.do_expand = do_expand
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
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

        self.num_recv_tokens = num_recv_tokens

    @property
    def _recv_count_is_exact(self) -> bool:
        # CPU sync returns one entry per expert slot, including zero counts.
        return bool(self.num_recv_tokens_per_expert_list)

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
            if not self._recv_count_is_exact
            else self.recv_src_metadata.shape[0]
        )
        if not self._recv_count_is_exact:
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
            if not self._recv_count_is_exact:
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


class ElasticBuffer:
    """
    The elastic communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine, using NVLink and/or RDMA)

    Attributes:
        store: the explicit `torch.distributed.Store` given to the constructor (the default of `connect_ranks`).
        allow_multiple_reduction: whether to allow multiple reductions in combine. If disabled,
            only one reduction will be done in the combine epilogue for best precision,
            but it may increase data transfer size.
        prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            Selects the kernel configuration for overlap without changing the launch grid.
        num_bytes: the allocated communication buffer size in bytes, excluding workspace.
        num_max_tokens_per_rank: the default maximum tokens per rank.
        max_num_scaleout_ranks: the declared maximum number of scaleout ranks (NVLink domains) the buffer is sized for.
        max_num_scaleup_ranks: the declared maximum number of scaleup ranks (positions per domain).
        world_rank: this rank's caller-assigned ID within the maximum world's capacity; independent of physical topology.
        num_launch_sms: the fixed main dispatch/combine block count, selected at construction.
        runtime: the C++ runtime.

    Elasticity: active ranks must form a supported product of discovered NVLink domains and internal positions. `connect_ranks` prepares peers, `set_active_ranks` activates membership, and
    `disconnect_ranks` excludes and retires departing peers. The kernels read the topology descriptor (`EPTopology`).
    Only active ranks launch dispatch/combine. Construction initializes this rank alone.
    A single active rank can run dispatch/combine before connecting peers, including warm-up and graph capture.
    New-peer connections may run while the committed graph replays. Serialize
    control operations on each buffer; a background setup thread must select the buffer's CUDA device.
    Activation (`set_active_ranks`) runs outside capture, between complete dispatch/combine pairs.
    Order topology commits and graph replays on the same CUDA stream, or establish the dependencies explicitly.
    Before activation or peer retirement, every rank of the old world, including departing ranks, must finish
    its queued CUDA work, call `barrier(with_cpu_sync=True)` to drain the transport, and rendezvous on the host before
    any peer departs. The update methods do not establish that completion boundary.
    Invalid inputs and mapping refusals before peer retirement preserve active views; failures after retirement or view activation are fatal.

    World ranks and public expert IDs index the declared maximum world. Internal source metadata and the
    kernels' `rank_idx` still index the committed extent: `scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx`.
    """

    def __init__(
        self,
        world_rank: int,
        max_num_scaleout_ranks: int,
        max_num_scaleup_ranks: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_experts_per_rank: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
        allow_multiple_reduction: bool = True,
        prefer_overlap_with_compute: bool = True,
        store: Optional[dist.Store] = None,
        num_cpu_timeout_secs: int = 300,
        num_gpu_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
        num_launch_sms: int = 24,
    ):
        """
        Initialize this rank's buffer without accessing a Store or waiting for peers.
        The initial topology contains this rank alone. Call `connect_ranks`, then `set_active_ranks` to join a world.
        Buffer capacity is calculated from the workload settings and declared maximum world.

        Arguments:
            world_rank: this rank's fixed caller-assigned ID, in `[0, max_num_scaleout_ranks * max_num_scaleup_ranks)`.
                It does not encode physical topology; the same IDs are passed to `connect_ranks`.
            max_num_scaleout_ranks: positive maximum number of NVLink domains, shared by all peers.
            max_num_scaleup_ranks: positive maximum positions per domain, at most 128, shared by all peers.
                These maxima size the buffer, handles and kernels; they do not describe current membership.
            num_max_tokens_per_rank: required positive maximum tokens per rank used for buffer sizing.
            hidden: required positive hidden dimension of each token.
            num_topk: required integer from 1 through 32; top-k capacity used for buffer sizing.
            use_fp8_dispatch: reserve smaller dispatch arenas for FP8/NVFP4 inputs; False reserves BF16 capacity and also
                accepts either quantized format. Must match across peers; it does not cast inputs.
            num_experts_per_rank: required positive integer giving the fixed physical routing slot capacity per GPU.
                Must match across peers, including joiners.
                Size it for the largest per-GPU expert placement across all intended topologies, not just the initial one.
                The caller supplies this fixed value; slots may remain unused. It is not the model's logical expert count.
                The caller assigns weights and maps logical experts to physical slots;
                the transport neither replicates experts nor creates weights.
            num_launch_sms: fixed main dispatch/combine block count (default 24; even, from 2 through 64).
                Must match across peers and remains unchanged through topology changes.
            deterministic: whether to sort received tokens deterministically (eager execution only).
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            store: the default Store for `connect_ranks`, not accessed during construction.
            num_cpu_timeout_secs: timeout for CPU synchronization, in seconds.
            num_gpu_timeout_secs: timeout for GPU operations, in seconds.
            explicitly_destroy: require explicit `destroy()` instead of releasing resources in the destructor.
        """
        if (
            any(
                type(value) is not int
                for value in (world_rank, max_num_scaleout_ranks, max_num_scaleup_ranks)
            )
            or max_num_scaleout_ranks <= 0
            or max_num_scaleup_ranks <= 0
            or not 0 <= world_rank < max_num_scaleout_ranks * max_num_scaleup_ranks
        ):
            raise ValueError(
                "world_rank must lie in the declared maximum world, whose dimensions must be positive integers"
            )
        if type(num_experts_per_rank) is not int or num_experts_per_rank <= 0:
            raise ValueError("num_experts_per_rank must be a positive integer")
        if (
            type(num_launch_sms) is not int
            or not 2 <= num_launch_sms <= 64
            or num_launch_sms % 2
        ):
            raise ValueError("num_launch_sms must be an even integer between 2 and 64")
        self._num_launch_sms = num_launch_sms
        self.num_experts_per_rank = num_experts_per_rank
        self.world_rank = world_rank
        self.max_num_scaleout_ranks = max_num_scaleout_ranks
        self.max_num_scaleup_ranks = max_num_scaleup_ranks
        self._prepared_mapping = {world_rank: 0}
        self._fabric_record = None
        self._fabric_records = {}
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.deterministic = deterministic
        self.store = store

        num_bytes = self.get_buffer_size_hint(
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
                f"Initializing EP elastic buffer with {num_bytes} bytes at world rank {world_rank}"
            )
        self.num_bytes = num_bytes
        self.num_max_tokens_per_rank = num_max_tokens_per_rank

        # Avoid matching replacement processes to closed UD wire-up endpoints.
        os.environ.setdefault("UCX_UD_LINGER_TIMEOUT", "1s")
        num_nixl_channels = int(os.getenv("NIXL_EP_NIXL_NUM_CHANNELS", 64))
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
            world_rank,
            num_nixl_channels,
            uuid.uuid4().hex,
            num_experts_per_rank,
            num_launch_sms,
        )
        # Complete local initialization before callers use the buffer on another stream.
        torch.cuda.synchronize()

    @property
    def num_launch_sms(self) -> int:
        """The fixed main dispatch/combine block count selected at construction."""
        return self._num_launch_sms

    def destroy(self) -> None:
        """Destroy the runtime; requires `explicitly_destroy=True` at construction."""
        assert self.explicitly_destroy

        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None  # Cannot use anymore

    @staticmethod
    def get_buffer_size_hint(
        max_num_scaleout_ranks: int,
        max_num_scaleup_ranks: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
        allow_multiple_reduction: bool = True,
    ) -> int:
        """
        Query the communication buffer size (excluding workspace) for the declared maximum world, without constructing a buffer.

        Arguments:
            max_num_scaleout_ranks: the maximum number of NVLink domains.
            max_num_scaleup_ranks: the maximum positions per domain.
            num_max_tokens_per_rank: the maximum number of tokens per rank.
            hidden: the hidden dimension of each token.
            num_topk: required integer from 1 through 32; top-k capacity used for buffer sizing.
            use_fp8_dispatch: size dispatch arenas for FP8/NVFP4 inputs; pass the same value to the constructor.
            allow_multiple_reduction: whether to allow multiple reductions in combine.

        Returns:
            size: the communication buffer size in bytes (2 MB-aligned).
        """
        if type(num_topk) is not int or not 1 <= num_topk <= 32:
            raise ValueError("num_topk must be an integer between 1 and 32")
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

    def connect_ranks(
        self,
        world_ranks: List[int],
        store: Optional[dist.Store] = None,
        key_prefix: str = "NIXL_EP",
        timeout_secs: float = 120.0,
    ) -> None:
        """
        Exchange agent names and region descriptors through a Store, fetch transport metadata through
        NIXL's metadata manager, and prepare memory views for activation by `set_active_ranks`.
        Discover NVLink domains and agree on internal rank coordinates without changing caller IDs.
        Existing and joining ranks connect reciprocally, including all intended new peers; established mappings must agree.
        Preparation preserves established coordinates and leaves the live mapping unchanged until activation.
        Already-connected peers are ignored. Publish discovery data under `{key_prefix}/{world_rank}`
        and delete it after all listed peers have read it and
        native preparation finishes. Complete each exchange on all participants before reusing its rank keys.
        Resets only new peers' barrier words on a setup stream. Native preparation releases the GIL so another
        Python thread can replay the committed graph; it does not change the live topology or active views.

        Arguments:
            world_ranks: world ranks of the peers (not this rank); only new peers participate in the exchange.
            store: the Store for reciprocal connection-data exchange (default: the constructor's explicit `store`).
            key_prefix: the same Store key prefix on both sides; separate buffers using the same rank IDs need separate prefixes or Stores.
            timeout_secs: the bound for each Store wait for discovery data or read acknowledgements;
                NIXL metadata fetch and wire-up share `NIXL_EP_NIXL_CONNECT_TIMEOUT_SECS`.
        """
        connected = set(self.get_connected_ranks())
        world_ranks = [rank for rank in world_ranks if rank not in connected]
        if not world_ranks:
            return
        store = self.store if store is None else store
        if store is None:
            raise ValueError(
                "Peer discovery requires an explicit Store; pass store to ElasticBuffer or "
                "connect_ranks(store=...)"
            )
        peers = set(world_ranks)
        if len(peers) != len(world_ranks):
            raise ValueError("Connect each new peer once")
        for rank in peers:
            if (
                not 0 <= rank < self.max_num_scaleout_ranks * self.max_num_scaleup_ranks
                or rank == self.world_rank
            ):
                raise ValueError(
                    "Connected ranks must be other ranks within the declared maximum world"
                )
        if self._fabric_record is None:
            self._fabric_record = inspect_fabric_device()
        known = {
            rank: evidence
            for rank, evidence in self._fabric_records.items()
            if rank in connected
        }
        known[self.world_rank] = self._fabric_record
        snapshot = {
            rank: slot
            for rank, slot in self._prepared_mapping.items()
            if connected and rank in known
        }
        with exchange_records(
            store,
            self.world_rank,
            world_ranks,
            (
                self.runtime.get_local_agent_name(),
                self.runtime.get_local_peer_info(),
                self._fabric_record,
                snapshot,
                known,
            ),
            key_prefix=key_prefix,
            timeout_seconds=timeout_secs,
        ) as records:
            snapshots = [snapshot]
            for _, _, _, peer_mapping, peer_records in records.values():
                snapshots.append(peer_mapping)
                for rank, evidence in peer_records.items():
                    if rank in known and known[rank] != evidence:
                        raise ValueError(
                            "Conflicting GPU discovery records; retire the old peer before reconnecting"
                        )
                    known[rank] = evidence
            for rank, (_, _, evidence, _, _) in records.items():
                if rank in known and known[rank] != evidence:
                    raise ValueError(
                        "Conflicting GPU discovery records; retire the old peer before reconnecting"
                    )
                known[rank] = evidence
            mapping = build_rank_mapping(
                known,
                snapshots,
                self.max_num_scaleout_ranks,
                self.max_num_scaleup_ranks,
            )
            world_to_slot = [-1] * (
                self.max_num_scaleout_ranks * self.max_num_scaleup_ranks
            )
            for rank, slot in mapping.items():
                world_to_slot[rank] = slot
            self.runtime.connect_ranks(
                list(world_ranks),
                [records[rank][0] for rank in world_ranks],
                [tuple(records[rank][1]) for rank in world_ranks],
                world_to_slot,
            )
            self._prepared_mapping = mapping
            self._fabric_records = known

    def disconnect_ranks(self, world_ranks: List[int]) -> None:
        """
        Exclude departing peers and retire their connections outside capture, after draining the old world and a host rendezvous.
        An active caller's remaining ranks must form a product of domains and positions; the mode follows that world.
        An inactive caller only retires connections, preserving its membership descriptor. An empty list does nothing.
        A live rank sitting a world out is only excluded with `set_active_ranks`.
        """
        world_ranks = list(world_ranks)
        if not world_ranks:
            return
        active = self.get_active_ranks()
        if self.world_rank in active:
            self._update_topology(
                active_ranks=sorted(set(active) - set(world_ranks)),
                disconnect_ranks=world_ranks,
            )
        else:
            self._update_topology(mode=self._get_mode(), disconnect_ranks=world_ranks)
        for rank in world_ranks:
            self._fabric_records.pop(rank, None)

    def get_connected_ranks(self) -> List[int]:
        """World ranks of the connected peers (this rank excluded)."""
        return list(self.runtime.get_connected_ranks())

    def _masks_of(self, active_ranks: Sequence[int]) -> Tuple[int, int]:
        return masks_of(
            active_ranks, self._prepared_mapping, self.max_num_scaleup_ranks
        )

    _mode_of = staticmethod(mode_of)

    def _update_topology(
        self,
        *,
        mode: Optional[str] = None,
        active_ranks: Optional[Sequence[int]] = None,
        disconnect_ranks: Sequence[int] = (),
    ) -> None:
        scaleout_mask, scaleup_mask = (
            self._masks_of(active_ranks)
            if active_ranks is not None
            else self._get_topology_masks()
        )
        if mode is None:
            mode = self._mode_of(scaleout_mask)
        # The position mask crosses the extension as 64-bit words (a whole NVL72 domain needs more than 64 bits)
        words = [
            (scaleup_mask >> (64 * i)) & ((1 << 64) - 1)
            for i in range(max(1, (scaleup_mask.bit_length() + 63) // 64))
        ]
        self.runtime.update_topology_masks(
            _mode_code(mode), scaleout_mask, words, disconnect_ranks
        )

    def set_active_ranks(self, active_ranks: Sequence[int]) -> None:
        """Validate and activate connected ranks forming a product of domains and positions; the mode follows the new world.

        Call after draining the old world and completing a host rendezvous.
        """
        self._update_topology(active_ranks=active_ranks)

    def _get_topology_masks(self) -> Tuple[int, int]:
        """Committed `(scaleout_mask, scaleup_mask)`: bit d / bit k denotes an active domain / position."""
        scaleout_mask, words = self.runtime.get_topology_masks()
        return scaleout_mask, sum(int(w) << (64 * i) for i, w in enumerate(words))

    def get_active_ranks(self) -> List[int]:
        """The world ranks of the committed active ranks, sorted."""
        return list(self.runtime.get_active_ranks())

    def _get_topology_extent(self) -> Tuple[int, int]:
        """Committed extents: highest active domain index + 1 and highest active position + 1."""
        scaleout_mask, scaleup_mask = self._get_topology_masks()
        return scaleout_mask.bit_length(), scaleup_mask.bit_length()

    def _get_mode(self) -> str:
        """The committed communication mode."""
        return _MODE_NAMES[self.runtime.get_mode()]

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: int = 1,
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

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be
                `[num_tokens, hidden]`, and type must be `torch.bfloat16`; for the second type (FP8 mode),
                the first element of the tuple must be `[num_tokens, hidden]` with type `torch.float8_e4m3fn`,
                the second is the scale factors. NVFP4 uses `(codes, scales)`: contiguous uint8 codes of shape
                `[num_tokens, hidden // 2]` (two e2m1 values per byte, even element in the low nibble) and contiguous
                float8_e4m3fn or uint8 block scales of shape `[num_tokens, hidden // 16]`, with `hidden % 64 == 0`
                and scale storage offset divisible by four; clone unaligned scale slices before dispatch.
                Dispatch transports bytes without quantizing or applying scales. The caller supplies any global scale
                to the consuming GEMM; a shared GEMM scale requires inputs quantized with that same global scale.
            topk_idx: `[num_tokens, num_topk]` with `nixl.runtime.ep.topk_idx_t` (typically `torch.int64`), the expert
                physical slot indices selected by each token, `-1` means no selection. With S = `num_experts_per_rank`,
                expert_id = world_rank * S + local_expert_idx, where `0 <= local_expert_idx < S`.
                Valid IDs are in `[0, max_num_scaleout_ranks * max_num_scaleup_ranks * S)` and stay fixed across
                membership and extent changes, including graph replay. Slot e belongs to world rank e // S,
                independent of physical topology. Selections of inactive ranks are dropped.
                The caller owns logical expert placement and routing weights; extent changes alone require no renumbering.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            cumulative_local_expert_recv_stats: `[num_experts_per_rank]` with `torch.int`, cumulative physical-slot
                receive counts. The local slot count and required allocation stay fixed across topology changes,
                including graph replay; unused slots receive no tokens.
            num_max_tokens_per_rank: the maximum number of tokens per rank (the constructor's value if not given).
            expert_alignment: align the number of tokens received by each local expert to this variable.
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
            use_tma_aligned_col_major_sf: whether to use TMA-aligned column-major layout for scale factors (FP8 only).

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`. NVFP4 returns uint8 codes and
                row-major float8_e4m3fn scales; GEMM-specific scale swizzling is left to the caller.
            recv_topk_idx: received expert indices
            recv_topk_weights: received expert weights (`None` if `topk_weights` was not provided).
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        if expert_alignment <= 0:
            raise ValueError("expert_alignment must be positive")
        check_torch_deterministic()
        x, sf = x if isinstance(x, tuple) else (x, None)

        # NVFP4 scale bytes share storage with the int32 packs copied by the kernels.
        nvfp4 = sf is not None and x.dtype == torch.uint8
        if nvfp4:
            assert (
                x.ndim == 2
                and sf.dtype in (torch.float8_e4m3fn, torch.uint8)
                and x.shape[1] % 32 == 0
                and sf.shape == (x.shape[0], x.shape[1] // 8)
                and sf.is_contiguous()
            ), "NVFP4 requires codes [tokens, hidden / 2] uint8 and contiguous scales [tokens, hidden / 16] float8_e4m3fn or uint8, hidden divisible by 64"
            sf = sf.view(torch.int32)

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
            expert_alignment,
            num_max_tokens_per_rank,
            cloned_topk_idx if do_handle_copy else topk_idx,
            num_recv_tokens,
            num_recv_tokens_per_expert_list,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            num_unaligned_recv_tokens_per_expert,
            recv_src_metadata,
            token_metadata_at_forward,
            channel_linked_list,
        )

        event_overlap = EventOverlap(event)

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

        if nvfp4:
            recv_sf = recv_sf.view(torch.float8_e4m3fn)
        recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x

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
        bias: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        previous_event: Optional[EventHandle] = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, EventOverlap]:
        """
        Combine (reduce) tokens from different ranks back to their original ranks.
        Supports both single-node and multi-node settings.
        Upstream's optional top-k weight return, used for routing-weight gradients in dispatch backward, is unsupported.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16` and `hidden` divisible by 256, the tokens to send for
                reducing to their original ranks. This combine alignment also applies after FP8/NVFP4 dispatch.
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
            handle._recv_count_is_exact,
            handle.num_max_tokens_per_rank,
            previous_event,
            previous_event_before_epilogue,
            async_with_compute_stream,
            allocate_on_comm_stream,
            handle.do_expand,
        )
        return combined_x, EventOverlap(event)
