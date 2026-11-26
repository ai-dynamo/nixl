# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Host-only API shell. Method docstrings retain the production contract for review.

Only lightweight topology/lifecycle state is modeled. Tensors created here are meta
placeholders, never communicated or computed; see the API-shell README.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from ..utils.event import EventHandle, EventOverlap


def _empty(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


class _StreamPlaceholder:
    """Opaque token: deliberately not a CUDA stream and cannot schedule work."""


class EPHandle:
    """
    Communication handle returned by `ElasticBuffer.dispatch`.
    It is consumed by `ElasticBuffer.combine` to reverse the token routing.

    Attributes:
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        topk_idx: dispatch top-k stable expert IDs, `[num_tokens, num_topk]`; cloned when `do_handle_copy=True`.
        num_recv_tokens: the actual receive count with CPU synchronization; otherwise the declared maximum world's capacity.
        num_recv_tokens_per_expert_list: CPU per-expert receive counts; empty without CPU synchronization.
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
        num_unaligned_recv_tokens_per_expert: the actual (unaligned) number of tokens received per local
            expert, shape `[num_experts_per_rank]` with `torch.int`, in both compact and expanded layouts.
        recv_src_metadata: source token indices and buffer slot indices.
        token_metadata_at_forward: per-channel forwarded token metadata (allocated in both modes, written in hybrid mode).
        channel_linked_list: per-channel per-scaleup-peer linked list (allocated in both modes, written in hybrid mode).
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
        # May not be accurate without CPU sync
        self.num_recv_tokens = num_recv_tokens
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.psum_num_recv_tokens_per_scaleup_rank = (
            psum_num_recv_tokens_per_scaleup_rank
        )
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.recv_src_metadata = recv_src_metadata
        self.token_metadata_at_forward = token_metadata_at_forward
        self.channel_linked_list = channel_linked_list

    @property
    def num_recv_tokens_device(self) -> torch.Tensor:
        """The real receive count: a 0-d view of the fixed last prefix-sum slot."""
        return _empty((), torch.int32)

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
        pass


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
        num_bytes: the allocated communication buffer size in bytes, excluding workspace.
        num_max_tokens_per_rank: the default maximum tokens per rank.
        max_num_scaleout_ranks: the declared maximum number of scaleout ranks (NVLink domains) the buffer is sized for.
        max_num_scaleup_ranks: the declared maximum number of scaleup ranks (positions per domain).
        world_rank: this rank's caller-assigned ID, independent of physical topology.
        num_launch_sms: the read-only main dispatch/combine block count, fixed for this buffer's lifetime.
            An even integer from 2 through 64 (default 24).
        runtime: the C++ runtime.

    Elasticity: production discovers physical domains during connection preparation and maps world ranks to internal
    domain/position slots. The active hardware must form a supported product of domains and positions.
    `connect_ranks` prepares peers; `set_active_ranks` activates the topology the kernels read.
    Only active ranks launch dispatch/combine. Construction initializes this rank alone.
    Connect peers explicitly, then activate the intended world. `disconnect_ranks` excludes and retires peers;
    on an active local rank, the survivors must still form a nonempty product of domains and positions.
    An inactive local rank preserves its active-world descriptor while retiring any connected peers.
    A single active rank can run dispatch/combine before connecting peers, including warm-up and graph capture.
    New-peer connections may run while the committed graph replays. Serialize
    control operations on each buffer; a background setup thread must select the buffer's CUDA device.
    Activation (`set_active_ranks`) and peer retirement run outside capture, between complete dispatch/combine pairs.
    Membership and direct/hybrid mode may change while the same graph uses the fixed `num_launch_sms` grid.
    Order topology commits and graph replays on the same CUDA stream, or establish the dependencies explicitly.
    Before activation or peer retirement, every rank of the old world, including departing ranks, must finish
    its queued CUDA work, call `barrier(with_cpu_sync=True)` to drain the transport, and rendezvous on the host before
    any peer departs. The update methods do not establish that completion boundary.
    Invalid inputs and mapping refusals before peer retirement preserve active views; failures after retirement or view activation are fatal.

    World ranks and public expert IDs remain independent of internal topology coordinates.
    Internal source metadata uses the committed topology's rank encoding.
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
            world_rank: this rank's caller-assigned fixed ID in
                `[0, max_num_scaleout_ranks * max_num_scaleup_ranks)`, independent of physical topology.
                The same IDs are passed to `connect_ranks` and `set_active_ranks`.
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
            num_launch_sms: fixed main dispatch/combine block count, an even integer from 2 through 64 (default 24).
                Must match across peers and remains unchanged for the buffer lifetime, including graph replay.
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
        if num_max_tokens_per_rank <= 0 or hidden <= 0:
            raise ValueError("num_max_tokens_per_rank and hidden must be positive")
        self.num_experts_per_rank = num_experts_per_rank
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        (self.deterministic, self.explicitly_destroy) = (
            deterministic,
            explicitly_destroy,
        )
        self.store = store
        self.num_bytes = self.get_buffer_size_hint(
            max_num_scaleout_ranks,
            max_num_scaleup_ranks,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            use_fp8_dispatch,
            allow_multiple_reduction,
        )
        self.world_rank = world_rank
        self.max_num_scaleout_ranks = max_num_scaleout_ranks
        self.max_num_scaleup_ranks = max_num_scaleup_ranks
        self._num_launch_sms = num_launch_sms
        self.runtime = None
        self._destroyed = False
        self._stream = _StreamPlaceholder()
        self._active: Tuple[int, ...] = (world_rank,)
        self._connected: set[int] = set()

    @property
    def num_launch_sms(self) -> int:
        """The fixed main dispatch/combine block count selected at construction."""
        return self._num_launch_sms

    def destroy(self) -> None:
        """Destroy the runtime; requires `explicitly_destroy=True` at construction."""
        if not self.explicitly_destroy:
            raise ValueError("destroy() requires explicitly_destroy=True")
        self._destroyed = True
        self._connected.clear()

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
        return 0

    def read_profile(self) -> Dict[str, torch.Tensor]:
        """Read and clear diagnostic timestamps; requires NIXL_EP_ADAPTIVE_PROFILE=1 before library initialization.

        Returns a mapping from native phase names to CPU int64 tensors indexed by [block, warp].
        Values are device global-timer nanoseconds; zero means that writer did not reach that phase.
        This call synchronizes the device and completes the reset before returning; it cannot run during capture.
        Discard warm-up samples with one read, then run one isolated dispatch/combine pair or graph replay per read.
        Instrumented phase boundaries are diagnostic samples, not uninstrumented kernel completion times.
        """
        self._ensure_open()
        return {}

    def barrier(
        self, use_comm_stream: bool = True, with_cpu_sync: bool = False
    ) -> None:
        """
        Drain outstanding transport operations and barrier across the active ranks, optionally synchronizing the CPU.

        Arguments:
            use_comm_stream: whether to use the communication stream (otherwise uses the current compute stream).
            with_cpu_sync: whether to also call `cudaDeviceSynchronize` before and after the barrier.
        """
        self._ensure_open()

    @staticmethod
    def capture() -> EventHandle:
        """Capture a CUDA event on the current stream."""
        return EventHandle()

    def get_comm_stream(self) -> torch.Stream:
        """Return the buffer's communication stream."""
        self._ensure_open()
        return self._stream

    def connect_ranks(
        self,
        world_ranks: List[int],
        store: Optional[dist.Store] = None,
        key_prefix: str = "NIXL_EP",
        timeout_secs: float = 120.0,
    ) -> None:
        """
        Exchange agent names, region descriptors and physical topology evidence through a Store, fetch transport
        metadata through NIXL's metadata manager, and prepare internal rank mappings and memory views for activation
        by `set_active_ranks`. Already-connected peers are ignored.
        Discovery data uses `{key_prefix}/{world_rank}` and is deleted after all new peers have read it and
        native preparation finishes. Each new peer must request this rank;
        complete the exchange on all participants before reusing its rank keys.
        Native preparation releases the GIL so another Python thread can replay the committed graph;
        it does not change the live topology or active views.

        Arguments:
            world_ranks: world ranks of the peers (not this rank); already-connected peers are ignored.
            store: the Store for new-peer discovery (default: the constructor's explicit `store`).
            key_prefix: the same Store key prefix on both sides; separate buffers using the same rank IDs need separate prefixes or Stores.
            timeout_secs: the bound for each Store wait for discovery data or read acknowledgements;
                NIXL metadata fetch and wire-up share `NIXL_EP_NIXL_CONNECT_TIMEOUT_SECS`.
        """
        self._ensure_open()
        world_ranks = [rank for rank in world_ranks if rank not in self._connected]
        if not world_ranks:
            return
        store = self.store if store is None else store
        if store is None:
            raise ValueError(
                "Peer discovery requires an explicit Store; pass store to ElasticBuffer or connect_ranks(store=...)"
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
        self._connected.update(peers)

    def disconnect_ranks(self, world_ranks: List[int]) -> None:
        """
        Exclude departing peers and retire their connections outside capture, after draining the old world and a host rendezvous.
        An active caller's remaining ranks must form a product of domains and positions; the mode follows that world.
        An inactive caller only retires connections, preserving its membership descriptor. An empty list does nothing.
        A live rank sitting a world out is only excluded with `set_active_ranks`.
        """
        self._ensure_open()
        if not world_ranks:
            return
        peers = set(world_ranks)
        if len(peers) != len(world_ranks) or not peers.issubset(self._connected):
            raise ValueError("Disconnect only connected peers, each once")
        active = self._active
        if self.world_rank in active:
            active = tuple(rank for rank in active if rank not in peers)
        self._active, self._connected = active, self._connected - peers

    def get_connected_ranks(self) -> List[int]:
        """World ranks of the connected peers (this rank excluded)."""
        self._ensure_open()
        return sorted(self._connected)

    def set_active_ranks(self, active_ranks: Sequence[int]) -> None:
        """Validate and activate connected ranks forming a product of domains and positions; the mode follows the new world.

        Call outside capture after draining the old world and completing a host rendezvous.
        The buffer keeps the fixed main dispatch/combine block count selected at construction.
        """
        self._ensure_open()
        active = tuple(sorted(set(active_ranks)))
        if not active or any(
            type(rank) is not int
            or not 0 <= rank < self.max_num_scaleout_ranks * self.max_num_scaleup_ranks
            for rank in active
        ):
            raise ValueError(
                "Active ranks must be nonempty integers within the declared maximum world"
            )
        if self.world_rank in active and (
            not set(active).issubset(self._connected | {self.world_rank})
        ):
            raise ValueError(
                "Connect the requested peers before activating the requested topology"
            )
        self._active = active

    def get_active_ranks(self) -> List[int]:
        """The world ranks of the committed active ranks, sorted."""
        self._ensure_open()
        return list(self._active)

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
        Uses the fixed `num_launch_sms` main-kernel block count selected at construction.
        Membership and direct/hybrid mode can change between completed pairs, including graph replays.

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
                membership and extent changes, including graph replay. Slot e belongs to world rank e // S;
                production resolves its internal coordinates from the discovered topology. Selections of inactive ranks are dropped.
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
        self._ensure_dispatchable()
        if expert_alignment <= 0:
            raise ValueError("expert_alignment must be positive")
        num_topk = topk_idx.shape[1]
        (data, scales) = x if isinstance(x, tuple) else (x, None)
        received = _empty((0, *data.shape[1:]), data.dtype)
        if scales is not None:
            scale_dtype = (
                torch.float8_e4m3fn if data.dtype == torch.uint8 else scales.dtype
            )
            received = (received, _empty((0, *scales.shape[1:]), scale_dtype))
        received_indices = None if do_expand else _empty((0, num_topk), topk_idx.dtype)
        received_weights = (
            None
            if topk_weights is None
            else _empty((0,) if do_expand else (0, num_topk), topk_weights.dtype)
        )
        capacity = (
            self.num_max_tokens_per_rank
            if num_max_tokens_per_rank is None
            else num_max_tokens_per_rank
        )
        handle = EPHandle(
            do_expand,
            expert_alignment,
            capacity,
            _empty(topk_idx.shape, topk_idx.dtype) if do_handle_copy else topk_idx,
            0,
            [0] * self.num_experts_per_rank if do_cpu_sync else [],
            _empty((self.max_num_scaleup_ranks + 1,), torch.int32),
            _empty((self.num_experts_per_rank,), torch.int32),
            _empty((self.num_experts_per_rank,), torch.int32),
            _empty((0, 2 + num_topk), torch.int32),
            _empty((0,), torch.int32),
            _empty((0,), torch.int32),
        )
        event = EventOverlap(EventHandle() if async_with_compute_stream else None)
        return (
            received,
            received_indices,
            received_weights,
            handle,
            event,
        )

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
        Dispatch and combine use the fixed `num_launch_sms` main-kernel block count throughout the buffer lifetime.
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
        self._ensure_dispatchable()
        return (
            _empty((handle.topk_idx.shape[0], x.shape[1]), torch.bfloat16),
            EventOverlap(EventHandle() if async_with_compute_stream else None),
        )

    def _ensure_open(self):
        if self._destroyed:
            raise RuntimeError("This API-shell buffer has been destroyed")

    def _ensure_dispatchable(self):
        self._ensure_open()
        if self.world_rank not in self._active:
            raise ValueError("Dispatch/combine require this rank to be active")
