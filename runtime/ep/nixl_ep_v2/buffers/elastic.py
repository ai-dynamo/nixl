# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Host-only API shell. Method docstrings retain the production contract for review.

Only lightweight topology/lifecycle state is modeled. Tensor results are meta
placeholders, never communicated or computed; see the API-shell README.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from ..utils.event import EventHandle, EventOverlap

_DEFAULT_SM_READ_GBS, _DEFAULT_SM_WRITE_GBS = 200, 50


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
        num_sms: the active SM count recorded when the handle was created, checked by eager combine. A captured
            dispatch/combine graph reads the current count from the topology descriptor.
        topk_idx: dispatch top-k expert indices, `[num_tokens, num_topk]`; cloned when `do_handle_copy=True`.
        psum_num_recv_tokens_per_scaleup_rank: inclusive prefix sum of deduplicated received token counts
            per scaleup rank, shape `[max_num_scaleup_ranks + 1]`. A token is counted once per rank even if
            multiple of its top-k experts land on the same rank. The last element equals the total number
            of received tokens.
        psum_num_recv_tokens_per_expert: prefix sum of alignment-padded received token counts per local
            expert, shape `[num_experts_per_rank]`. Each expert's count is padded to `expert_alignment`.
            In non-expand mode, this is the inclusive prefix sum. In expand mode, `psum[i]` equals
            the aligned cumulative count of experts before `i` plus the actual (unaligned) token count
            of expert `i` — so `psum[i] - align(psum[i-1], expert_alignment)` recovers the real
            count for expert `i`, and `align(psum[i], expert_alignment)` gives expert `i+1`'s
            starting offset.
        num_recv_tokens_per_expert_list: Python list of per-expert received token counts (CPU-side).
        num_unaligned_recv_tokens_per_expert: the actual (unaligned) number of tokens received per local
            expert, shape `[num_experts_per_rank]` with `torch.int`, in both compact and expanded layouts.
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
    New-peer connections and host-only topology staging may run while the committed graph replays. Serialize
    control operations on each buffer; a background setup thread must select the buffer's CUDA device.
    Activation (`commit_topology`, `set_num_active_sms`) runs outside capture, between complete dispatch/combine pairs.
    Order topology commits and graph replays on the same CUDA stream, or establish the dependencies explicitly.
    Before activation or peer retirement, every rank of the old world, including departing ranks, must finish
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
        *,
        num_experts_per_rank: int = 0,
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
            use_fp8_dispatch: reserve smaller dispatch arenas for FP8/NVFP4 inputs; False reserves BF16 capacity and also
                accepts either quantized format. Must match across peers, including with explicit `num_bytes`; it does not cast inputs.
            num_experts_per_rank: fixed physical routing slots per GPU, shared by all peers, including joiners.
                Required (positive) for dispatch/combine; zero permits barrier-only buffers. This is not the model's
                logical expert count. Slots may remain unused; the caller assigns weights and maps logical experts
                to physical slots. The transport neither replicates experts nor creates weights.
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
            (rank_idx, num_ranks) = (group.rank(), group.size())
        if (
            type(rank_idx) is not int
            or type(num_ranks) is not int
            or (not 0 <= rank_idx < num_ranks)
        ):
            raise ValueError(
                "Without a process group, rank_idx and a positive num_ranks are required"
            )
        if type(num_experts_per_rank) is not int or num_experts_per_rank < 0:
            raise ValueError("num_experts_per_rank must be a nonnegative integer")
        positions = num_group_scaleup_ranks or num_ranks
        if positions <= 0 or num_ranks % positions:
            raise ValueError("num_group_scaleup_ranks must divide num_ranks")
        (self.rank_idx, self.num_ranks) = (rank_idx, num_ranks)
        self.num_experts_per_rank = num_experts_per_rank
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        (self.deterministic, self.explicitly_destroy) = (
            deterministic,
            explicitly_destroy,
        )
        (self.store, self.num_bytes) = (
            store,
            num_bytes if num_bytes is not None else 0,
        )
        self.max_num_scaleout_ranks = max_num_scaleout_ranks or num_ranks // positions
        self.max_num_scaleup_ranks = max_num_scaleup_ranks or positions
        self.scaleout_rank_idx = (
            rank_idx // positions if scaleout_rank_idx is None else scaleout_rank_idx
        )
        self.scaleup_rank_idx = (
            rank_idx % positions if scaleup_rank_idx is None else scaleup_rank_idx
        )
        domain_base = self.scaleout_rank_idx - rank_idx // positions
        position_base = self.scaleup_rank_idx - rank_idx % positions
        if not (
            0 <= domain_base
            and domain_base + num_ranks // positions <= self.max_num_scaleout_ranks
            and 0 <= position_base
            and position_base + positions <= self.max_num_scaleup_ranks
        ):
            raise ValueError(
                "Cohort coordinates must be within the declared maximum world"
            )
        self.world_rank = self.world_rank_of(
            self.scaleout_rank_idx, self.scaleup_rank_idx
        )
        self.num_launch_sms = 24
        self.runtime = None
        self._destroyed = False
        self._auto_sm_config: Optional[Tuple[int]] = None
        (self._sms, self._pending, self._peers_changed) = (24, None, False)
        self._stream = _StreamPlaceholder()
        self._active = tuple(
            (
                self.world_rank_of(
                    domain_base + i // positions, position_base + i % positions
                )
                for i in range(num_ranks)
            )
        )
        self.masks_of(self._active)
        self._connected = set(self._active) - {self.world_rank}
        self._pending_disconnects = set()

    def destroy(self) -> None:
        """Destroy the runtime; requires `explicitly_destroy=True` at construction."""
        if not self.explicitly_destroy:
            raise ValueError("destroy() requires explicitly_destroy=True")
        self._destroyed = True
        self._connected.clear()
        self._pending_disconnects.clear()
        self._pending = None

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
            use_fp8_dispatch: size dispatch arenas for FP8/NVFP4 inputs; pass the same value to the constructor.
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

    def get_theoretical_num_sms(
        self,
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
        Traffic uses active domain/position counts and their fixed physical slots. One active domain
        has no RDMA traffic even above domain index 0. Fewer active experts than `num_topk` requires explicit `num_sms`.
        Every active rank's `dispatch(num_sms=0)` commits the same active SM count.

        Arguments:
            num_topk: the number of top-k experts per token.
            rdma_gbs: the RDMA bandwidth in GB/s (0 for auto-detect).
            nvlink_gbs: the NVLink bandwidth in GB/s (0 for auto-detect).
            sm_read_gbs: the per-SM HBM read bandwidth in GB/s.
            sm_write_gbs: the per-SM HBM write bandwidth in GB/s.

        Returns:
            num_sms: the raw model estimate (even, at least 2), before the launch-grid limit.
        """
        self._ensure_open()
        return 2

    def get_local_metadata(self) -> bytes:
        """This rank's NIXL agent metadata, which a peer loads to address this rank (`connect_ranks`)."""
        self._ensure_open()
        return f"api-shell:{self.world_rank}".encode()

    def get_local_peer_info(self) -> Tuple[int, int, int, int]:
        """This region's `(base address, CUDA device index, world rank, allocation bytes)`."""
        self._ensure_open()
        return (0, 0, self.world_rank, 0)

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
        self._ensure_open()

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
        Resets only new peers' barrier words on a setup stream. Native preparation releases the GIL so another
        Python thread can replay the committed graph; it does not change the live topology or active views.

        Arguments:
            world_ranks: world ranks of the peers (not this rank, not connected yet).
            remote_mds: the peers' `get_local_metadata()`, aligned with `world_ranks`.
            peer_infos: the peers' `get_local_peer_info()`, aligned with `world_ranks`.
            store: read `{key_prefix}/{world_rank}` from this store instead (default: the constructor's explicit `store`).
            key_prefix: the publisher's key prefix; replacement freshness is the caller's responsibility (see `publish`).
            timeout_secs: how long to wait for each peer's key in `store` (the wire-up has its own `NIXL_EP_NIXL_CONNECT_TIMEOUT_SECS`).
        """
        self._ensure_open()
        if (remote_mds is None) != (peer_infos is None):
            raise ValueError(
                "remote_mds and peer_infos are given together (the peers' metadata and peer info), or neither (Store discovery)"
            )
        if remote_mds is not None and not (
            len(remote_mds) == len(peer_infos) == len(world_ranks)
        ):
            raise ValueError("Metadata lists must match the length of world_ranks")
        peers = set(world_ranks)
        if len(peers) != len(world_ranks) or peers.intersection(self._connected):
            raise ValueError(
                "Connect each peer once; peers must not already be connected"
            )
        if peers.intersection(self._pending_disconnects):
            raise ValueError(
                "Commit the pending disconnect before reconnecting the same rank"
            )
        for rank in peers:
            if (
                not 0 <= rank < self.max_num_scaleout_ranks * self.max_num_scaleup_ranks
                or rank == self.world_rank
            ):
                raise ValueError(
                    "Connected ranks must be other ranks within the declared maximum world"
                )
        self._connected.update(peers)
        self._peers_changed = bool(peers) or self._peers_changed

    def disconnect_ranks(self, world_ranks: List[int]) -> None:
        """
        Forget peers whose process goes away (staged; dropped at the next `commit_topology`). Commit a topology without
        the peer first (a staged, uncommitted topology must not hold it either). A rank that stays alive but sits a world
        out is left out of `active_ranks` instead.
        """
        self._ensure_open()
        peers = set(world_ranks)
        if len(peers) != len(world_ranks) or not peers.issubset(self._connected):
            raise ValueError("Disconnect only connected peers, each once")
        protected = set(self._active) if self.world_rank in self._active else set()
        if self._pending is not None and self.world_rank in self._pending[0]:
            protected.update(self._pending[0])
        if protected.intersection(peers):
            raise ValueError(
                "Commit a topology excluding peers before disconnecting them; exclude them from staged topology too"
            )
        self._connected.difference_update(peers)
        self._pending_disconnects.update(peers)
        self._peers_changed = bool(peers) or self._peers_changed

    def get_connected_ranks(self) -> List[int]:
        """World ranks of the connected peers (this rank excluded)."""
        self._ensure_open()
        return sorted(self._connected)

    def world_rank_of(self, scaleout_rank_idx: int, scaleup_rank_idx: int) -> int:
        """The world rank of coordinates `(scaleout_rank_idx, scaleup_rank_idx)` in the declared maximum world."""
        return scaleout_rank_idx * self.max_num_scaleup_ranks + scaleup_rank_idx

    def masks_of(self, active_ranks: Sequence[int]) -> Tuple[int, int]:
        """The `(scaleout_mask, scaleup_mask)` of world ranks forming a product of domains and positions."""
        active = sorted(set(active_ranks))
        if not active or any(
            (
                r < 0 or r >= self.max_num_scaleout_ranks * self.max_num_scaleup_ranks
                for r in active
            )
        ):
            raise ValueError(
                "Active ranks must be nonempty and within the declared maximum world"
            )
        domains = sorted({r // self.max_num_scaleup_ranks for r in active})
        positions = sorted({r % self.max_num_scaleup_ranks for r in active})
        if active != [self.world_rank_of(d, p) for d in domains for p in positions]:
            raise ValueError(
                "Active ranks must form a product of domains and positions"
            )
        return (sum((1 << d for d in domains)), sum((1 << p for p in positions)))

    @staticmethod
    def mode_of(scaleout_mask: int) -> str:
        """Exactly one active NVLink domain selects direct mode, regardless of its index."""
        return "direct" if bin(scaleout_mask).count("1") == 1 else "hybrid"

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
        self._ensure_open()
        active = (
            tuple(sorted(set(active_ranks)))
            if active_ranks is not None
            else self._active
        )
        self.masks_of(active)
        if self.world_rank in active and (
            not set(active).issubset(self._connected | {self.world_rank})
        ):
            raise ValueError(
                "Connect the requested peers before staging an active topology"
            )
        auto_sm_config = self._auto_sm_config if num_active_sms == 0 else None
        sms = num_active_sms or (2 if auto_sm_config is not None else self._sms)
        self._validate_sms(sms)
        self._pending = (active, sms, auto_sm_config)

    def commit_topology(self) -> None:
        """
        Make the staged topology descriptor (`EPTopology`) live: one stream-ordered device write on the current stream
        (a device synchronisation first when peers were connected or disconnected).
        """
        self._ensure_open()
        if self._pending is None and (not self._peers_changed):
            raise ValueError(
                "Nothing staged: stage topology or connect/disconnect peers first"
            )
        if self._pending is not None:
            (self._active, self._sms, self._auto_sm_config) = self._pending
        (self._pending, self._peers_changed) = (None, False)
        self._pending_disconnects.clear()

    def set_active_ranks(
        self, active_ranks: Sequence[int], num_active_sms: int = 0
    ) -> None:
        """`stage_topology(active_ranks=...)` + `commit_topology()` in one step; the mode follows the new world."""
        self.stage_topology(active_ranks=active_ranks, num_active_sms=num_active_sms)
        self.commit_topology()

    def get_topology_masks(self) -> Tuple[int, int]:
        """Committed `(scaleout_mask, scaleup_mask)`: bit d / bit k denotes an active domain / position."""
        self._ensure_open()
        return self.masks_of(self._active)

    def get_active_ranks(self) -> List[int]:
        """The world ranks of the committed active ranks, sorted."""
        self._ensure_open()
        return list(self._active)

    def get_topology_extent(self) -> Tuple[int, int]:
        """Committed extents: highest active domain index + 1 and highest active position + 1."""
        (domains, positions) = self.get_topology_masks()
        return (domains.bit_length(), positions.bit_length())

    @property
    def is_active(self) -> bool:
        """Whether this rank belongs to the committed world, including singleton bootstrap worlds."""
        self._ensure_open()
        return self.world_rank in self._active

    def set_num_active_sms(self, num_active_sms: int) -> None:
        """
        Recommit the committed topology descriptor with a new active SM count (even, at most `num_launch_sms`); its world
        and mode are untouched. Pins manual mode until a successful `dispatch(num_sms=0)`.
        Refused while a staged world awaits `commit_topology`: commit it first, or stage after this call.
        """
        self._ensure_open()
        if self._pending is not None:
            raise ValueError(
                "Commit the staged topology before changing the active SM count"
            )
        self._validate_sms(num_active_sms)
        self._pending = (self._active, num_active_sms, None)
        self.commit_topology()

    def get_topology(self) -> Tuple[str, int]:
        """The committed `(mode, num_active_sms)`."""
        (domains, _) = self.get_topology_masks()
        return (self.mode_of(domains), self._sms)

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
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
                the second is the scale factors. NVFP4 uses `(codes, scales)`: contiguous uint8 codes of shape
                `[num_tokens, hidden // 2]` (two e2m1 values per byte, even element in the low nibble) and contiguous
                float8_e4m3fn or uint8 block scales of shape `[num_tokens, hidden // 16]`, with `hidden % 64 == 0`
                and scale storage offset divisible by four; clone unaligned scale slices before dispatch.
                Dispatch transports bytes without quantizing or applying scales. The caller supplies any global scale
                to the consuming GEMM; a shared GEMM scale requires inputs quantized with that same global scale.
            topk_idx: `[num_tokens, num_topk]` with `nixl.runtime.ep.topk_idx_t` (typically `torch.int64`), the expert
                physical slot indices selected by each token, `-1` means no selection. With S = `num_experts_per_rank`,
                slot e belongs to extent rank e // S, whose physical coordinates are
                `(rank // scaleup_extent, rank % scaleup_extent)`. The routing range is
                `scaleout_extent * scaleup_extent * S`, including holes; inactive slots are not renumbered away.
                The caller updates routing IDs when the scaleup extent changes, and supplies correct weights.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            cumulative_local_expert_recv_stats: `[num_experts_per_rank]` with `torch.int`, cumulative physical-slot
                receive counts. The local slot count and required allocation stay fixed across topology changes,
                including graph replay; unused slots receive no tokens.
            num_max_tokens_per_rank: the maximum number of tokens per rank (the constructor's value if not given).
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_sms: the number of SMs to use (0 for automatic via `get_theoretical_num_sms`). A successful automatic
                dispatch remembers top-k for later topology updates; a nonzero value pins manual mode.
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
        auto_sm_config = (num_topk,) if num_sms == 0 else None
        sms = (
            2
            if auto_sm_config is not None
            else max(2, min((num_sms + 1) // 2 * 2, self.num_launch_sms))
        )
        if self._pending is not None and (
            sms != self._sms or auto_sm_config != self._auto_sm_config
        ):
            raise ValueError("Commit the staged topology before changing the SM policy")
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
            sms,
            _empty(topk_idx.shape, topk_idx.dtype) if do_handle_copy else topk_idx,
            0,
            [0] * self.num_experts_per_rank if do_cpu_sync else [],
            _empty((self.max_num_scaleup_ranks + 1,), torch.int32),
            _empty((self.num_experts_per_rank,), torch.int32),
            _empty((self.num_experts_per_rank,), torch.int32),
            _empty((0, 2 + num_topk), torch.int32),
            _empty((0,), torch.int32),
            _empty((0,), torch.int32),
            do_cpu_sync,
        )
        event = EventOverlap(EventHandle() if async_with_compute_stream else None)
        if sms != self._sms:
            self.set_num_active_sms(sms)
        self._auto_sm_config = auto_sm_config
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
        if handle.num_sms != self._sms:
            raise ValueError("The handle SM count differs from the committed topology")
        return (
            _empty((handle.topk_idx.shape[0], x.shape[1]), torch.bfloat16),
            EventOverlap(EventHandle() if async_with_compute_stream else None),
        )

    def _ensure_open(self):
        if self._destroyed:
            raise RuntimeError("This API-shell buffer has been destroyed")

    def _ensure_dispatchable(self):
        self._ensure_open()
        if self.num_experts_per_rank <= 0:
            raise ValueError("Set num_experts_per_rank when constructing an EP buffer")
        if len(self._active) < 2 or not self.is_active:
            raise ValueError(
                "Dispatch/combine require this rank in a world with at least two active ranks"
            )

    def _validate_sms(self, count):
        if not 2 <= count <= self.num_launch_sms or count % 2:
            raise ValueError(
                "Active SM count must be even and between 2 and num_launch_sms"
            )
