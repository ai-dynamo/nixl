# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Elastic membership of the adaptive EP kernels on the NIXL transport (RDMA + NVLink; the host derives the kernel mode of
each phase, direct iff one NVLink domain is active): a plan of phases
(phases: lists of public world ranks whose discovered slots form a product of
NVLink domains and positions) on a job whose processes all stay alive. Participating ranks activate each phase.
By default, departing ranks also adopt that phase to become inactive while keeping their connections.
With `--reconnect-on-leave`, departing ranks retire their peers into a local singleton state and sit out;
rejoining ranks reconnect before activation and may receive new internal positions. Custom plans must remain a product
of domains and positions after reconnection; unsupported plans fail activation. The default plan restores the full cohort
between shrinks. Participating ranks run one eager dispatch (CPU sync) + combine routed over the ACTIVE experts, checked without
any collective (every rank regenerates every other rank's routing from a seed), then replay the CUDA graph captured at
the first phase they took part in, both with its original routing (departed owners are dropped) and with fresh routing
over the current active experts. Payloads, received expert indices/weights and expert statistics are checked.
Single-rank traffic phases are supported. `--dtype` selects BF16, FP8 or synthetic NVFP4 dispatch bytes for
checked and timed pairs; combine stays BF16. Each run uses matching buffer capacity.
Ranks outside the phase mirror host barriers but do not dispatch or combine. Timing is opt-in: `--benchmark` measures CUDA-event
call/pair times, `--kineto` measures kernels, `--perf-phases` selects phases for both, and `--feed-pairs` measures transitions.
Feed warm-up retains connections while visiting phases with new participants; those phases must also be valid under the initial mapping.
`--skip-perf-test` suppresses event benchmarks; `--save-output [PATH]` saves per-rank results and configuration as JSON.

Launch: one invocation per node (`--num-processes` GPUs each) under node-level `WORLD_SIZE` / `RANK`,
no process group; the rank 0 process serves a TCPStore on `MASTER_ADDR` / `MASTER_PORT` (127.0.0.1:8361) for the whole
run and keeps it alive until every peer has torn down (if a previous run still holds the port, choose another).
`--store-timeout-secs` bounds every Store wait, `--num-cpu-timeout-secs` only the dispatch's CPU sync. `--num-experts`
is the total physical slot count of the declared maximum world and must be divisible by its rank count.
For a single-domain eight-GPU job, setting `--num-processes` to 8, `--max-scaleup-ranks` to 72,
`--position-offset` to 64 and `--num-experts` to 576 assigns public IDs 64..71 with eight expert slots per rank;
internal positions are discovered, not prescribed. `--plan` accepts an inline JSON list of phases or a plan file.
"""
import argparse
import json
import os
import tempfile
import time
from datetime import timedelta
from statistics import median_high as median
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import torch
import torch.distributed as dist
from buffer_setup import rank_coordinates, rank_mapping, topology_experts

import nixl.runtime.ep as nixl_ep_v2
from nixl.runtime.ep.utils.math import calc_diff, per_token_cast_to_fp8
from nixl.runtime.ep.utils.nixl_bootstrap import NixlBootstrap
from nixl.runtime.ep.utils.profiling import summarize_profile
from nixl.runtime.ep.utils.refs import combine as ref_combine
from nixl.runtime.ep.utils.refs import generate_pre_combine_data, ordered_accumulate
from nixl.runtime.ep.utils.testing import bench, bench_kineto


def init_test_environment(
    local_rank: int, num_local_ranks: int, seed: int
) -> Tuple[int, int]:
    """Use node-level `WORLD_SIZE`/`RANK` without a process group: this process's index in the
    cohort of every invocation and the cohort size; selects the GPU and seeds per rank.
    """
    num_nodes, node_rank = int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("RANK", 0))
    assert num_nodes > 0 and 0 <= node_rank < num_nodes
    assert 0 <= local_rank < num_local_ranks
    assert (
        not dist.is_initialized()
    ), "This test uses a standalone Store, not a process group"
    rank_idx, num_ranks = (
        node_rank * num_local_ranks + local_rank,
        num_nodes * num_local_ranks,
    )
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(seed + rank_idx)
    return rank_idx, num_ranks


def make_buffer(
    args: argparse.Namespace,
    rank_idx: int,
    num_ranks: int,
    store: dist.Store,
    **placement,
) -> nixl_ep_v2.ElasticBuffer:
    key_prefix = placement.pop("key_prefix", "NIXL_EP")
    positions = placement.pop("num_group_scaleup_ranks", num_ranks)
    max_so = placement.pop("max_num_scaleout_ranks", 0) or num_ranks // positions
    max_su = placement.pop("max_num_scaleup_ranks", 0) or positions
    world = placement.pop("world_ranks", None)
    if world is None:
        world = [
            (rank // positions) * max_su + rank % positions for rank in range(num_ranks)
        ]
    assert len(world) == num_ranks and len(set(world)) == num_ranks
    max_ranks = max_so * max_su
    assert args.num_experts > 0 and args.num_experts % max_ranks == 0
    buffer = nixl_ep_v2.ElasticBuffer(
        world_rank=world[rank_idx],
        max_num_scaleout_ranks=max_so,
        max_num_scaleup_ranks=max_su,
        store=store,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        num_topk=placement.pop("num_topk", args.num_topk),
        num_experts_per_rank=args.num_experts // max_ranks,
        num_launch_sms=args.num_sms,
        use_fp8_dispatch=getattr(args, "dtype", "bf16") != "bf16",
        allow_multiple_reduction=bool(args.allow_multiple_reduction),
        prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
        explicitly_destroy=True,
        num_gpu_timeout_secs=args.num_gpu_timeout_secs,
        num_cpu_timeout_secs=args.num_cpu_timeout_secs,
        **placement,
    )
    if num_ranks > 1:
        buffer.connect_ranks(
            [rank for rank in world if rank != buffer.world_rank],
            key_prefix=key_prefix,
            timeout_secs=args.store_timeout_secs,
        )
        buffer.set_active_ranks(world)
    return buffer


def extent_of(phase: Sequence[int], max_su: int, mapping=None) -> Tuple[int, int]:
    """Highest occupied internal domain and position, plus one."""
    slots = [mapping[r] if mapping is not None else r for r in phase]
    return max(s // max_su for s in slots) + 1, max(s % max_su for s in slots) + 1


def load_plan(spec: str) -> List[List[int]]:
    """Read the phase list from inline JSON or a JSON file."""
    if os.path.isfile(spec):
        with open(spec) as source:
            spec = source.read()
    return json.loads(spec)


def default_plan(max_su: int, present: List[int], mapping=None) -> List[List[int]]:
    """The full world, then each candidate (a domain alone, uniform and holed position shrinks) followed by the full
    world again; all worlds are nonempty.
    The full world is the set of PRESENT ranks."""
    mapping = mapping or {r: r for r in present}
    inverse = {slot: rank for rank, slot in mapping.items()}
    domains = sorted({mapping[r] // max_su for r in present})
    positions = sorted({mapping[r] % max_su for r in present})

    def ids(ds, ks):
        return sorted(inverse[d * max_su + k] for d in ds for k in ks)

    all_ranks = ids(domains, positions)
    assert all_ranks == sorted(
        present
    ), f"The present ranks {sorted(present)} are not a product of domains and positions"
    phases = [all_ranks]
    candidates = []
    if len(domains) > 1:
        candidates.append(ids(domains[:1], positions))  # the first domain alone
    if len(positions) > 1:
        candidates.append(
            ids(domains, positions[: len(positions) // 2])
        )  # every domain, half the positions
        candidates.append(
            ids(domains, positions[:-1])
        )  # every domain, one position fewer
        candidates.append(
            ids(domains, positions[1:])
        )  # every domain, the first position out (a hole)
        candidates.append(
            ids(domains, positions[1::2])
        )  # every domain, every other position
    if len(domains) > 1:
        candidates.append(
            ids(domains[-1:], positions)
        )  # the last domain alone (the extent keeps the holes)
    candidates.append(ids(domains[:1], positions[:1]))
    for c in candidates:
        if c and c != all_ranks and c not in phases:
            phases.append(c)
            phases.append(all_ranks)
    return phases


def sample_routing(
    generator: torch.Generator,
    num_tokens: int,
    num_experts: int,
    active_mask: torch.Tensor,
    num_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(topk_idx, topk_weights)` of `num_tokens` tokens drawn from `generator`: `num_topk` distinct experts among the
    `active_mask` ones (the inactive scores are set to -1 before the top-k), positive weights normalised per token
    """
    scores = torch.rand(num_tokens, num_experts, generator=generator, device="cuda")
    scores[:, ~active_mask] = -1.0
    weights, idx = torch.topk(
        scores, min(num_topk, num_experts), dim=-1, largest=True, sorted=False
    )
    idx = torch.where(weights >= 0, idx, -1)
    weights = weights.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    if idx.shape[1] < num_topk:
        padding = num_topk - idx.shape[1]
        idx = torch.nn.functional.pad(idx, (0, padding), value=-1)
        weights = torch.nn.functional.pad(weights, (0, padding))
    return idx.to(nixl_ep_v2.topk_idx_t), weights.float().contiguous()


# A dispatch payload: BF16 rows, or `(data, scales)` for FP8 and NVFP4
Payload = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def as_nvfp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic NVFP4 transport bytes: packed values and e4m3 block scales."""
    hidden = x.shape[1]
    raw = x.contiguous().view(torch.uint8)
    codes = raw[:, : hidden // 2].contiguous()
    scales = raw[:, hidden // 2 : hidden // 2 + hidden // 16].contiguous()
    return codes, scales.view(torch.float8_e4m3fn)


def payload_parts(x: Payload) -> Tuple[torch.Tensor, ...]:
    return x if isinstance(x, tuple) else (x,)


# Rows move as bytes: float8 tensors lack indexing and concatenation kernels on some devices
def payload_rows(x: Payload, index) -> Payload:
    """The rows `index` of every component of a payload"""
    rows = tuple(t.view(torch.uint8)[index].view(t.dtype) for t in payload_parts(x))
    return rows if isinstance(x, tuple) else rows[0]


def cat_payloads(parts: Sequence[Payload]) -> Payload:
    components = zip(*(payload_parts(p) for p in parts))
    joined = tuple(
        torch.cat([t.view(torch.uint8) for t in c]).view(c[0].dtype) for c in components
    )
    return joined if isinstance(parts[0], tuple) else joined[0]


def payload_equal(a: Payload, b: Payload) -> bool:
    """Same components, dtypes, shapes and bytes (compared as uint8: arbitrary e4m3 scale bytes can be NaN)"""
    a_parts, b_parts = payload_parts(a), payload_parts(b)
    return len(a_parts) == len(b_parts) and all(
        x.dtype == y.dtype
        and x.shape == y.shape
        and torch.equal(x.view(torch.uint8), y.view(torch.uint8))
        for x, y in zip(a_parts, b_parts)
    )


def combine_recipe(buffer: nixl_ep_v2.ElasticBuffer):
    # Without `allow_multiple_reduction` the combine input is already reduced within the rank.
    return (
        True,
        buffer.allow_multiple_reduction and buffer._get_mode() == "hybrid",
    )


# Bound each large float32 reference intermediate; the final BF16 rows remain allocated.
COMBINE_SLICE_BYTES = 1 << 30


class Case:
    """Deterministic data: rank `r` in phase `p` routes by `seed(p, r)` over the active experts and token `t` carries
    `payload(global idx)`, so every rank can rebuild what it should have received from whom.
    """

    def __init__(self, buffer: nixl_ep_v2.ElasticBuffer, args: argparse.Namespace):
        self.buffer = buffer
        self.args = args
        self.dtype = getattr(args, "dtype", "bf16")
        self.max_so, self.max_su = (
            buffer.max_num_scaleout_ranks,
            buffer.max_num_scaleup_ranks,
        )
        self.num_max_tokens_per_rank = args.num_tokens
        self.hidden, self.num_topk, self.num_experts = (
            args.hidden,
            args.num_topk,
            args.num_experts,
        )

    def extent(self, phase: List[int]) -> Tuple[int, int]:
        return extent_of(phase, self.max_su, rank_mapping(self.buffer))

    def flat_idx(self, world_rank: int, phase: List[int]) -> int:
        # The kernels encode a token's source as `scaleout * num_scaleup_ranks + scaleup` (over the extent)
        _, su_extent = self.extent(phase)
        domain, position = rank_coordinates(self.buffer, world_rank)
        return domain * su_extent + position

    def active_expert_mask(self, phase: List[int]) -> torch.Tensor:
        epr = self.buffer.num_experts_per_rank
        active_world = torch.tensor(phase, device="cuda")
        owner = torch.arange(self.num_experts, device="cuda") // epr
        return torch.isin(owner, active_world)

    def reaches_me(self, topk_idx: torch.Tensor, phase: List[int]) -> torch.Tensor:
        """Per token of a source's routing: does one of its selections name an expert this rank owns at its fixed world rank"""
        return (self.buffer.world_rank in phase) & (
            (topk_idx // self.buffer.num_experts_per_rank) == self.buffer.world_rank
        ).any(dim=1)

    def num_tokens_of(self, world_rank: int) -> int:
        # Every world rank sends a different number of tokens (the lower the rank, the more), at least one
        return max(1, self.num_max_tokens_per_rank - world_rank)

    def routing(
        self, phase_idx: int, world_rank: int, phase: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(topk_idx, topk_weights)` of world rank `world_rank` in phase `phase_idx`: distinct active experts, positive weights"""
        g = torch.Generator(device="cuda")
        g.manual_seed(self.args.seed * 1000003 + phase_idx * 4099 + world_rank + 1)
        return sample_routing(
            g,
            self.num_tokens_of(world_rank),
            self.num_experts,
            self.active_expert_mask(phase),
            self.num_topk,
        )

    def global_idx(self, world_rank: int, phase: List[int]) -> torch.Tensor:
        return self.flat_idx(
            world_rank, phase
        ) * self.num_max_tokens_per_rank + torch.arange(
            self.num_tokens_of(world_rank), device="cuda"
        )

    def payload_indices(self, world_rank: int) -> torch.Tensor:
        return world_rank * self.num_max_tokens_per_rank + torch.arange(
            self.num_tokens_of(world_rank), device="cuda"
        )

    def payload(self, global_idx: torch.Tensor) -> Payload:
        x = self.pre_combine_data(global_idx.clamp_min(0), 1)[:, 0, :]
        if self.dtype == "fp8":
            return per_token_cast_to_fp8(x)
        if self.dtype == "nvfp4":
            return as_nvfp4(x)
        return x

    def pre_combine_data(
        self, global_idx: torch.Tensor, num_topk: Optional[int] = None
    ) -> torch.Tensor:
        # The payload's modulus is the declared maximum world size, which every process and every cohort agrees on
        return generate_pre_combine_data(
            global_idx,
            self.num_max_tokens_per_rank,
            self.num_topk if num_topk is None else num_topk,
            self.hidden,
            num_ranks=self.max_so * self.max_su,
            seed=self.args.seed,
        )

    def dispatch_kwargs(self, do_cpu_sync: bool) -> dict:
        """The `dispatch` arguments beside the inputs, shared by the eager, perf and graph paths"""
        return dict(
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            expert_alignment=1,
            do_handle_copy=True,
            do_cpu_sync=do_cpu_sync,
        )

    def combine_input(
        self, src_global_idx: torch.Tensor, recv_topk_idx: torch.Tensor
    ) -> torch.Tensor:
        """The combine input from the dispatch output: the pre-combine data of every received row (rows past the real
        count, when present, carry -1 and are never read), zeroed where the selection is not this rank's, reduced within the rank
        """
        y = torch.empty(
            (src_global_idx.numel(), self.hidden), dtype=torch.bfloat16, device="cuda"
        )
        for rows in self.row_slices(y.shape[0]):
            local_y = self.pre_combine_data(src_global_idx[rows].clamp_min(0))
            local_y[recv_topk_idx[rows] == -1] = 0
            y[rows] = ordered_accumulate(local_y)
        return y

    def combine_reference(
        self, phase: List[int], topk_idx: torch.Tensor
    ) -> torch.Tensor:
        """This rank's expected combine output in the phase for `topk_idx` (dropped selections hold -1 and contribute zero)"""
        active = (topk_idx >= 0) & self.active_expert_mask(phase)[topk_idx.clamp_min(0)]
        topk_idx = torch.where(active, topk_idx, -1)
        topk_idx = topology_experts(self.buffer, topk_idx)
        global_idx = self.global_idx(self.buffer.world_rank, phase)
        ref = torch.empty(
            (global_idx.numel(), self.hidden), dtype=torch.bfloat16, device="cuda"
        )
        for rows in self.row_slices(ref.shape[0]):
            ref_y = self.pre_combine_data(global_idx[rows])
            ref_y[~active[rows]] = 0
            ref[rows] = ref_combine(
                ref_y,
                topk_idx[rows],
                self.max_so,
                self.max_su,
                self.num_experts,
                None,
                *combine_recipe(self.buffer),
            )
        return ref

    def row_slices(self, num_rows: int) -> List[slice]:
        """Bound temporary memory without changing a token's reduction order."""
        step = max(1, COMBINE_SLICE_BYTES // (self.num_topk * self.hidden * 4))
        return [slice(start, start + step) for start in range(0, num_rows, step)]

    def expected_receive(
        self, phase_idx: int, phase: List[int]
    ) -> Tuple[torch.Tensor, Payload, torch.Tensor, torch.Tensor]:
        """What this rank receives in the phase, sorted by source global index: `(global_idx, x, local_topk_idx, weights)`"""
        me = self.buffer.world_rank
        epr = self.buffer.num_experts_per_rank
        chunks = []
        for src in phase:
            topk_idx, topk_weights = self.routing(phase_idx, src, phase)
            owner = topk_idx // epr
            mine = self.reaches_me(topk_idx, phase)
            gidx = self.global_idx(src, phase)[mine]
            local_idx = torch.where(
                owner[mine] == me,
                topk_idx[mine] - me * epr,
                torch.full_like(topk_idx[mine], -1),
            )
            chunks.append(
                (
                    gidx,
                    payload_rows(self.payload(self.payload_indices(src)), mine),
                    local_idx,
                    topk_weights[mine],
                )
            )
        gidx = torch.cat([c[0] for c in chunks])
        order = torch.argsort(gidx)
        return (
            gidx[order],
            payload_rows(cat_payloads([c[1] for c in chunks]), order),
            torch.cat([c[2] for c in chunks])[order],
            torch.cat([c[3] for c in chunks])[order],
        )


def validate_plan(
    case: Case, plan: List[List[int]], present: Optional[List[int]] = None
) -> None:
    """Check phase IDs and initial-mapping shapes, except for later reconnect phases.

    Activation checks later reconnect phases against their prepared mappings.
    """
    assert plan, "The test plan must contain at least one phase"
    max_so, max_su = case.max_so, case.max_su
    for phase_idx, phase in enumerate(plan):
        assert phase, "A topology phase must be nonempty"
        assert all(0 <= r < max_so * max_su for r in phase) and len(set(phase)) == len(
            phase
        ), f"Phase {phase} names a rank outside the {max_so}x{max_su} world or a rank twice"
        if present is not None:
            assert set(phase) <= set(
                present
            ), f"Phase {phase} names ranks that are not in the job {present}"
        if (phase_idx == 0 or not case.args.reconnect_on_leave) and set(
            phase
        ) <= rank_mapping(case.buffer).keys():
            case.buffer._masks_of(phase)  # Future joiners are checked after connection.


def check_dispatch(
    case: Case,
    phase_idx: int,
    phase: List[int],
    recv_x,
    recv_topk_idx,
    recv_topk_weights,
    handle,
    tag: str,
):
    num_recv = handle.num_recv_tokens
    assert handle.num_recv_tokens_device.item() == num_recv, (
        handle.num_recv_tokens_device.item(),
        num_recv,
    )
    gidx = handle.recv_src_metadata[:num_recv, 0]
    order = torch.argsort(gidx)
    exp_gidx, exp_x, exp_idx, exp_w = case.expected_receive(phase_idx, phase)
    assert (
        gidx.numel() == exp_gidx.numel()
    ), f"[{tag}] received {gidx.numel()} tokens, expected {exp_gidx.numel()}"
    assert torch.equal(
        gidx[order], exp_gidx
    ), f"[{tag}] received source tokens differ from the expectation"
    assert payload_equal(
        payload_rows(payload_rows(recv_x, slice(num_recv)), order), exp_x
    ), f"[{tag}] received payload mismatch ({case.dtype})"
    assert torch.equal(
        recv_topk_idx[:num_recv][order].to(torch.int64), exp_idx.to(torch.int64)
    ), f"[{tag}] local expert indices mismatch"
    assert torch.equal(
        recv_topk_weights[:num_recv][order], exp_w
    ), f"[{tag}] weights mismatch"
    return num_recv


def run_eager(case: Case, phase_idx: int, phase: List[int], tag: str):
    buffer = case.buffer
    me = buffer.world_rank
    topk_idx, topk_weights = case.routing(phase_idx, me, phase)
    x = case.payload(case.payload_indices(me))
    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        **case.dispatch_kwargs(do_cpu_sync=True),
    )
    torch.cuda.synchronize()
    assert torch.equal(handle.topk_idx, topk_idx), "handle changed external expert IDs"
    num_recv = check_dispatch(
        case, phase_idx, phase, recv_x, recv_topk_idx, recv_topk_weights, handle, tag
    )

    # Combine: the pre-combine data is a function of the source token, reduced within the rank first (the CPU sync
    # made the count exact: every row is a received one)
    y = case.combine_input(
        handle.recv_src_metadata[:num_recv, 0], recv_topk_idx[:num_recv]
    )
    combined_x, _ = buffer.combine(x=y, handle=handle)
    torch.cuda.synchronize()

    ref = case.combine_reference(phase, topk_idx)
    assert torch.equal(
        combined_x, ref
    ), f"[{tag}] combined_x mismatch, diff={calc_diff(combined_x, ref)}"
    return num_recv


def remote_payload_rows(
    topk_rows: Sequence[Sequence[int]],
    slots_per_rank: int,
    max_scaleup_ranks: int,
    source_domain: int,
    active_ranks: Sequence[int],
    mapping=None,
) -> Tuple[int, int]:
    """Per-source token/destination pairs outside its domain, deduplicated by domain and by rank."""
    active = set(active_ranks)
    mapping = mapping or {rank: rank for rank in active}
    domains = ranks = 0
    for row in topk_rows:
        remote = {
            slot // slots_per_rank
            for slot in row
            if slot >= 0
            and slot // slots_per_rank in active
            and mapping[slot // slots_per_rank] // max_scaleup_ranks != source_domain
        }
        domains += len({mapping[rank] // max_scaleup_ranks for rank in remote})
        ranks += len(remote)
    return domains, ranks


def pair_bytes(case: Case, pc: "PairCase", phase: List[int]) -> Dict[str, int]:
    """Payload only: activation/scale bytes, excluding routing metadata, padding and completion traffic.
    Remote bytes include both directions attributable to this source's routing, not a physical NIC's traffic.
    Received/input bytes include local delivery; their pair bandwidth is effective throughput, not NVLink bandwidth.
    """
    domains, ranks = remote_payload_rows(
        pc.dispatch_args["topk_idx"].tolist(),
        case.buffer.num_experts_per_rank,
        case.buffer.max_num_scaleup_ranks,
        rank_coordinates(case.buffer)[0],
        phase,
        rank_mapping(case.buffer),
    )
    return dict(
        dispatch_received_payload=pc.num_recv * pc.dispatch_bytes_per_token,
        combine_input_payload=pc.num_recv * pc.combine_bytes_per_token,
        dispatch_remote_payload=domains * pc.dispatch_bytes_per_token,
        combine_remote_payload=(
            domains if case.buffer.allow_multiple_reduction else ranks
        )
        * pc.combine_bytes_per_token,
    )


def benchmark_options(
    args: argparse.Namespace, phase_idx: int, perf_phases: set
) -> Tuple[bool, bool]:
    selected = phase_idx in perf_phases
    return (
        (args.benchmark or selected) and not args.skip_perf_test,
        args.kineto or selected,
    )


def run_perf(
    case: Case,
    phase_idx: int,
    phase: List[int],
    mode: str,
    args: argparse.Namespace,
    *,
    measure_events: bool = True,
    measure_kernels: bool = True,
) -> dict:
    """Mean kernel and CUDA-event call/pair times, with L2 flushing and active-rank alignment before each sample.
    PDL kernel durations include dependency waits and can overlap; their sum is not end-to-end latency.
    Each isolated call starts with flushed L2; the pair flushes only before dispatch, so its time need not equal their sum.
    One `PERF` line per active rank; eager dispatch includes its CPU synchronization and output allocation.
    """
    buffer = case.buffer
    me = buffer.world_rank
    pc = PairCase(case, phase_idx, phase, args)
    pc.pair()
    torch.cuda.synchronize()

    def dispatch():
        return buffer.dispatch(**pc.dispatch_args)

    def combine():
        return buffer.combine(**pc.combine_args)

    record = dict(tokens=pc.num_tokens, num_recv=pc.num_recv)
    fields = {}
    if measure_kernels:
        t_d, copy_d = bench_kineto(
            dispatch,
            kernel_names=("dispatch_impl", "dispatch_copy_epilogue_impl"),
            barrier=buffer.barrier,
        )
        t_c, red_c = bench_kineto(
            combine,
            kernel_names=("combine_impl", "combine_reduce_epilogue_impl"),
            barrier=buffer.barrier,
        )
        record["kineto"] = dict(
            zip(
                ("dispatch_us", "copy_us", "combine_us", "reduce_us"),
                (
                    float(t) * 1e6 if t is not None else None
                    for t in (t_d, copy_d, t_c, red_c)
                ),
            )
        )
        fields.update(record["kineto"])
    if measure_events:
        record["events"] = {}
        for name, fn in (
            ("dispatch", dispatch),
            ("combine", combine),
            ("pair", pc.pair),
        ):
            times = bench(
                fn, num_warmups=5, num_tests=20, flush_l2=True, barrier=buffer.barrier
            )
            record["events"][name] = dict(
                zip(("avg_us", "min_us", "max_us"), (float(t) * 1e6 for t in times))
            )
            fields[f"{name}_wall_us"] = record["events"][name]["avg_us"]
    torch.cuda.synchronize()
    record["bytes"] = sizes = pair_bytes(case, pc, phase)
    if measure_events:
        pair = record["events"]["pair"]
        fields.update(pair_min_us=pair["min_us"], pair_max_us=pair["max_us"])
        # bytes / microseconds / 1000 = GB/s; a zero duration cannot yield a bandwidth.
        for name, size in (
            (
                "pair_payload_gbps",
                sizes["dispatch_received_payload"] + sizes["combine_input_payload"],
            ),
            (
                "remote_payload_gbps",
                sizes["dispatch_remote_payload"] + sizes["combine_remote_payload"],
            ),
        ):
            if pair["avg_us"] > 0:
                record[name] = fields[name] = size / pair["avg_us"] / 1000
    so_extent, su_extent = case.extent(phase)
    print(
        f"PERF tag={args.perf_tag} mode={mode} phase={phase_idx} live={len(phase)} bounds={so_extent}x{su_extent} rank={me} "
        f"tokens={pc.num_tokens} "
        + " ".join(
            f"{name}={value:.1f}" if value is not None else f"{name}=N/A"
            for name, value in fields.items()
        ),
        flush=True,
    )
    return record


def save_results(
    cohort: NixlBootstrap, path: str, config: dict, plan: List[List[int]], records: dict
) -> Optional[str]:
    ranks = cohort._all_gather("results", records)
    error = None
    if cohort.rank == 0:
        try:
            with tempfile.TemporaryDirectory(
                dir=os.path.dirname(os.path.abspath(path))
            ) as directory:
                temporary_path = os.path.join(directory, "results.json")
                with open(temporary_path, "w") as output:
                    json.dump(
                        dict(schema_version=1, config=config, plan=plan, ranks=ranks),
                        output,
                        indent=2,
                        allow_nan=False,
                    )
                    output.write("\n")
                os.replace(temporary_path, path)
            print(f"Results saved to {path}", flush=True)
        except (OSError, TypeError, ValueError) as exc:
            error = f"Failed to save results to {path}: {type(exc).__name__}: {exc}"
    return cohort._all_gather("results-written", error)[0]


def print_profile(
    buffer: nixl_ep_v2.ElasticBuffer, phase_idx: int, execution: str
) -> None:
    for name, bounds in summarize_profile(buffer.read_profile()).items():
        print(
            f"PROFILE rank={buffer.world_rank} phase={phase_idx} execution={execution} stage={name} "
            f'first_us={bounds["first_us"]:.3f} last_us={bounds["last_us"]:.3f} writers={bounds["writers"]}',
            flush=True,
        )


class PairCase:
    """Prepare a phase's inputs without communication; `pair` runs dispatch + combine on its active ranks."""

    def __init__(
        self, case: Case, phase_idx: int, phase: List[int], args: argparse.Namespace
    ):
        buffer = case.buffer
        me = buffer.world_rank
        n = args.perf_tokens if args.perf_tokens > 0 else case.num_max_tokens_per_rank
        assert n <= case.num_max_tokens_per_rank, (n, case.num_max_tokens_per_rank)
        self.case, self.num_tokens = case, n
        g = torch.Generator(device="cuda")
        g.manual_seed(args.seed * 7919 + phase_idx * 104729 + me + 12345)
        topk_idx, topk_weights = sample_routing(
            g, n, case.num_experts, case.active_expert_mask(phase), case.num_topk
        )
        x = torch.randn(
            n, case.hidden, dtype=torch.bfloat16, device="cuda", generator=g
        )
        if case.dtype == "fp8":
            x = per_token_cast_to_fp8(x)
        elif case.dtype == "nvfp4":
            x = as_nvfp4(x)
        self.dispatch_args = dict(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            **case.dispatch_kwargs(do_cpu_sync=True),
        )
        payload = x if isinstance(x, tuple) else (x,)
        self.dispatch_bytes_per_token = (
            sum(t.numel() * t.element_size() for t in payload) // n
        )
        # Each active source can contribute at most n deduplicated rows; allocate before the measured transition.
        self.y = torch.randn(
            n * len(phase),
            case.hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=g,
        )
        self.combine_bytes_per_token = case.hidden * self.y.element_size()

    def pair(self):
        _, _, _, handle, _ = self.case.buffer.dispatch(**self.dispatch_args)
        self.num_recv = handle.num_recv_tokens
        self.combine_args = dict(x=self.y[: self.num_recv], handle=handle)
        self.case.buffer.combine(**self.combine_args)


def run_feed(
    pc: PairCase, num_pairs: int
) -> Tuple[torch.cuda.Event, List[torch.cuda.Event], List[float]]:
    """Time every pair; return the first start event, all end events and per-pair microseconds."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
    for s, e in zip(starts, ends):
        s.record()
        pc.pair()
        e.record()
    torch.cuda.synchronize()
    return starts[0], ends, [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]


# The barriers `capture` runs (before the warm-ups, before the capture, after it), for idle ranks that mirror them
NUM_CAPTURE_BARRIERS = 3


def capture(forward, barrier=dist.barrier, do_capture=True):
    # Warm up eagerly (JIT compile, first-touch allocations), then capture one graph.
    # `do_capture=False` runs the same warm-ups and barriers without capturing
    torch.cuda.synchronize()
    barrier()
    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(2):
            forward()
    torch.cuda.synchronize()
    barrier()
    graph, outputs = None, None
    if do_capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = forward()
    torch.cuda.synchronize()
    barrier()
    return graph, outputs


class GraphCase:
    """One graph of dispatch (no CPU sync) + combine, with routing sampled over `full_phase`,
    and replayed in every later phase against the reference of that phase (dropped selections included).
    """

    # The phase index the graph's routing is sampled at (`Case.routing` seeds by it): one no plan reaches, apart from every eager phase
    CAPTURE_ROUTING_PHASE = 10_000
    REROUTE_ROUTING_PHASE = 20_000
    STATS_GUARD = 104729

    def __init__(
        self,
        case: Case,
        full_phase: List[int],
        capture_phase_of: Dict[int, List[int]],
        barrier_fn: Callable[[], None],
    ):
        self.case = case
        # The phase whose routing each rank captured; warmup may use a smaller active world.
        self.capture_phase_of = capture_phase_of
        # Whole-job barrier, including idle ranks.
        self.barrier = barrier_fn
        me = case.buffer.world_rank
        self.topk_idx, self.topk_weights = case.routing(
            self.CAPTURE_ROUTING_PHASE, me, full_phase
        )
        self.x = case.payload(case.payload_indices(me))
        # Fixed local slots and a guard word: replay must neither resize statistics nor write past them.
        self.stats_storage = torch.full(
            (case.buffer.num_experts_per_rank + 1,),
            self.STATS_GUARD,
            dtype=torch.int,
            device="cuda",
        )
        self.stats = self.stats_storage[:-1]
        self.stats.zero_()
        self.handle: Optional[nixl_ep_v2.EPHandle] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.outputs: Optional[torch.Tensor] = None
        self.recv_x: Optional[Payload] = None
        self.recv_topk_idx: Optional[torch.Tensor] = None
        self.recv_topk_weights: Optional[torch.Tensor] = None

    def forward(self):
        """One dispatch + combine; outputs are retained only from capture."""
        case = self.case
        dispatch_args = dict(
            x=self.x,
            topk_idx=self.topk_idx,
            topk_weights=self.topk_weights,
            **case.dispatch_kwargs(do_cpu_sync=False),
        )
        if torch.cuda.is_current_stream_capturing():
            # The refusal must happen before communication is captured; keep the context valid and capture the real pair next.
            try:
                case.buffer.dispatch(
                    cumulative_local_expert_recv_stats=self.stats[:-1], **dispatch_args
                )
            except RuntimeError as error:
                assert "cumulative_local_expert_recv_stats" in str(error), str(error)
                assert "fixed local expert slots" in str(error), str(error)
            else:
                raise AssertionError(
                    "captured dispatch accepted statistics shorter than the fixed local expert slots"
                )
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = case.buffer.dispatch(
            cumulative_local_expert_recv_stats=self.stats, **dispatch_args
        )
        # Without a CPU sync the count is only on the GPU: the combine input covers the worst-case-sized output
        y = case.combine_input(handle.recv_src_metadata[:, 0], recv_topk_idx)
        combined_x, _ = case.buffer.combine(x=y, handle=handle)
        return combined_x, handle, recv_x, recv_topk_idx, recv_topk_weights

    # The barriers the shared `capture` runs (before the warm-ups, before the capture, after it), for idle ranks that mirror them
    NUM_CAPTURE_BARRIERS = NUM_CAPTURE_BARRIERS

    def capture(self, accompany: bool = False):
        # The capture protocol over this rank's barrier; the other active ranks run it in
        # lockstep (`accompany`: warm-ups and barriers only, for a rank that has its graph), idle ranks mirror the barriers
        graph, outputs = capture(self.forward, self.barrier, do_capture=not accompany)
        if not accompany:
            self.graph = graph
            (
                self.outputs,
                self.handle,
                self.recv_x,
                self.recv_topk_idx,
                self.recv_topk_weights,
            ) = outputs

    def _replay_routing(self, src, phase, routing_phase_idx):
        if routing_phase_idx is None:
            routing_phase_idx = self.CAPTURE_ROUTING_PHASE
            phase = self.capture_phase_of[src]
        return self.case.routing(routing_phase_idx, src, phase)

    def replay_and_check(
        self, phase: List[int], tag: str, routing_phase_idx: Optional[int] = None
    ):
        """Replay captured or fresh routing twice, checking outputs and cumulative counts."""
        assert self.graph is not None and self.outputs is not None
        topk_idx, weights = self._replay_routing(
            self.case.buffer.world_rank, phase, routing_phase_idx
        )
        self.topk_idx.copy_(topk_idx)
        self.topk_weights.copy_(weights)
        self.stats.zero_()
        for num_replays in (1, 2):
            # A previous combine result can equal the reference; require each replay to write it.
            self.outputs.fill_(float("nan"))
            self.graph.replay()
            torch.cuda.synchronize()
            num_recv = self.check(
                phase, tag, num_replays=num_replays, routing_phase_idx=routing_phase_idx
            )
            if num_replays == 1 and getattr(self.case.args, "profile", False):
                self.case.buffer.read_profile()  # Keep one replay per profile read.
        return num_recv

    def check(
        self,
        phase: List[int],
        tag: str,
        num_replays: int = 1,
        routing_phase_idx: Optional[int] = None,
    ):
        """Check outputs for the replay's routing; num_replays counts executions since statistics reset."""
        case = self.case
        me = case.buffer.world_rank
        assert self.handle is not None and self.recv_x is not None
        combined_x = self.outputs
        assert torch.equal(
            self.handle.topk_idx, self.topk_idx
        ), "handle changed external expert IDs"

        # Drop departed physical owners while deriving source metadata from the current extent.
        ref = case.combine_reference(phase, self.topk_idx)
        assert torch.equal(
            combined_x, ref
        ), f"[{tag}] graph combined_x mismatch, diff={calc_diff(combined_x, ref)}"
        # Source metadata follows the current extent; payloads and expert owners use stable world IDs.
        expected_src_parts, expected_x_parts = [], []
        expected_idx_parts, expected_w_parts = [], []
        epr = case.buffer.num_experts_per_rank
        expected_stats = torch.zeros(epr, dtype=torch.int64, device="cuda")
        for src in phase:
            src_idx, src_w = self._replay_routing(src, phase, routing_phase_idx)
            mine = case.reaches_me(src_idx, phase)
            expected_src_parts.append(case.global_idx(src, phase)[mine])
            expected_x_parts.append(
                payload_rows(case.payload(case.payload_indices(src)), mine)
            )
            expected_idx_parts.append(
                torch.where(src_idx[mine] // epr == me, src_idx[mine] - me * epr, -1)
            )
            expected_w_parts.append(src_w[mine])
            local_idx = src_idx[src_idx // epr == me].to(torch.int64) - me * epr
            expected_stats += torch.bincount(local_idx, minlength=epr)
        expected_src = torch.cat(expected_src_parts)
        expected_x = cat_payloads(expected_x_parts)
        expected = expected_src.numel()
        num_recv = self.handle.num_recv_tokens_device.item()
        assert num_recv == expected, (
            f"[{tag}] graph receive count {num_recv}, expected {expected}; "
            f"psum per position {self.handle.psum_num_recv_tokens_per_scaleup_rank.tolist()}, "
            f"captured at {self.capture_phase_of[me]}"
        )
        received_src = self.handle.recv_src_metadata[:num_recv, 0]
        order, expected_order = torch.argsort(received_src), torch.argsort(expected_src)
        assert torch.equal(
            received_src[order], expected_src[expected_order]
        ), f"[{tag}] graph received source tokens mismatch"
        assert payload_equal(
            payload_rows(payload_rows(self.recv_x, slice(num_recv)), order),
            payload_rows(expected_x, expected_order),
        ), f"[{tag}] graph received payload mismatch ({case.dtype})"
        assert self.recv_topk_idx is not None and self.recv_topk_weights is not None
        assert torch.equal(
            self.recv_topk_idx[:num_recv][order].to(torch.int64),
            torch.cat(expected_idx_parts)[expected_order].to(torch.int64),
        ), f"[{tag}] graph local expert indices mismatch"
        assert torch.equal(
            self.recv_topk_weights[:num_recv][order],
            torch.cat(expected_w_parts)[expected_order],
        ), f"[{tag}] graph weights mismatch"
        assert torch.equal(
            self.stats[:epr].to(torch.int64), expected_stats * num_replays
        ), f"[{tag}] graph expert statistics mismatch"
        assert self.stats.shape == (epr,)
        assert torch.equal(
            self.handle.num_unaligned_recv_tokens_per_expert, expected_stats
        ), f"[{tag}] graph per-expert receive counts mismatch"
        assert torch.equal(
            self.handle.psum_num_recv_tokens_per_expert, expected_stats.cumsum(0)
        ), f"[{tag}] graph per-expert receive prefix sums mismatch"
        assert (
            self.stats_storage[-1].item() == self.STATS_GUARD
        ), f"[{tag}] graph wrote past statistics capacity"
        return num_recv

    def release(self):
        self.recv_topk_idx = self.recv_topk_weights = None
        del self.outputs, self.recv_x, self.graph, self.stats, self.stats_storage
        self.outputs = self.recv_x = self.graph = self.stats = self.stats_storage = None
        torch.cuda.synchronize()
        self.handle = None


def expected_mode(
    phase: Sequence[int], max_num_scaleup_ranks: int, mapping=None
) -> str:
    """
    The mode the host must have derived for the active ranks `phase` (world ranks), stated independently of the library:
    direct iff exactly one NVLink domain is active, hybrid otherwise.
    """
    mapping = mapping or {r: r for r in phase}
    one_domain = len({mapping[r] // max_num_scaleup_ranks for r in phase}) == 1
    return "direct" if one_domain else "hybrid"


class TransitionTimings(TypedDict, total=False):
    num_connect: int
    num_disconnect: int
    ev_activate_start: torch.cuda.Event
    ev_activate_end: torch.cuda.Event
    connect: float
    activate: float
    disconnect: float
    barrier: float
    total: float


def switch_world(
    buffer: nixl_ep_v2.ElasticBuffer,
    phase: List[int],
    barrier_fn: Callable[[], None],
    store_timeout_secs: float,
    reconnect_on_leave: bool = False,
    timings: Optional[TransitionTimings] = None,
) -> None:
    """
    Drain the old world, activate participants and optionally retire peers; `barrier_fn` spans all processes.
    With reconnection, departing buffers return to a singleton and participation follows `phase`.
    Otherwise departing ranks adopt `phase` to become inactive while retaining connections.
    `timings`: host ms per step, peer counts and two CUDA events around activation.
    `store_timeout_secs` bounds the Store wait of `connect_ranks`.
    """
    t: TransitionTimings = timings if timings is not None else {}
    t0 = time.perf_counter()
    buffer.barrier(with_cpu_sync=True)
    barrier_fn()
    t_drained = time.perf_counter()
    me = buffer.world_rank
    active = me in phase
    update_membership = active or (
        not reconnect_on_leave and me in buffer.get_active_ranks()
    )
    connected = set(buffer.get_connected_ranks())
    if reconnect_on_leave:
        needed = (set(phase) - {me}) if active else set()
    else:
        needed = connected | ((set(phase) - {me}) if active else set())
    joining = sorted(needed - connected)
    t["num_connect"] = len(joining)
    t1 = time.perf_counter()
    if joining:
        buffer.connect_ranks(
            joining,
            key_prefix="adaptive-elastic/EP_NIXL",
            timeout_secs=store_timeout_secs,
        )
    t2 = time.perf_counter()
    torch.cuda.synchronize()
    barrier_fn()
    t3 = time.perf_counter()
    t["ev_activate_start"] = torch.cuda.Event(enable_timing=True)
    t["ev_activate_start"].record()
    if update_membership:
        buffer.set_active_ranks(phase)
    t["ev_activate_end"] = torch.cuda.Event(enable_timing=True)
    t["ev_activate_end"].record()
    t4 = time.perf_counter()
    torch.cuda.synchronize()
    barrier_fn()
    t5 = time.perf_counter()
    if not reconnect_on_leave:
        assert (me in buffer.get_active_ranks()) == active
    if update_membership:
        assert buffer.get_active_ranks() == sorted(phase)
        expected = expected_mode(
            phase, buffer.max_num_scaleup_ranks, rank_mapping(buffer)
        )
        assert (
            buffer._get_mode() == expected
        ), f"the host derived {buffer._get_mode()} for {phase}, expected {expected}"

    # Participants retire excluded peers; departing buffers retire all peers and remain singletons.
    leaving = sorted(connected - needed) if reconnect_on_leave else []
    t["num_disconnect"] = len(leaving)
    t6 = time.perf_counter()
    if leaving:
        buffer.disconnect_ranks(leaving)
    t7 = time.perf_counter()
    torch.cuda.synchronize()
    barrier_fn()
    t8 = time.perf_counter()
    if reconnect_on_leave and not active:
        assert buffer.get_active_ranks() == [me]
        assert not buffer.get_connected_ranks()
    t.update(
        {
            "connect": (t2 - t1) * 1e3,
            "activate": (t4 - t3) * 1e3,
            "disconnect": (t7 - t6) * 1e3,
            "barrier": ((t_drained - t0) + (t3 - t2) + (t5 - t4) + (t8 - t7)) * 1e3,
            "total": (t8 - t0) * 1e3,
        }
    )


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks = init_test_environment(local_rank, num_local_ranks, args.seed)
    # The job's Store, served by the rank 0 process for the whole run; every Store wait is bounded by `--store-timeout-secs`
    store = dist.TCPStore(
        os.getenv("MASTER_ADDR", "127.0.0.1"),
        int(os.getenv("MASTER_PORT", "8361")),
        world_size=num_ranks,
        is_master=rank_idx == 0,
        timeout=timedelta(seconds=args.store_timeout_secs),
    )
    # The test's own cohort over the job: barriers, and its NVLink-domain size when the positions are offset
    cohort = NixlBootstrap(
        store,
        rank_idx,
        num_ranks,
        timeout_seconds=args.store_timeout_secs,
    )

    def barrier():
        # A Store barrier orders nothing on the device: finish this rank's queued CUDA work first
        torch.cuda.synchronize()
        cohort.barrier("test")

    def cohort_print(message: str):
        if local_rank == 0:
            print(message, flush=True)
        barrier()

    # Public fixture IDs may include holes; hardware discovery determines their coordinates.
    placement = {}
    if args.position_offset:
        group_su = cohort.num_scaleup_ranks
        placement = dict(
            world_ranks=[
                (rank // group_su) * args.max_scaleup_ranks
                + args.position_offset
                + rank % group_su
                for rank in range(num_ranks)
            ]
        )
    buffer = make_buffer(
        args,
        rank_idx,
        num_ranks,
        store,
        num_group_scaleup_ranks=cohort.num_scaleup_ranks,
        max_num_scaleup_ranks=args.max_scaleup_ranks,
        **placement,
    )
    max_so, max_su = buffer.max_num_scaleout_ranks, buffer.max_num_scaleup_ranks
    mode = buffer._get_mode()
    present = (
        buffer.get_active_ranks()
    )  # the whole job, wherever it sits in the declared maximum world
    assert mode == expected_mode(
        present, max_su, rank_mapping(buffer)
    ), f"setup derived {mode} for {present}"
    plan = (
        load_plan(args.plan)
        if args.plan
        else default_plan(max_su, present, rank_mapping(buffer))
    )
    cohort_print(
        f"Elastic test: world {max_so}x{max_su}, world rank {buffer.world_rank}, internal coordinates {rank_coordinates(buffer)}, "
        f"initial mode {mode}, launch grid {buffer.num_launch_sms}, "
        f"connected {buffer.get_connected_ranks()}, active {present}\n"
        f"Plan: {plan}"
    )
    case = Case(buffer, args)
    validate_plan(case, plan, present)
    profile = getattr(args, "profile", False)

    me = buffer.world_rank
    # Phases whose active ranks get benchmark timings, e.g. the phase a world grew or shrank into
    perf_phases = set(int(p) for p in args.perf_phases.split(",") if p.strip() != "")
    assert all(
        0 <= p < len(plan) for p in perf_phases
    ), f"--perf-phases {sorted(perf_phases)} outside the plan of {len(plan)} phases"
    phase_records, transition_records = [], []

    # The phase each rank captures its graph at: the first phase of the plan it takes part in (reversed: the first wins)
    capture_phase_of = {r: list(phase) for phase in reversed(plan) for r in phase}

    feed = (
        args.feed_pairs
    )  # > 0: time every switch under a synthetic feed of dispatch + combine pairs
    if feed > 0:
        # Warm each participating rank while retaining connections and their mappings.
        warmed: set[int] = set()
        for phase_idx, phase in enumerate(plan):
            if set(phase) <= warmed:
                continue
            switch_world(buffer, phase, barrier, args.store_timeout_secs)
            if me in phase:
                PairCase(case, phase_idx, phase, args).pair()
            barrier()
            warmed.update(phase)
        # Restore initial membership without allocating full-world token buffers.
        switch_world(buffer, present, barrier, args.store_timeout_secs)
    pc = PairCase(case, 0, plan[0], args) if feed > 0 and me in plan[0] else None
    graph: Optional[GraphCase] = None
    captured: set[int] = set()
    # End events and times of the pairs this rank ran at the end of the previous phase.
    pre: Optional[Tuple[List[torch.cuda.Event], List[float]]] = None
    for phase_idx, phase in enumerate(plan):
        tim: TransitionTimings = {}
        switch_world(
            buffer,
            phase,
            barrier,
            args.store_timeout_secs,
            args.reconnect_on_leave,
            timings=tim,
        )
        active = me in phase
        active_mode = buffer._get_mode() if active else None
        if feed > 0 and phase_idx > 0:
            # The switch as an engine sees it: the last pair of the old world, the switch, the first pairs of the new
            prev = plan[phase_idx - 1]
            role = (
                ("stay" if me in prev else "join")
                if active
                else ("leave" if me in prev else "idle")
            )
            gpu_gap = after = float("nan")
            first_list = ""
            times: List[float] = []
            if active:
                assert pc is not None
                first_event, _, times = run_feed(pc, feed)
                first_list = ",".join(f"{v:.0f}" for v in times[:3])
                after = median(times[3:]) if len(times) > 3 else median(times)
                if pre is not None:
                    gpu_gap = pre[0][-1].elapsed_time(
                        first_event
                    )  # ms, same stream and device
            before = median(pre[1]) if pre is not None else float("nan")
            dev_ms = tim["ev_activate_start"].elapsed_time(tim["ev_activate_end"])
            print(
                f"TRANS tag={args.perf_tag} mode={active_mode or 'inactive'} from={phase_idx - 1} to={phase_idx} rank={me} role={role} "
                f'live_before={len(prev)} live_after={len(phase)} num_connect={tim["num_connect"]} num_disconnect={tim["num_disconnect"]} '
                f'connect_ms={tim["connect"]:.2f} activate_ms={tim["activate"]:.3f} '
                f'disconnect_ms={tim["disconnect"]:.2f} barrier_ms={tim["barrier"]:.2f} switch_ms={tim["total"]:.2f} '
                f"activate_dev_ms={dev_ms:.3f} gpu_gap_ms={gpu_gap:.2f} before_us={before:.0f} first_us={first_list} after_us={after:.0f}",
                flush=True,
            )
            if args.save_output:
                transition_records.append(
                    dict(
                        from_phase=phase_idx - 1,
                        to_phase=phase_idx,
                        mode=active_mode,
                        role=role,
                        live_before=len(prev),
                        live_after=len(phase),
                        num_connect=tim["num_connect"],
                        num_disconnect=tim["num_disconnect"],
                        connect_ms=tim["connect"],
                        activate_ms=tim["activate"],
                        disconnect_ms=tim["disconnect"],
                        barrier_ms=tim["barrier"],
                        total_ms=tim["total"],
                        activate_dev_ms=dev_ms,
                        gpu_gap_ms=gpu_gap if pre is not None and active else None,
                        before_us=before if pre is not None else None,
                        first_us=times[:3],
                        after_us=after if times else None,
                    )
                )

        # Ranks active for the first time capture now; the other active ranks run the warm-ups with them, idle ranks mirror the barriers
        new_capturers = sorted(set(phase) - captured)
        if me in new_capturers:
            graph = GraphCase(case, phase, capture_phase_of, barrier)
            graph.capture()
        elif new_capturers and active:
            assert graph is not None
            graph.capture(accompany=True)
        elif new_capturers:
            for _ in range(GraphCase.NUM_CAPTURE_BARRIERS):
                barrier()
        captured |= set(new_capturers)

        if active:
            so_extent, su_extent = case.extent(phase)
            tag = f"rank {me} {active_mode} phase {phase_idx} {phase} (extent {so_extent}x{su_extent})"
            assert graph is not None
            if profile:
                buffer.read_profile()  # Discard warm-up or prior-phase samples before this isolated pair.
            num_recv = run_eager(case, phase_idx, phase, tag)
            if profile:
                print_profile(buffer, phase_idx, "eager")
            num_recv_graph = graph.replay_and_check(phase, tag)
            if profile:
                print_profile(buffer, phase_idx, "graph")
            num_recv_rerouted = graph.replay_and_check(
                phase,
                tag,
                routing_phase_idx=GraphCase.REROUTE_ROUTING_PHASE + phase_idx,
            )
            if profile:
                print_profile(buffer, phase_idx, "graph-rerouted")
            # One line per phase from the lowest active rank (local rank 0 may sit the phase out)
            if me == min(phase):
                print(
                    f" > [{tag}] eager: {num_recv} tokens received, bitwise match; "
                    f"graph replay (captured at phase {graph.capture_phase_of[me]}): {num_recv_graph} tokens received; "
                    f"rerouted: {num_recv_rerouted} tokens received, bitwise match",
                    flush=True,
                )
            measure_events, measure_kernels = benchmark_options(
                args, phase_idx, perf_phases
            )
            performance = (
                run_perf(
                    case,
                    phase_idx,
                    phase,
                    cast(str, active_mode),
                    args,
                    measure_events=measure_events,
                    measure_kernels=measure_kernels,
                )
                if measure_events or measure_kernels
                else None
            )
            if args.save_output:
                phase_records.append(
                    dict(
                        phase=phase_idx,
                        ranks=phase,
                        mode=active_mode,
                        extent=[so_extent, su_extent],
                        eager_num_recv=num_recv,
                        graph_num_recv=num_recv_graph,
                        graph_rerouted_num_recv=num_recv_rerouted,
                        performance=performance,
                    )
                )
        # Prepare the next phase before the final old-world feed, keeping setup out of the transition gap.
        next_pc = None
        if feed > 0 and phase_idx + 1 < len(plan) and me in plan[phase_idx + 1]:
            next_pc = PairCase(case, phase_idx + 1, plan[phase_idx + 1], args)
        barrier()
        # The feed's last pairs of this phase, for the next switch (active ranks only)
        pre = None
        if feed > 0 and active and phase_idx + 1 < len(plan):
            assert pc is not None
            _, ends, times = run_feed(pc, feed)
            pre = (ends, times)
        barrier()
        pc = next_pc
    if graph is not None:
        graph.release()

    # Back to the full world (the host derives its mode) before tearing down
    switch_world(
        buffer, present, barrier, args.store_timeout_secs, args.reconnect_on_leave
    )
    cohort_print("Adaptive elastic test passed")
    save_error = None
    if args.save_output:
        save_error = save_results(
            cohort,
            args.save_output,
            dict(
                vars(args),
                maximum_world=[max_so, max_su],
                torch_version=str(torch.__version__),
            ),
            plan,
            dict(world_rank=me, phases=phase_records, transitions=transition_records),
        )
    buffer.destroy()
    # Rank 0 owns the TCPStore server; keep it alive until all peers finish teardown.
    barrier()
    if save_error is not None and cohort.rank == 0:
        raise RuntimeError(save_error)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-processes", type=int, default=4, help="processes to spawn (one per GPU)"
    )
    parser.add_argument(
        "--num-sms",
        type=int,
        default=24,
        help="fixed dispatch/combine block count for every buffer in the plan",
    )
    parser.add_argument("--num-gpu-timeout-secs", type=int, default=100)
    parser.add_argument(
        "--store-timeout-secs",
        type=int,
        default=1800,
        help="Store wait timeout in seconds (client connection, barriers, "
        "topology discovery and peer metadata exchange)",
    )
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=192,
        help="physical slots in the declared maximum world",
    )
    parser.add_argument("--allow-multiple-reduction", type=int, default=1)
    parser.add_argument("--prefer-overlap-with-compute", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--position-offset",
        type=int,
        default=0,
        help="offset added to public rank IDs within each test fixture domain",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive EP kernels: elastic membership on the NIXL transport"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--plan",
        type=str,
        default="",
        help="inline JSON or JSON file of phases (lists of world ranks); default: a built-in plan",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="CUDA-event call/pair timings in every phase (off by default)",
    )
    parser.add_argument(
        "--kineto",
        action="store_true",
        help="per-kernel timings in every phase (off by default)",
    )
    parser.add_argument(
        "--skip-perf-test",
        action="store_true",
        help="skip event benchmarks, including those selected by --perf-phases; Kineto and feed timings are independent",
    )
    parser.add_argument(
        "--save-output",
        nargs="?",
        const="adaptive_elastic_results.json",
        default="",
        help="save configuration, correctness counts, optional PERF and TRANS records as JSON",
    )
    parser.add_argument(
        "--perf-phases",
        type=str,
        default="",
        help="comma-separated phases whose active ranks get kernel and call timings (PERF lines)",
    )
    parser.add_argument(
        "--perf-tokens",
        type=int,
        default=0,
        help="tokens per active rank in the timed phases (default: --num-tokens)",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp8", "nvfp4"),
        help="dispatch format for checked and timed pairs; combine stays BF16",
    )
    parser.add_argument(
        "--perf-tag", type=str, default="run", help="label of the PERF and TRANS lines"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print phase boundaries for each eager pair and graph replay; "
        "requires NIXL_EP_ADAPTIVE_PROFILE set to 1",
    )
    parser.add_argument(
        "--feed-pairs",
        type=int,
        default=0,
        help="> 0: time every world switch under this many dispatch + combine pairs before and after it (TRANS lines)",
    )
    parser.add_argument(
        "--reconnect-on-leave",
        action="store_true",
        help="retire all peers while a rank sits out; reconnect may reassign internal "
        "positions, so each phase must remain a product of domains and positions "
        "(default: keep connections)",
    )
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=100)
    parser.add_argument(
        "--max-scaleup-ranks",
        type=int,
        default=0,
        help="declared maximum positions per domain (0 = the job's), e.g. 72 for an NVL72 domain",
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    assert (
        not args.profile or os.environ.get("NIXL_EP_ADAPTIVE_PROFILE") == "1"
    ), "--profile requires NIXL_EP_ADAPTIVE_PROFILE=1 before launch"
    assert (
        args.position_offset == 0 or args.max_scaleup_ranks > 0
    ), "--position-offset needs --max-scaleup-ranks"
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
