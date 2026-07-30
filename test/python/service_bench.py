#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Single-node WRITE/READ benchmark over ``nixlServiceAgent``.

The script spawns one worker process per configured rank/GPU on the local
node.  Each worker runs ``benchmark``, which:

* builds a ``nixlServiceAgent`` named after its rank id (DIRECT mode by
  default; any marshal mode via ``--service-mode``);
* registers one VRAM send buffer + one VRAM recv buffer, plus (for
  ``--direction read``) a dedicated per-peer destination buffer on rank 0;
* publishes each rank's MD blob, addresses, and (for RL modes) ping-pong
  reference addresses via ``torch.distributed.all_gather_object``;
* for ``--direction write`` (default): issues one one-sided WRITE per peer
  rank 0 sends to, into ``peer.recv_buffer``, and waits for the matching
  notifications on the receiving peers;
* for ``--direction read``: issues one one-sided READ per peer rank 0
  reads from, pulling into a dedicated per-peer region of its own; peers
  serve the READ passively - and, for RL modes, run the per-iteration
  perturbation themselves, since the source/encoder role moves to them -
  and drain the completion notification exactly as a WRITE destination
  does (notifications always target ``remote_agent``, the peer, regardless
  of direction);
* verifies contents on whichever side ends up holding the transferred
  data - WRITE destinations, or the READ initiator (rank 0).

Run with::

    python test/python/service_bench.py
    python test/python/service_bench.py --direction read --service-mode staging
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import os
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nixl_cu13._bindings as bindings
import nixl_cu13._service_bindings as svc

# Number of ranks/GPUs in the benchmark.
DEFAULT_NUM_RANKS = 2
MAX_NUM_RANKS = 8

# Default per-peer slot size as a base-2 byte exponent.  30 means 1 GiB.
DEFAULT_BUFFER_BYTES_EXP = 30

# Element type of the send/recv buffers.  Must be 4 bytes wide for the
# arange-based fill pattern below (no overflow for buffers up to ~8 GiB).
BUFFER_DTYPE = torch.int32
BUFFER_ELEM_SIZE = torch.iinfo(BUFFER_DTYPE).bits // 8
BUFFER_ALIGNMENT = 8

# NIXL backend plugin used for all transfers.
NIXL_BACKEND = "UCX"

DEFAULT_ITERATIONS = 50
DEFAULT_WARMUPS = 10

DEFAULT_CONCURRENT_XFERS = 1

# Transfer direction. "write" (default): rank 0 pushes into every peer's
# buffer. "read": rank 0 pulls from every peer's buffer instead - the
# source/destination and (for RL modes) encoder/decoder roles all invert,
# see the module docstring.
DIRECTIONS = ("write", "read")
DEFAULT_DIRECTION = "write"


def _format_size(num_bytes: int) -> str:
    if num_bytes < (1 << 20):
        value = num_bytes / (1 << 10)
        suffix = "KB"
    elif num_bytes < (1 << 30):
        value = num_bytes / (1 << 20)
        suffix = "MB"
    elif num_bytes < (1 << 40):
        value = num_bytes / (1 << 30)
        suffix = "GB"
    else:
        value = num_bytes / (1 << 40)
        suffix = "TB"
    return f"{value:g} {suffix}"


@dataclass(frozen=True)
class ServiceModeDefinition:
    uses_service_mem: bool
    is_rl: bool # RL = Reinforcement Learning
    make_config: Callable[[], object]
    # (sender_ref_addr, receiver_ref_addr) -> opt_args; non-RL modes ignore it
    make_opt_args: Callable[[int, int], object]


@dataclass(frozen=True)
class ResolvedServiceMode:
    config: object
    service_mem_bytes: int
    max_concurrent_transfers: int


SERVICE_MODE_DEFINITIONS = {
    "direct": ServiceModeDefinition(
        uses_service_mem=False,
        is_rl=False,
        make_config=lambda: svc.nixlMarshalDirectConfig(),
        make_opt_args=lambda _sr, _rr: svc.nixlMarshalDirectOptArgs(),
    ),
    "staging": ServiceModeDefinition(
        uses_service_mem=True,
        is_rl=False,
        make_config=lambda: svc.nixlMarshalStagingConfig(),
        make_opt_args=lambda _sr, _rr: svc.nixlMarshalStagingOptArgs(),
    ),
    "compress": ServiceModeDefinition(
        uses_service_mem=True,
        is_rl=False,
        make_config=lambda: svc.nixlMarshalCompressConfig(
            algo=svc.nixl_marshal_compress_algo_t.ANS,
        ),
        make_opt_args=lambda _sr, _rr: svc.nixlMarshalCompressOptArgs(),
    ),
    "delta": ServiceModeDefinition(
        uses_service_mem=True,
        is_rl=True,
        make_config=lambda: svc.nixlMarshalDeltaConfig(),
        make_opt_args=lambda sr, rr: svc.nixlMarshalDeltaOptArgs(
            senderRef=sr,
            receiverRef=rr,
            senderMemType=bindings.nixl_mem_t.VRAM_SEG,
            receiverMemType=bindings.nixl_mem_t.VRAM_SEG,
            elementSize=BUFFER_ELEM_SIZE,
        ),
    ),
    "compress_ans_delta": ServiceModeDefinition(
        uses_service_mem=True,
        is_rl=True,
        make_config=lambda: svc.nixlMarshalCompressConfig(
            algo=svc.nixl_marshal_compress_algo_t.ANS_DELTA,
        ),
        make_opt_args=lambda sr, rr: svc.nixlMarshalCompressOptArgs(
            delta=svc.nixlMarshalDeltaOptArgs(
                senderRef=sr,
                receiverRef=rr,
                senderMemType=bindings.nixl_mem_t.VRAM_SEG,
                receiverMemType=bindings.nixl_mem_t.VRAM_SEG,
                elementSize=BUFFER_ELEM_SIZE,
            ),
        ),
    ),
}

# Available service modes for ``--service-mode``.  "direct" maps to the
# passthrough marshal mode (no staging/compression), every other entry
# triggers the staging-pool path through ``registerServiceMem``.
SERVICE_MODES = tuple(SERVICE_MODE_DEFINITIONS)


def _align_down(num_bytes: int, alignment: int = BUFFER_ALIGNMENT) -> int:
    return num_bytes - (num_bytes % alignment)


def _align_up(num_bytes: int, alignment: int = BUFFER_ALIGNMENT) -> int:
    return num_bytes + ((alignment - (num_bytes % alignment)) % alignment)


def _cuda_tensor_1d_aligned(
    num_elems: int,
    dtype: torch.dtype,
    *,
    alignment: int = BUFFER_ALIGNMENT,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Return a 1-D CUDA tensor whose storage starts at *alignment* bytes.

    Requires ``num_elems * dtype.element_size()`` (the buffer size in bytes) to
    be a multiple of *alignment* (default ``BUFFER_ALIGNMENT``). Raises
    ``AssertionError`` when ``nbytes % alignment != 0``. Callers that need a
    size that is not a multiple of *alignment* should pad via ``_align_up`` first.
    """
    assert alignment > 0 and (alignment & (alignment - 1)) == 0
    elem_size = torch.empty((), dtype=dtype).element_size()
    nbytes = num_elems * elem_size
    assert nbytes % alignment == 0, (
        f"{nbytes} byte buffer must be a multiple of {alignment}"
    )
    storage = torch.empty(nbytes + alignment - 1, dtype=torch.uint8, device=device)
    offset = (-storage.data_ptr()) % alignment
    return storage[offset:offset + nbytes].view(dtype)


def _assert_ptr_aligned(ptr: int, alignment: int, name: str) -> None:
    assert ptr % alignment == 0, (
        f"{name} pointer {ptr:#x} is not {alignment}-byte aligned"
    )


def _fill_pattern(rank: int, num_elems: int) -> torch.Tensor:
    """Per-rank deterministic, non-zero fill pattern for the send buffer.

    ``arange(N) + (rank + 1)`` so that:
      * no element is ever zero (the +rank+1 shift),
      * every rank produces a distinct sequence (validation can detect a
        write from the wrong peer or no write at all), and
      * the receiver can reconstruct the expected slot contents from
        ``(peer_rank, num_elems)`` alone — no extra metadata exchange.
    """
    return (
        (torch.arange(num_elems, dtype=BUFFER_DTYPE, device="cuda") + (rank + 1)) % 5
    )


PER_RL_STEP_WEIGHTS_DIFF_PERCENTAGE = 0.01
DELTA_MAX_MAGNITUDE = 256
DELTA_FILL_SEED = 1234
DELTA_CHANGE_SEED = 5678


def _fill_ref_pattern(num_elems: int) -> torch.Tensor:
    """Shared random fill for the delta reference buffer.

    Deterministic (fixed seed), so every rank - including a READ initiator
    that never talks to the peer about its data - can independently
    reconstruct the exact same baseline tensor without any communication.
    """
    gen = torch.Generator(device="cuda").manual_seed(DELTA_FILL_SEED)
    return torch.randint(
        torch.iinfo(BUFFER_DTYPE).min, torch.iinfo(BUFFER_DTYPE).max + 1,
        (num_elems,), generator=gen, dtype=BUFFER_DTYPE, device="cuda",
    )


def _precompute_delta_changes(
    num_elems: int, num_perturbations: int, device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the random changes per iteration for the RL delta.

    Deterministic (fixed seed), so whichever rank actually applies the
    perturbation each iteration (the WRITE sender, or - for READ - the
    peer/source) and whichever rank verifies the result afterward (the
    WRITE receiver, or the READ initiator) always agree on the sequence
    without exchanging it.
    """
    num_changes = max(1, int(num_elems * PER_RL_STEP_WEIGHTS_DIFF_PERCENTAGE))
    gen = torch.Generator(device=device).manual_seed(DELTA_CHANGE_SEED)
    change_idx = torch.randint(
        0, num_elems, (num_perturbations, num_changes),
        generator=gen, device=device,
    )
    change_vals = torch.randint(
        1, DELTA_MAX_MAGNITUDE, (num_perturbations, num_changes),
        generator=gen, dtype=BUFFER_DTYPE, device=device,
    )
    return change_idx, change_vals


def _load_safetensors_payload(safetensors_file: str, device: str) -> torch.Tensor:
    """Concatenate every tensor in *safetensors_file* into one 1-D
    ``BUFFER_DTYPE`` tensor on *device*, truncated to a ``BUFFER_ALIGNMENT``
    multiple so the byte length is safely divisible by the element size."""
    from safetensors import safe_open
    chunks: list[torch.Tensor] = []
    with safe_open(safetensors_file, framework="pt", device=device) as f:
        for key in f.keys():
            raw_bytes = f.get_tensor(key).contiguous().untyped_storage()
            chunks.append(
                torch.tensor([], dtype=torch.uint8, device=device)
                .set_(raw_bytes).clone()
            )
    raw = torch.cat(chunks)
    usable_bytes = _align_down(raw.numel(), BUFFER_ALIGNMENT)
    assert usable_bytes >= BUFFER_ALIGNMENT, (
        f"safetensors payload ({raw.numel()} bytes) is too small for an "
        f"{BUFFER_ALIGNMENT}-byte aligned buffer"
    )
    return raw[:usable_bytes].view(BUFFER_DTYPE)


def _make_buffers(
    rank: int,
    dev_id: int,
    buffer_bytes: int,
    safetensors_file: str | None,
    num_concurrent_xfers: int,
    is_rl: bool = False,
    random_src: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the transfer buffers as ``(send_buffer, recv_buffer, buffer_bytes)``.

    Every rank calls this unconditionally, regardless of ``--direction``:
    ``send_buffer`` is this rank's own data (used as the WRITE source when
    this rank is rank 0 under ``write``, as the READ source when this rank
    is a peer under ``read``, and otherwise only as a local baseline for
    content verification), and ``recv_buffer``/``rl_pair`` similarly always
    exist so the RL ping-pong and W0 baseline are available on every rank
    regardless of which role it ends up playing this run.

    ``recv_buffer``'s purpose depends on the mode:

    * non-RL -- a fresh zeroed landing zone for an incoming WRITE.
    * RL -- the delta *reference* baseline: the sender perturbs it and the
      receiver reconstructs into it in place, and ``send_buffer`` starts as a
      copy of it.

    The initial send contents come from one of three sources, in order:

    1. *safetensors_file* -- the file's concatenated tensor payload.  The
       returned ``buffer_bytes`` reflects the actual (aligned) payload size.
    2. *is_rl* or *random_src* without safetensors -- a shared deterministic
       random pattern (``_fill_ref_pattern``), identical on every rank.
       ``random_src`` lets non-RL modes use the same random data as RL.
    3. Default -- a per-rank synthetic pattern (``_fill_pattern``).

    ``num_concurrent_xfers`` slots are stored contiguously in each buffer.
    The returned ``buffer_bytes`` is the *per-slot* size.
    """
    device = f"cuda:{dev_id}"

    if safetensors_file:
        contents = _load_safetensors_payload(safetensors_file, device)
        buffer_bytes = contents.numel() * BUFFER_ELEM_SIZE
    else:
        assert buffer_bytes > 0
        assert buffer_bytes % BUFFER_ALIGNMENT == 0, (
            f"buffer_bytes ({buffer_bytes}) must be a multiple of "
            f"{BUFFER_ALIGNMENT}"
        )
        num_elems = buffer_bytes // BUFFER_ELEM_SIZE
        contents = (
            _fill_ref_pattern(num_elems) if (is_rl or random_src)
            else _fill_pattern(rank, num_elems)
        )

    slot_elems = contents.numel()
    total_elems = slot_elems * num_concurrent_xfers

    send_buffer = _cuda_tensor_1d_aligned(total_elems, BUFFER_DTYPE, device=device)
    # Identical initial content is replicated into every concurrent slot
    send_buffer.view(num_concurrent_xfers, slot_elems).copy_(contents)
    del contents
    recv_buffer = _cuda_tensor_1d_aligned(total_elems, BUFFER_DTYPE, device=device)
    if is_rl:
        # RL: recv_buffer is the delta reference; it starts identical to send_buffer.
        recv_buffer.copy_(send_buffer)
    else:
        recv_buffer.zero_()
    return send_buffer, recv_buffer, buffer_bytes


def _make_read_initiator_buffers(
    dev_id: int,
    buffer_bytes: int,
    num_concurrent_xfers: int,
    num_peers: int,
    is_rl: bool,
) -> torch.Tensor:
    """Per-peer destination buffer for a READ initiator: one contiguous
    ``num_concurrent_xfers * buffer_bytes`` region per peer, laid out
    contiguously in the same order as the initiator's ``send_to`` list
    (see ``benchmark``'s use of ``peer_idx``).

    For RL modes each region doubles as that peer's decode reference, so it
    must start at the exact baseline the peer's own encode reference starts
    at (``_fill_ref_pattern``). Both sides derive that baseline
    independently from the same fixed seed - no data exchange is needed for
    delta reconstruction to line up (mirrors how ``_make_buffers`` seeds
    ``recv_buffer``/``send_buffer`` identically on every rank for RL modes).
    For non-RL modes each region simply starts zeroed, so a successful READ
    is unambiguous.

    Returns a zero-length tensor when ``num_peers == 0`` (this rank plays no
    READ-initiator role this run).
    """
    device = f"cuda:{dev_id}"
    slot_elems = buffer_bytes // BUFFER_ELEM_SIZE
    total_elems = num_peers * num_concurrent_xfers * slot_elems
    dst_buffer = _cuda_tensor_1d_aligned(total_elems, BUFFER_DTYPE, device=device)
    if num_peers == 0:
        return dst_buffer
    if is_rl:
        dst_buffer.view(num_peers * num_concurrent_xfers, slot_elems).copy_(
            _fill_ref_pattern(slot_elems)
        )
    else:
        dst_buffer.zero_()
    return dst_buffer


def _resolve_service_mode(
    service_mode: str,
    max_concurrent_transfers: int,
) -> ResolvedServiceMode:
    """Resolve mode config and service-memory sizing for one benchmark run."""
    mode_definition = SERVICE_MODE_DEFINITIONS[service_mode]
    config = mode_definition.make_config()

    if not mode_definition.uses_service_mem:
        return ResolvedServiceMode(
            config=config,
            service_mem_bytes=0,
            max_concurrent_transfers=max_concurrent_transfers,
        )

    return ResolvedServiceMode(
        config=config,
        service_mem_bytes=svc.recommendServiceMemSize(
            config,
            maxConcurrentTransfers=max_concurrent_transfers,
        ),
        max_concurrent_transfers=max_concurrent_transfers,
    )


def _make_service_agent(name: str, mode_config: object):
    """Construct a service agent with the chosen marshal mode + UCX backend."""
    cfg = svc.nixlServiceAgentConfig()
    cfg.useProgThread = True
    cfg.useListenThread = False
    cfg.listenPort = 0
    cfg.syncMode = bindings.NIXL_THREAD_SYNC_NONE
    cfg.pthrDelay = 0
    cfg.lthrDelay = 100000
    cfg.captureTelemetry = False
    cfg.mode = mode_config

    agent = svc.nixlServiceAgent(name, cfg)
    backend_handle = agent.createBackend(NIXL_BACKEND, {})
    return agent, backend_handle


def _build_traffic_pattern(rank: int, num_ranks: int) -> tuple[list[int], list[int]]:
    """Return ``(active_peers, passive_from)`` for rank 0 vs. every peer.

    Direction-agnostic: rank 0 is always the one that actively creates,
    posts, and polls transfer handles against every entry in
    ``active_peers`` (the peers it *writes to* under ``--direction write``,
    or *reads from* under ``--direction read``); every other rank is
    passive and only drains notifications from ``passive_from`` (always
    ``[0]``, since rank 0 is always the one calling ``postXferReq`` and a
    notification always targets ``remote_agent`` - the peer - keyed by the
    caller's own name, regardless of direction).
    """
    if rank == 0:
        return [r for r in range(num_ranks) if r != 0], []
    return [], [0]


def benchmark(rank: int, num_ranks: int,
                    iterations: int, warmups: int,
                    buffer_bytes: int,
                    service_mode: str,
                    num_concurrent_xfers: int,
                    random_src: bool = False,
                    safetensors_file: str | None = None,
                    direction: str = DEFAULT_DIRECTION):
    """Run the configured WRITE or READ step ``warmups + iterations`` times.

    Setup (registration + MD exchange + xfer-request creation) happens once.
    Each iteration uses a distinct per-peer request handle and a fresh
    notification tag that includes the global iteration index, so warmup and
    measurement steps stay disjoint and notifications can't leak between
    them.  Only the last ``iterations`` steps are timed; the per-rank send
    buffer fill, the recv-buffer reset before timing starts, and the final
    content validation are all outside the timed window.

    Rank 0 is always the active side: it writes to (``direction="write"``,
    the default) or reads from (``direction="read"``) every other rank.
    Non-zero ranks are always passive: they hold the other end of the
    transfer and drain the completion notification, which always targets
    ``remote_agent`` - the peer - regardless of direction.  For READ, the
    source/destination and (for RL modes) encoder/decoder roles invert
    relative to WRITE: peers become the data source and, for RL modes, run
    the per-iteration ping-pong perturbation themselves instead of rank 0.

    The ``service_mode`` selects the marshal mode of the underlying service
    agent.  For ``"direct"`` no service memory is registered.  For every
    other mode a runtime-sized VRAM slab is allocated and handed to
    ``registerServiceMem`` so the marshal backend has a staging pool.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert num_ranks == dist.get_world_size()
    assert rank == dist.get_rank()
    assert iterations > 0
    assert warmups >= 0
    assert service_mode in SERVICE_MODES, service_mode
    assert direction in DIRECTIONS, direction

    dev_id = torch.cuda.current_device()
    mode = SERVICE_MODE_DEFINITIONS[service_mode]

    # Every rank builds both buffers unconditionally, regardless of
    # direction/role - see _make_buffers' docstring for why the "unused"
    # ones still matter (RL baselines, and READ-initiator verification
    # against its own send_buffer for the shared-data case).
    send_buffer, recv_buffer, buffer_bytes = _make_buffers(
        rank, dev_id, buffer_bytes, safetensors_file, num_concurrent_xfers,
        mode.is_rl, random_src)
    is_rl = mode.is_rl
    # RL uses the data source's two buffers as a ping-pong pair: each
    # iteration the src buffer becomes this iteration's ref and vice versa.
    rl_pair = [recv_buffer, send_buffer] if is_rl else None
    fan_out = max(num_ranks - 1, 1) if rank == 0 else 1
    resolved_mode = _resolve_service_mode(
        service_mode,
        max_concurrent_transfers=fan_out * num_concurrent_xfers,
    )
    assert (
        not mode.uses_service_mem
        or resolved_mode.service_mem_bytes > 0
    )

    send_to, recv_from = _build_traffic_pattern(rank, num_ranks)
    # Whichever side plays the data-source (encoder) role for this
    # direction: rank 0 for WRITE, the peer for READ. RL perturbation
    # always runs on this side, regardless of which physical rank it is -
    # see evolve_own_rl_buffers below.
    is_data_source = (
        (direction == "write" and bool(send_to))
        or (direction == "read" and bool(recv_from))
    )

    # TODO-Roee: support compression backend.
    agent, backend = _make_service_agent(str(rank), resolved_mode.config)

    slot_elems = buffer_bytes // BUFFER_ELEM_SIZE

    # RL: precompute the perturbations per iteration on whichever side is
    # the data source this run (see is_data_source above).
    change_idx = change_vals = None
    rl_w0 = None
    if is_rl:
        rl_w0 = recv_buffer.clone()
        if is_data_source:
            change_idx, change_vals = _precompute_delta_changes(
                slot_elems, max(warmups, iterations), device=send_buffer.device,
            )

    _assert_ptr_aligned(send_buffer.data_ptr(), BUFFER_ALIGNMENT, "send_buffer")
    _assert_ptr_aligned(recv_buffer.data_ptr(), BUFFER_ALIGNMENT, "recv_buffer")
    assert buffer_bytes % BUFFER_ALIGNMENT == 0

    total_buffer_bytes = num_concurrent_xfers * buffer_bytes
    send_reg = bindings.nixlRegDList(
        bindings.VRAM_SEG,
        [(send_buffer.data_ptr(), total_buffer_bytes, dev_id, "")],
    )
    recv_reg = bindings.nixlRegDList(
        bindings.VRAM_SEG,
        [(recv_buffer.data_ptr(), total_buffer_bytes, dev_id, "")],
    )

    # Service-memory pool — only allocated/registered for non-DIRECT modes.
    # DIRECT is a passthrough that has no use for staging memory; allocating
    # it anyway would just waste VRAM.  For all other modes the marshal
    # backend stages payloads through this region, so we hand it a runtime-sized
    # slab of VRAM and register it via the dedicated
    # service-mem path (NOT registerMem — that's the user data path).
    svc_buffer = None
    svc_reg = None

    # READ-initiator destination buffer — only allocated on the active rank
    # (rank 0) when direction=="read": one region per peer in send_to, which
    # doubles as that peer's decode reference for RL modes.  None (unused)
    # for direction=="write", or on any passive (non-zero) rank.
    read_dst_buffer = None
    read_dst_reg = None

    # rank -> list of active request handles (one per slot) for the current iteration.
    peer_handles: dict[int, list[int]] = {}
    try:
        agent.registerMem(send_reg, [backend])
        agent.registerMem(recv_reg, [backend])

        if mode.uses_service_mem:
            # C++ sizing (recommendServiceMemSize) should already return a
            # BUFFER_ALIGNMENT-rounded size.  Call _align_up defensively so we
            # never under-allocate if the C++ side ever returns a non-aligned
            # value; this is a no-op in normal operation.  _assert_ptr_aligned
            # on svc_buffer validates the allocator's pointer alignment.
            service_mem_bytes = _align_up(
                resolved_mode.service_mem_bytes, BUFFER_ALIGNMENT,
            )
            if service_mem_bytes != resolved_mode.service_mem_bytes:
                warnings.warn(
                    f"C++ service_mem_bytes ({resolved_mode.service_mem_bytes}) is not "
                    f"{BUFFER_ALIGNMENT}-byte aligned; padded to {service_mem_bytes}",
                    stacklevel=2,
                )
            svc_buffer = _cuda_tensor_1d_aligned(
                service_mem_bytes, torch.uint8,
            )
            _assert_ptr_aligned(
                svc_buffer.data_ptr(), BUFFER_ALIGNMENT, "svc_buffer",
            )
            svc_reg = bindings.nixlRegDList(
                bindings.VRAM_SEG,
                [(svc_buffer.data_ptr(), service_mem_bytes, dev_id, "")],
            )
            agent.registerServiceMem(svc_reg, [backend])

        if direction == "read" and send_to:
            read_dst_buffer = _make_read_initiator_buffers(
                dev_id, buffer_bytes, num_concurrent_xfers, len(send_to), is_rl,
            )
            _assert_ptr_aligned(
                read_dst_buffer.data_ptr(), BUFFER_ALIGNMENT, "read_dst_buffer",
            )
            read_dst_reg = bindings.nixlRegDList(
                bindings.VRAM_SEG,
                [(
                    read_dst_buffer.data_ptr(),
                    len(send_to) * num_concurrent_xfers * buffer_bytes,
                    dev_id,
                    "",
                )],
            )
            agent.registerMem(read_dst_reg, [backend])

        # Bootstrap: publish this rank's MD blob, device id, and every
        # address a peer might need depending on direction/mode - the WRITE
        # destination address, the READ source address, and (for RL modes)
        # the two RL ping-pong base addresses, indexed the same way
        # evolve_own_rl_buffers indexes rl_pair (index `ping_pong_idx % 2` is
        # the "ref" side).  The MD installs rkeys/auth in every peer's local
        # backend; agent name == rank id, so peers address each other by
        # rank without any extra mapping.
        local_info = {
            "md": bytes(agent.getLocalMD()),
            "dev_id": dev_id,
            "write_dst_addr": recv_buffer.data_ptr(),
            "read_src_addr": send_buffer.data_ptr(),
            "read_rl_ref_addrs": (
                (rl_pair[0].data_ptr(), rl_pair[1].data_ptr()) if is_rl else None
            ),
        }

        dist.barrier()
        gathered: list = [None] * num_ranks
        dist.all_gather_object(gathered, local_info)

        for r in range(num_ranks):
            if r == rank:
                continue
            peer_name = agent.loadRemoteMD(gathered[r]["md"]).decode()
            assert peer_name == str(r), (peer_name, r)

        def evolve_own_rl_buffers(ping_pong_idx: int) -> tuple[int, int]:
            # Return (ref_base, src_base) base pointers for this iteration,
            # ping-ponging THIS rank's own rl_pair and applying this step's
            # perturbation - but only when this rank is the data source for
            # the configured direction (is_data_source); otherwise a no-op
            # returning (0, send_buffer.data_ptr()), since this rank's own
            # buffers are then either used read-only (WRITE destination) or
            # not used for the transfer at all (READ initiator, which has
            # read_dst_buffer instead). Callers always pass the local,
            # reset-relative index (w during warmup, i during timed - never
            # the globally-unique it), so create_peer_handles's ping_pong_idx
            # parameter can index a peer's published addresses identically.
            if not (is_rl and is_data_source):
                return 0, send_buffer.data_ptr()
            ref_buf = rl_pair[ping_pong_idx % 2]
            src_buf = rl_pair[(ping_pong_idx + 1) % 2]
            # each buffer is the previous iteration's result switched src <-> ref
            # so starting the second iteration, we should apply both the current perturbation
            # in addition to last one's catching up.
            src_view = src_buf.view(num_concurrent_xfers, slot_elems)
            if ping_pong_idx > 0:
                src_view.index_add_(
                    1, change_idx[ping_pong_idx - 1],
                    change_vals[ping_pong_idx - 1].repeat(num_concurrent_xfers, 1),
                )
            src_view.index_add_(
                1, change_idx[ping_pong_idx],
                change_vals[ping_pong_idx].repeat(num_concurrent_xfers, 1),
            )
            # Sync so the perturbation lands before the marshal reads it.
            torch.cuda.synchronize()
            return ref_buf.data_ptr(), src_buf.data_ptr()

        def create_peer_handles(
            it: int, ping_pong_idx: int, sender_ptrs: tuple[int, int],
        ) -> None:
            """Build this iteration's handles for every peer in ``send_to``.

            ``it`` is the globally-unique index used only for notification
            tags (distinct across warmup and timed steps, so a stray
            straggler can't be mistaken for the wrong iteration).
            ``ping_pong_idx`` is the *local* (per-phase, reset-relative)
            index used for RL ping-pong role selection - it must match
            whatever index the data source's own ``evolve_own_rl_buffers``
            call used this step, which is ``it`` during warmup but the
            local ``i`` (not ``it = warmups + i``) during the timed loop,
            since the reset-before-timing block restarts the ping-pong from
            a common baseline. For WRITE this arrives pre-applied inside
            ``sender_ptrs`` (rank 0 computed it locally, same process); for
            READ this rank (the initiator) never perturbs anything itself,
            so it must index the peer's published addresses with the exact
            same ``ping_pong_idx`` the peer used, or it will race a stale or
            not-yet-perturbed buffer half the time.
            """
            if direction == "write":
                ref_base, src_base = sender_ptrs
                for r in send_to:
                    peer_dst_addr = gathered[r]["write_dst_addr"]
                    peer_dev_id = gathered[r]["dev_id"]
                    concurrent_handles: list[int] = []
                    for j in range(num_concurrent_xfers):
                        slot_off = j * buffer_bytes
                        sender_ref_ptr = ref_base + slot_off if ref_base else 0
                        sender_src_ptr = src_base + slot_off
                        dst_slot_addr = peer_dst_addr + slot_off
                        src = bindings.nixlXferDList(
                            bindings.VRAM_SEG,
                            [(sender_src_ptr, buffer_bytes, dev_id)],
                        )
                        dst = bindings.nixlXferDList(
                            bindings.VRAM_SEG,
                            [(dst_slot_addr, buffer_bytes, peer_dev_id)],
                        )
                        opt_args = mode.make_opt_args(sender_ref_ptr, dst_slot_addr)
                        reqh = agent.createXferReq(
                            bindings.NIXL_WRITE,
                            src,
                            dst,
                            str(r),
                            f"from-{rank}-init-{it}-{j}",
                            [backend],
                            opt_args,
                        )
                        assert reqh != 0
                        concurrent_handles.append(reqh)
                    peer_handles[r] = concurrent_handles
            else:  # direction == "read"
                # sender_ptrs is unused here: rank 0 (the READ initiator) is
                # never the data source, so it carries no ping-pong state of
                # its own - each peer's CURRENT (src, ref) addresses are
                # derived below from its published read_rl_ref_addrs base
                # pointers plus ping_pong_idx % 2, exactly mirroring how that
                # peer's own evolve_own_rl_buffers indexes its local rl_pair.
                for peer_idx, r in enumerate(send_to):
                    peer_dev_id = gathered[r]["dev_id"]
                    if is_rl:
                        peer_ref_addr = gathered[r]["read_rl_ref_addrs"][ping_pong_idx % 2]
                        peer_src_addr = gathered[r]["read_rl_ref_addrs"][(ping_pong_idx + 1) % 2]
                    else:
                        peer_ref_addr = 0
                        peer_src_addr = gathered[r]["read_src_addr"]
                    peer_region_base = peer_idx * num_concurrent_xfers * buffer_bytes
                    concurrent_handles: list[int] = []
                    for j in range(num_concurrent_xfers):
                        slot_off = j * buffer_bytes
                        my_dst_addr = read_dst_buffer.data_ptr() + peer_region_base + slot_off
                        peer_src_slot = peer_src_addr + slot_off
                        peer_ref_slot = peer_ref_addr + slot_off if peer_ref_addr else 0
                        local_descs = bindings.nixlXferDList(
                            bindings.VRAM_SEG,
                            [(my_dst_addr, buffer_bytes, dev_id)],
                        )
                        remote_descs = bindings.nixlXferDList(
                            bindings.VRAM_SEG,
                            [(peer_src_slot, buffer_bytes, peer_dev_id)],
                        )
                        # senderRef = the peer's own (encoder's) reference;
                        # receiverRef = this rank's own (decoder's)
                        # reference - swapped relative to WRITE, where rank
                        # 0 is the encoder.
                        opt_args = mode.make_opt_args(peer_ref_slot, my_dst_addr)
                        reqh = agent.createXferReq(
                            bindings.NIXL_READ,
                            local_descs,
                            remote_descs,
                            str(r),
                            f"from-{rank}-init-{it}-{j}",
                            [backend],
                            opt_args,
                        )
                        assert reqh != 0
                        concurrent_handles.append(reqh)
                    peer_handles[r] = concurrent_handles

        def release_peer_handles() -> None:
            for concurrent_handles in peer_handles.values():
                for reqh in concurrent_handles:
                    agent.releaseXferReq(reqh)
            peer_handles.clear()

        notifs: dict[str, list[bytes]] = {}

        def run_iteration(it: int) -> None:
            # ``notifs`` accumulates across iterations because a peer can be
            # one step ahead and deliver iteration-(it+1) tags while we're
            # still draining iteration-(it).  pybind11 STL casters deep-copy
            # the dict on the C++ boundary, so we must rebind from the
            # return value (the parameter is effectively in-only).
            nonlocal notifs

            # Active side (no-op when ``send_to`` is empty): creates+posts
            # the transfer regardless of direction - a READ initiator calls
            # postXferReq/getXferStatus exactly as a WRITE sender does, only
            # the descriptor roles built by create_peer_handles differ.
            for peer_r in send_to:
                for j, reqh in enumerate(peer_handles[peer_r]):
                    status = agent.postXferReq(reqh, f"from-{rank}-{it}-{j}")
                    assert status in (bindings.NIXL_SUCCESS, bindings.NIXL_IN_PROG), (
                        peer_r, j, status,
                    )
            outstanding_sends = {
                reqh for peer_r in send_to for reqh in peer_handles[peer_r]
            }
            while outstanding_sends:
                for reqh in list(outstanding_sends):
                    if agent.getXferStatus(reqh) != bindings.NIXL_IN_PROG:
                        outstanding_sends.remove(reqh)

            # Passive side (no-op when ``recv_from`` is empty).  Expect one
            # notification per (active rank, slot) tag - regardless of
            # direction, the notification always targets remote_agent (this
            # rank) keyed by the caller's own name (always "0" in this
            # star topology, since rank 0 is always the one calling
            # postXferReq above).
            outstanding = {
                str(r): {
                    f"from-{r}-{it}-{j}".encode()
                    for j in range(num_concurrent_xfers)
                }
                for r in recv_from
            }
            while outstanding:
                notifs = agent.getNotifs(notifs, [backend])
                for name in list(outstanding):
                    bucket = notifs.get(name)
                    if not bucket:
                        continue
                    wanted = outstanding[name]
                    for tag in [t for t in wanted if t in bucket]:
                        bucket.remove(tag)
                        wanted.discard(tag)
                    if not bucket:
                        del notifs[name]
                    if not wanted:
                        del outstanding[name]

        # RL+READ needs the peer to perturb (and CUDA-sync) before rank 0
        # may safely read this iteration's data, and rank 0 must finish
        # reading before the peer perturbs again for the next iteration -
        # both barriers below sit outside rank 0's own timed window.
        # Non-RL READ and both RL/non-RL WRITE need neither: the WRITE
        # sender perturbs and builds handles inline in the same process, and
        # non-RL READ has no perturbation step at all.
        needs_rl_read_barriers = is_rl and direction == "read"

        for w in range(warmups):
            if needs_rl_read_barriers:
                evolve_own_rl_buffers(w)  # no-op on rank 0; perturbs on peers
                dist.barrier()  # B1: every peer has perturbed for `w`
                sender_ptrs = (0, send_buffer.data_ptr())  # unused by the READ branch
            else:
                sender_ptrs = evolve_own_rl_buffers(w)
            try:
                create_peer_handles(w, w, sender_ptrs)
                run_iteration(w)
            finally:
                release_peer_handles()
            if needs_rl_read_barriers:
                dist.barrier()  # B2: rank 0 finished reading `w`

        torch.cuda.synchronize()
        dist.barrier()

        # Reset outside the timed window so validation proves the *timed*
        # iterations moved data, not the warmups.  cuda.synchronize() below
        # makes sure the reset kernels finished before peers write here.
        # Non-RL zeroes its recv buffer; RL reconstructs into recv_buffer in
        # place, so instead we restore both ping-pong buffers to W0.  The
        # READ initiator's own read_dst_buffer (unused above for WRITE) is
        # reset the same way, mirroring _make_read_initiator_buffers' own
        # initial fill.
        if not is_rl:
            recv_buffer.zero_()
        else:
            for buf in rl_pair:
                buf.copy_(rl_w0)
        if read_dst_buffer is not None:
            if is_rl:
                read_dst_buffer.view(-1, slot_elems).copy_(_fill_ref_pattern(slot_elems))
            else:
                read_dst_buffer.zero_()
        torch.cuda.synchronize()

        dist.barrier()
        elapsed_s = 0.0
        for i in range(iterations):
            it = warmups + i
            # Perturbation (either branch) uses the local ground-zero i, not
            # it: the reset-before-timing block above restored rl_pair (and,
            # for a READ initiator, read_dst_buffer) to the shared W0
            # baseline, so the timed loop's perturbation sequence - and the
            # verification below, which mirrors it via [:iterations] - both
            # restart from index 0 rather than continuing from warmups.
            if needs_rl_read_barriers:
                evolve_own_rl_buffers(i)  # no-op on rank 0; perturbs on peers
                dist.barrier()  # B1: every peer has perturbed for `i`
                sender_ptrs = (0, send_buffer.data_ptr())  # unused by the READ branch
            else:
                sender_ptrs = evolve_own_rl_buffers(i)
            t0 = time.perf_counter()
            try:
                create_peer_handles(it, i, sender_ptrs)
                run_iteration(it)
                elapsed_s += time.perf_counter() - t0
            finally:
                release_peer_handles()
            if needs_rl_read_barriers:
                dist.barrier()  # B2: rank 0 finished reading `it`
        dist.barrier()

        per_iter_s = elapsed_s / iterations
        # Active-rank iterations move len(send_to) peers' worth of data;
        # passive ranks move len(recv_from) (always 1 in this star
        # topology) peer's worth - regardless of direction, this is "how
        # many peer-slots did I move this iteration".
        bytes_per_iter = (
            len(send_to) if send_to else len(recv_from)
        ) * num_concurrent_xfers * buffer_bytes
        bw_gbps = bytes_per_iter / per_iter_s / 1e9 if per_iter_s > 0 else float("inf")

        if direction == "write":
            # Verify final contents only on receiver ranks
            for r in recv_from:
                if is_rl:
                    # W0 plus every timed perturbation should result in the final received buffer
                    sender_idx, sender_vals = _precompute_delta_changes(
                        slot_elems, max(warmups, iterations), device=recv_buffer.device,
                    )
                    expected_slot = (
                        rl_w0.view(num_concurrent_xfers, slot_elems)[0].clone()
                    )
                    expected_slot.index_add_(
                        0,
                        sender_idx[:iterations].reshape(-1),
                        sender_vals[:iterations].reshape(-1),
                    )
                    expected = expected_slot.repeat(num_concurrent_xfers)
                    assert torch.equal(recv_buffer, expected), (
                        f"rank {rank}: mismatch from sender {r}"
                    )
                else:
                    # safetensors/random_src: all ranks load the same data, so local send_buffer matches sender's.
                    # synthetic: _fill_pattern() reconstructs sender's pattern deterministically.
                    expected = (
                        send_buffer if (safetensors_file or random_src)
                        else _fill_pattern(r, slot_elems).repeat(num_concurrent_xfers)
                    )
                    assert torch.equal(recv_buffer, expected), (
                        f"rank {rank}: content mismatch from sender {r}"
                    )
        else:  # direction == "read"
            # Verify final contents only on the initiator (rank 0), one
            # per-peer region of read_dst_buffer at a time.
            for peer_idx, r in enumerate(send_to):
                region_start = peer_idx * num_concurrent_xfers * slot_elems
                region_elems = num_concurrent_xfers * slot_elems
                region = read_dst_buffer[region_start:region_start + region_elems]
                if is_rl:
                    # W0 plus every timed perturbation should result in the final read region.
                    source_idx, source_vals = _precompute_delta_changes(
                        slot_elems, max(warmups, iterations), device=region.device,
                    )
                    expected_slot = _fill_ref_pattern(slot_elems)
                    expected_slot.index_add_(
                        0,
                        source_idx[:iterations].reshape(-1),
                        source_vals[:iterations].reshape(-1),
                    )
                    expected = expected_slot.repeat(num_concurrent_xfers)
                    assert torch.equal(region, expected), (
                        f"rank {rank}: READ mismatch from source {r}"
                    )
                else:
                    # safetensors/random_src: all ranks load the same data, so local send_buffer matches source's.
                    # synthetic: _fill_pattern() reconstructs the source's pattern deterministically.
                    expected = (
                        send_buffer if (safetensors_file or random_src)
                        else _fill_pattern(r, slot_elems).repeat(num_concurrent_xfers)
                    )
                    assert torch.equal(region, expected), (
                        f"rank {rank}: READ content mismatch from source {r}"
                    )

        return (
            elapsed_s,
            per_iter_s,
            bw_gbps,
            buffer_bytes,
            resolved_mode.service_mem_bytes,
            resolved_mode.max_concurrent_transfers,
        )
    except Exception as exc:
        print(f"[service_bench][rank {rank}] benchmark failed: {exc}")
        raise


def _worker(rank: int, num_ranks: int,
            iterations: int, warmups: int,
            buffer_bytes: int,
            service_mode: str,
            safetensors_file: str | None,
            random_src: bool,
            num_concurrent_xfers: int,
            direction: str,
            master_addr: str, master_port: int,
            dist_backend: str) -> None:
    """Per-rank entry point launched by ``torch.multiprocessing.spawn``.

    Pins this process to GPU ``rank``, brings up the ``torch.distributed``
    process group used purely for the metadata bootstrap, then runs the
    benchmark.  The actual data path is NIXL — torch.distributed never
    touches the buffers.
    """
    assert torch.cuda.device_count() >= num_ranks, (
        f"need >= {num_ranks} CUDA devices on this node, "
        f"got {torch.cuda.device_count()}"
    )
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(num_ranks)

    dist.init_process_group(
        backend=dist_backend,
        rank=rank,
        world_size=num_ranks,
    )

    try:
        (
            elapsed_s,
            per_iter_s,
            bw_gbps,
            buffer_bytes,
            service_mem_bytes,
            max_concurrent_transfers,
        ) = benchmark(
            rank, num_ranks, iterations, warmups, buffer_bytes,
            service_mode, num_concurrent_xfers, random_src,
            safetensors_file, direction,
        )
        send_to, recv_from = _build_traffic_pattern(rank, num_ranks)
        if direction == "write":
            role = "send+recv" if send_to and recv_from else (
                "send" if send_to else "recv"
            )
        else:
            role = "read-init+served" if send_to and recv_from else (
                "read-init" if send_to else "served"
            )
        result_line = (
            f"\n============ Rank {rank} Results =============\n"
            f"svc={service_mode}\n"
            f"direction={direction}\n"
            f"role={role}\n"
            f"warmups={warmups}\n"
            f"iters={iterations}\n"
            f"total={elapsed_s * 1e3:.3f}ms\n"
            f"per_iter={per_iter_s * 1e6:.1f}us\n"
            f"bw={bw_gbps:.3f} GB/s\n"
            f"=========================================\n"
        )
        rank_results = [None] * num_ranks
        dist.all_gather_object(
            rank_results,
            {
                "rank": rank,
                "per_iter_s": per_iter_s,
                "result_line": result_line,
            },
        )
        if rank == 0:
            mean_per_iter_s = (
                sum(result["per_iter_s"] for result in rank_results) / num_ranks
            )
            outlier_results = [
                result for result in rank_results
                if abs(result["per_iter_s"] - mean_per_iter_s) / mean_per_iter_s > 0.05
            ]
            if outlier_results:
                outlier_ranks = [result["rank"] for result in outlier_results]
                print(
                    "WARNING: per-rank per_iter differs by more than 5% "
                    f"from the mean on ranks {outlier_ranks}; printing all ranks.",
                    flush=True,
                )
                for result in rank_results:
                    print(result["result_line"], flush=True)
            else:
                print(rank_results[0]["result_line"], flush=True)
            mode = SERVICE_MODE_DEFINITIONS[service_mode]
            summary = "\n========== Test Configuration ===========\n"
            summary += f"direction={direction}\n"
            summary += f"buffer_size={_format_size(buffer_bytes)}\n"
            summary += f"concurrent_xfers={num_concurrent_xfers}\n"
            if mode.uses_service_mem:
                summary += (
                    f"max_concurrent_transfers={max_concurrent_transfers}\n"
                    f"service_mem={_format_size(service_mem_bytes)}"
                )
            if mode.is_rl:
                summary += f"\nPER_RL_STEP_WEIGHTS_DIFF_PERCENTAGE={PER_RL_STEP_WEIGHTS_DIFF_PERCENTAGE}"
            summary += f"\n=========================================\n"
            print(summary, flush=True)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-ranks", type=int, default=DEFAULT_NUM_RANKS,
                        help="number of ranks/GPUs to use on this node "
                             f"(default: {DEFAULT_NUM_RANKS}, max: {MAX_NUM_RANKS})")
    parser.add_argument("--iterations", "-N", type=int, default=DEFAULT_ITERATIONS,
                        help=f"number of timed iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--warmups", "-K", type=int, default=DEFAULT_WARMUPS,
                        help="number of warmup iterations, not timed "
                             f"(default: {DEFAULT_WARMUPS})")
    parser.add_argument("--buffer-bytes-exp", type=int,
                        default=DEFAULT_BUFFER_BYTES_EXP,
                        help="base-2 exponent for the per-peer slot size in "
                             "bytes; this is also the per-rank send buffer "
                             "size. For example, 10 means 1 KiB and 20 means "
                             f"1 MiB (default: {DEFAULT_BUFFER_BYTES_EXP}, "
                             "1 GiB)")
    parser.add_argument("--concurrent-xfers", type=int,
                        default=DEFAULT_CONCURRENT_XFERS,
                        help="number of identical transfers the active rank "
                             "posts concurrently to/from each peer per "
                             "iteration "
                             f"(default: {DEFAULT_CONCURRENT_XFERS})")
    parser.add_argument("--direction", default=DEFAULT_DIRECTION,
                        choices=DIRECTIONS,
                        help="transfer direction. 'write' (default) has "
                             "rank 0 push into every peer's buffer; 'read' "
                             "has rank 0 pull from every peer's buffer "
                             "instead - peers become the data source and, "
                             "for RL modes, run the per-iteration "
                             "perturbation themselves "
                             f"(default: {DEFAULT_DIRECTION})")
    parser.add_argument("--service-mode", default="direct",
                        choices=SERVICE_MODES,
                        help="marshal mode for the nixlServiceAgent. "
                             "'direct' (default) is a passthrough and "
                             "allocates no service memory; any other mode "
                             "sizes its service-memory allocation from "
                             "recommendServiceMemSize")
    parser.add_argument("--safetensors-file", type=str, default=None,
                        help="path to a .safetensors file whose tensor data "
                             "will be used as the send buffer instead of "
                             "synthetic fill patterns. When given, "
                             "--buffer-bytes-exp is ignored and the transfer "
                             "size equals the file's total tensor payload")
    parser.add_argument("--random-src", action="store_true",
                        help="fill the non-RL send buffer with the shared "
                             "seeded random pattern (same as RL modes) instead "
                             "of the synthetic simple pattern. Use this to "
                             "compare 'compress' vs 'compress_ans_delta' on the "
                             "same data. No effect for RL modes or when "
                             "--safetensors-file is given.")
    parser.add_argument("--dist-backend", default="gloo",
                        choices=("gloo", "nccl"),
                        help="torch.distributed backend for the bootstrap "
                             "collective (data path is NIXL, not torch)")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_ranks < 1 or args.num_ranks > MAX_NUM_RANKS:
        raise ValueError(
            f"--num-ranks must be between 1 and {MAX_NUM_RANKS}, "
            f"got {args.num_ranks}"
        )
    if args.concurrent_xfers < 1:
        raise ValueError(
            f"--concurrent-xfers must be >= 1, got {args.concurrent_xfers}"
        )
    buffer_bytes = 1 << args.buffer_bytes_exp

    mp.spawn(
        _worker,
        args=(args.num_ranks, args.iterations, args.warmups, buffer_bytes,
              args.service_mode, args.safetensors_file,
              args.random_src, args.concurrent_xfers, args.direction,
              args.master_addr, args.master_port, args.dist_backend),
        nprocs=args.num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
