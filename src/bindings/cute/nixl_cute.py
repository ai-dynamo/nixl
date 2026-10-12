# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native NIXL device operations for CuTe kernels; host setup stays in NIXL.

Submission never adds a completion loop, fence, allocation, or kernel launch.
Handles are from prep_mem_view(); request addresses refer to caller-owned GPU
storage. Rebuild device.bc against the same NIXL/UCX used by the host process.
"""

from pathlib import Path

import cutlass
import cutlass.cute as cute

THREAD, WARP, BLOCK = 0, 1, 2
SUCCESS, IN_PROGRESS, DEFER = 0, 1, 1
REQUEST_BYTES, REQUEST_ALIGNMENT = 64, 16
_bitcode = cute.BitCode(str(Path(__file__).with_name("device.bc")))


@cute.extern(name="cute_nixl_put", source=_bitcode)
def _put(
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    size: cutlass.Uint64,
    local_index: cutlass.Uint64,
    remote_index: cutlass.Uint64,
    local_offset: cutlass.Uint64,
    remote_offset: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
    request: cutlass.Uint64,
    level: cutlass.Int32,
) -> cutlass.Int32:
    pass


@cute.extern(name="cute_nixl_atomic_add", source=_bitcode)
def _atomic_add(
    value: cutlass.Uint64,
    remote: cutlass.Uint64,
    index: cutlass.Uint64,
    offset: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
    request: cutlass.Uint64,
    level: cutlass.Int32,
) -> cutlass.Int32:
    pass


@cute.extern(name="cute_nixl_get_xfer_status", source=_bitcode)
def _get_xfer_status(request: cutlass.Uint64, level: cutlass.Int32) -> cutlass.Int32:
    pass


def _status(value):
    # CuTe 4.5.1 may cache extern results as signless i32; preserve native errors.
    return cutlass.Uint32(value).bitcast(cutlass.Int32)


def put(
    local,
    remote,
    size,
    request=0,
    *,
    local_index=0,
    remote_index=0,
    local_offset=0,
    remote_offset=0,
    channel=0,
    flags=0,
    level=THREAD,
):
    """Submit nixlPut and return its status unchanged.

    Args:
        local: Local memory-view handle.
        remote: Remote memory-view handle.
        size: Bytes to transfer.
        request: Private request address per participating thread, or zero to
            pass a native null request (not a blocking-mode selector).

    Other arguments directly select native descriptors, byte offsets, channel,
    flags and cooperation level. Keep the buffers/views/request alive until the
    operation completes. DEFER permits, but does not guarantee, deferral.
    """
    return _status(
        _put(
            cutlass.Uint64(local),
            cutlass.Uint64(remote),
            cutlass.Uint64(size),
            cutlass.Uint64(local_index),
            cutlass.Uint64(remote_index),
            cutlass.Uint64(local_offset),
            cutlass.Uint64(remote_offset),
            cutlass.Uint32(channel),
            cutlass.Uint64(flags),
            cutlass.Uint64(request),
            cutlass.Int32(level),
        )
    )


def atomic_add(
    value, remote, request=0, *, index=0, offset=0, channel=0, flags=0, level=THREAD
):
    """Submit native nixlAtomicAdd to a remote uint64 counter.

    Offset is in bytes. Request ownership and cooperation follow put(). There
    is no wrapper fence or wait; use the native ordering contract, including
    publication of writes made by cooperating producer threads.
    """
    return _status(
        _atomic_add(
            cutlass.Uint64(value),
            cutlass.Uint64(remote),
            cutlass.Uint64(index),
            cutlass.Uint64(offset),
            cutlass.Uint32(channel),
            cutlass.Uint64(flags),
            cutlass.Uint64(request),
            cutlass.Int32(level),
        )
    )


def get_xfer_status(request, *, level=THREAD):
    """Perform one native progress poll on a nonzero, in-progress request.

    Call only after IN_PROGRESS, using the submission's cooperation level.
    Do not poll after success/error, or after an untracked (null-request) call.
    All participating threads must enter cooperative calls with uniform
    arguments, except their private request addresses.
    """
    return _status(_get_xfer_status(cutlass.Uint64(request), cutlass.Int32(level)))
