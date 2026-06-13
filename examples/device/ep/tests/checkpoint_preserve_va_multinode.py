# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-node NIXL EP graph-stable checkpoint/restart reproducer.

This script is intentionally standalone so it can run inside Kubernetes pods or
other containers that already contain the built ``nixl_ep`` extension. Launch
one process per EP rank, for example:

    torchrun --nnodes 2 --nproc-per-node 1 \
        --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} \
        --master-port 29500 \
        examples/device/ep/tests/checkpoint_preserve_va_multinode.py \
        --store-master-addr ${MASTER_ADDR}

The script validates the direct NIXL EP Buffer path used by vLLM's
NixlEPAll2AllManager: VMM-backed graph-visible buffers are allocated, a
representative low-latency dispatch/combine is optionally captured in a CUDA
graph, graph-visible virtual addresses are recorded, checkpoint pause releases
non-checkpointable NIXL state while preserving those VAs, and resume rebuilds
fresh NIXL metadata before replaying the captured graph without recapture.
CUDA graph capture uses NIXL EP receive-hook mode so send and receive kernels
are launched on the capture stream.

When ``--snapshot-control-dir`` or ``DYN_SNAPSHOT_CONTROL_DIR`` is set, the
script follows the Dynamo snapshot-control contract: after pause it writes
``ready-for-checkpoint``, the original process exits cleanly on
``snapshot-complete``, and the restored process recreates its external
TCPStore connection, resumes NIXL EP, validates graph-visible addresses, and
replays the pre-captured CUDA graph after ``restore-complete``. Use
``--force-ucx-tcp`` or ``NIXL_EP_FORCE_UCX_TCP=1`` to force UCX away from RDMA
on clusters without RDMA device access. Use ``--ucx-intranode`` or
``NIXL_EP_UCX_INTRANODE=1`` for same-node CUDA IPC/NVLink runs. Use
``--ucx-gda-auto-device`` or ``NIXL_EP_UCX_GDA_AUTO_DEVICE=1`` to opt into
IBGDA preflight and automatic ``UCX_NET_DEVICES`` selection.
For same-node Kubernetes validation, ``--ucx-intranode`` requires ranks to
share PID/IPC namespaces; separate pods commonly cannot use UCX shared-memory
active messages. Use ``--spawn-local-ranks 2 --ucx-intranode`` in a pod that
has two GPUs, or deploy ranks with compatible host/shared PID and IPC settings.
For multi-node tests with one multi-GPU pod per node, run one parent per pod
with ``--spawn-local-ranks``, a shared external TCPStore,
``--global-world-size``, and either ``--rank-base`` or ``--node-rank``. Use
``--ucx-mixed-gda-intranode`` with ``--ucx-gda-auto-device`` when the same run
must exercise local CUDA IPC/NVL peers and remote IBGDA peers.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

EP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EP_ROOT))


DEFAULT_STORE_PORT = 9999
SNAPSHOT_CONTROL_ENV = "DYN_SNAPSHOT_CONTROL_DIR"
READY_FOR_CHECKPOINT = "ready-for-checkpoint"
SNAPSHOT_COMPLETE = "snapshot-complete"
RESTORE_COMPLETE = "restore-complete"
INTRANODE_ENV = "NIXL_EP_UCX_INTRANODE"
INTRANODE_DEFAULT_UCX_TLS = "sm,cuda_ipc,cuda_copy,self"
INTRANODE_SHARED_MEMORY_TLS = frozenset(
    ("sm", "posix", "sysv", "cma", "knem", "xpmem")
)
LOCAL_SPAWN_CHILD_ENV = "NIXL_EP_LOCAL_SPAWN_CHILD"
GLOBAL_WORLD_SIZE_ENV = "NIXL_EP_GLOBAL_WORLD_SIZE"
RANK_BASE_ENV = "NIXL_EP_RANK_BASE"
NODE_RANK_ENV = "NIXL_EP_NODE_RANK"
LOCAL_WORLD_SIZE_ENV = "NIXL_EP_LOCAL_WORLD_SIZE"
LOCAL_SPAWN_MASTER_ADDR = "127.0.0.1"
GDA_AUTO_DEVICE_ENV = "NIXL_EP_UCX_GDA_AUTO_DEVICE"
GDA_DEVICE_ENV = "NIXL_EP_UCX_GDA_DEVICE"
GDA_DEVICE_CANDIDATES_ENV = "NIXL_EP_UCX_GDA_DEVICE_CANDIDATES"
GDA_RETAIN_ENV_VARS = (
    "UCX_IB_GDA_RETAIN_INACTIVE_CTX",
    "UCX_GGA_GDA_RETAIN_INACTIVE_CTX",
)
GDA_DEFAULT_UCX_TLS = "rc_gda,rc,ud,cuda_copy,cuda_ipc,self"
MIXED_GDA_INTRANODE_ENV = "NIXL_EP_UCX_MIXED_GDA_INTRANODE"
MIXED_GDA_INTRANODE_DEFAULT_UCX_TLS = (
    "rc_gda,rc,ud,sm,cuda_ipc,cuda_copy,self"
)
GDA_CONTROL_TLS = frozenset(("rc", "ud"))

nixl_ep: Any


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


def env_int(*names: str) -> int | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return int(value)
    return None


@dataclass(frozen=True)
class RankEnv:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int


@dataclass(frozen=True)
class LocalSpawnConfig:
    local_world_size: int
    global_world_size: int
    rank_base: int
    cuda_visible_devices: str


@dataclass
class IterationState:
    recv_x: torch.Tensor
    recv_count: torch.Tensor
    handle: tuple[Any, ...]
    combined_x: torch.Tensor


@dataclass
class CapturedIteration:
    graph: torch.cuda.CUDAGraph
    state: IterationState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate NIXL EP checkpoint_pause_preserve_va/"
            "checkpoint_resume_preserve_va across real ranks."
        )
    )
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-experts-per-rank", type=int, default=1)
    parser.add_argument("--num-topk", type=int, default=1)
    parser.add_argument("--capture-warmups", type=int, default=3)
    parser.add_argument("--skip-cuda-graph", action="store_true")
    parser.add_argument(
        "--allow-eager-fallback",
        action="store_true",
        help=(
            "If CUDA graph capture fails, continue with eager validation. "
            "By default graph capture errors fail the run."
        ),
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30_000,
        help=(
            "Timeout for NIXL EP GPU waits and native metadata/peer-info "
            "connection waits. Use a smaller value to fail fast when UCX "
            "endpoint setup hangs."
        ),
    )
    parser.add_argument(
        "--spawn-local-ranks",
        type=int,
        default=0,
        help=(
            "Single-pod local launcher. The parent process spawns this many "
            "child rank processes in the same container/PID/IPC namespace, "
            "exposes the first N visible GPUs via CUDA_VISIBLE_DEVICES, sets "
            "global RANK/WORLD_SIZE and per-pod LOCAL_RANK/LOCAL_WORLD_SIZE, "
            "and runs the normal checkpoint/restore flow in each child. For "
            "pure same-node validation, use this with --ucx-intranode. For "
            "multi-node validation, run one parent per pod with "
            "--external-store, --global-world-size, and --rank-base or "
            "--node-rank. Separate same-node pods generally cannot use UCX "
            "shared-memory active messages and are not a supported "
            "intranode topology."
        ),
    )
    parser.add_argument(
        "--global-world-size",
        type=int,
        default=env_int(GLOBAL_WORLD_SIZE_ENV, "WORLD_SIZE"),
        help=(
            "Global WORLD_SIZE for --spawn-local-ranks children. Defaults to "
            f"${GLOBAL_WORLD_SIZE_ENV}, then $WORLD_SIZE, then the local "
            "spawn count for single-pod intranode runs."
        ),
    )
    parser.add_argument(
        "--rank-base",
        "--global-rank-base",
        dest="rank_base",
        type=int,
        default=env_int(RANK_BASE_ENV),
        help=(
            "Global rank assigned to local child 0 for --spawn-local-ranks. "
            "Child ranks are rank_base + local_rank. Defaults to "
            f"${RANK_BASE_ENV}, then --node-rank * --spawn-local-ranks, then "
            "0."
        ),
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=env_int(NODE_RANK_ENV, "NODE_RANK"),
        help=(
            "Per-pod node rank used to derive --rank-base when --rank-base is "
            f"not set. Defaults to ${NODE_RANK_ENV}, then $NODE_RANK."
        ),
    )
    parser.add_argument("--store-master-addr", default=os.getenv("MASTER_ADDR"))
    parser.add_argument("--store-port", type=int, default=DEFAULT_STORE_PORT)
    parser.add_argument(
        "--external-store",
        action="store_true",
        default=env_flag("NIXL_EP_EXTERNAL_STORE"),
        help=(
            "Connect all ranks to an already-running TCPStore instead of "
            "creating the TCPStore in rank 0."
        ),
    )
    parser.add_argument(
        "--barrier-timeout-sec",
        type=float,
        default=300.0,
        help="Timeout for TCPStore coordination barriers.",
    )
    parser.add_argument(
        "--hook-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for external checkpoint hooks. Each rank writes "
            "rank_<rank>.paused.json after checkpoint pause."
        ),
    )
    parser.add_argument(
        "--wait-for-resume-file",
        action="store_true",
        help=(
            "After writing the paused hook file, wait for "
            "rank_<rank>.resume before calling checkpoint_resume_preserve_va()."
        ),
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional sleep after checkpoint pause for external orchestration.",
    )
    parser.add_argument(
        "--snapshot-control-dir",
        type=Path,
        default=(
            Path(os.environ[SNAPSHOT_CONTROL_ENV])
            if os.getenv(SNAPSHOT_CONTROL_ENV)
            else None
        ),
        help=(
            "Dynamo snapshot-control directory. Defaults to "
            f"${SNAPSHOT_CONTROL_ENV}. When set, the script writes "
            "ready-for-checkpoint, exits on snapshot-complete in the original "
            "process, and resumes only after restore-complete in the restored "
            "process."
        ),
    )
    parser.add_argument(
        "--snapshot-timeout-sec",
        type=float,
        default=900.0,
        help="Timeout while waiting for Dynamo snapshot-control sentinels.",
    )
    parser.add_argument(
        "--metadata-namespace",
        default=os.getenv("NIXL_EP_METADATA_NAMESPACE", "checkpoint_preserve_va"),
        help=(
            "Shared namespace for NIXL metadata keys in TCPStore. Set this to "
            "a per-run value when reusing an external TCPStore."
        ),
    )
    parser.add_argument(
        "--metadata-generation",
        default=(
            os.getenv("NIXL_EP_METADATA_GENERATION")
            or os.getenv("TORCHELASTIC_RUN_ID")
            or os.getenv("SLURM_JOB_ID")
            or os.getenv("OMPI_COMM_WORLD_JOBID")
        ),
        help=(
            "Shared generation for NIXL metadata keys in TCPStore. Change it "
            "between attempts if an external TCPStore may contain stale keys. "
            "Required with --external-store unless a supported job id env var "
            "is set."
        ),
    )
    parser.add_argument(
        "--ucx-tls",
        default=os.getenv("NIXL_EP_UCX_TLS"),
        help=(
            "Optional UCX_TLS value to set before importing nixl_ep. Leave "
            "unset to preserve UCX/NIXL default RDMA-capable transport "
            "selection."
        ),
    )
    parser.add_argument(
        "--force-ucx-tcp",
        action="store_true",
        default=env_flag("NIXL_EP_FORCE_UCX_TCP"),
        help=(
            "Force UCX over TCP by setting UCX_TLS=tcp,cuda_copy,self before "
            "creating the NIXL EP runtime. RDMA remains the default unless "
            "this flag or --ucx-tls is used."
        ),
    )
    parser.add_argument(
        "--ucx-intranode",
        action="store_true",
        default=env_flag(INTRANODE_ENV),
        help=(
            "Configure UCX for same-node NIXL EP low-latency runs. The "
            "script sets UCX_TLS=sm,cuda_ipc,cuda_copy,self before importing "
            "nixl_ep and clears stale network-device restrictions with "
            "UCX_NET_DEVICES=all. Use --ucx-tls to override the TLS list; "
            "the override must still include cuda_ipc plus a shared-memory "
            "active-message transport. Ranks must share PID/IPC namespaces; "
            "Kubernetes separate pods usually do not. Use "
            "--spawn-local-ranks 2 in one pod/container or host/shared "
            "PID/IPC-compatible pod settings for same-node tests. Defaults to "
            f"${INTRANODE_ENV}."
        ),
    )
    parser.add_argument(
        "--ucx-mixed-gda-intranode",
        "--mixed-local-remote",
        action="store_true",
        default=env_flag(MIXED_GDA_INTRANODE_ENV),
        dest="ucx_mixed_gda_intranode",
        help=(
            "Configure UCX for one multi-GPU pod per node with local peers "
            "using CUDA IPC/NVL shared-memory transports and remote peers "
            "using IBGDA. Requires --spawn-local-ranks, "
            "--global-world-size greater than the local rank count, "
            "--ucx-gda-auto-device, and a shared external TCPStore. Sets "
            "UCX_TLS=rc_gda,rc,ud,sm,cuda_ipc,cuda_copy,self by default and "
            "uses the selected rc_gda device plus ordinary HCA in "
            "UCX_NET_DEVICES. Defaults to "
            f"${MIXED_GDA_INTRANODE_ENV}."
        ),
    )
    parser.add_argument(
        "--ucx-gda-auto-device",
        action="store_true",
        default=env_flag(GDA_AUTO_DEVICE_ENV),
        help=(
            "Opt into IBGDA runtime validation. The script sets the UCX GDA "
            "retain-inactive-context knobs before importing nixl_ep, runs "
            "ucx_info -d after CUDA context initialization, and sets "
            "UCX_NET_DEVICES to the selected full rc_gda device plus "
            "its ordinary HCA, such as cuda0-mlx5_5:1,mlx5_5:1. "
            "Defaults to "
            f"${GDA_AUTO_DEVICE_ENV}."
        ),
    )
    parser.add_argument(
        "--ucx-gda-device",
        default=os.getenv(GDA_DEVICE_ENV),
        help=(
            "Optional full UCX rc_gda device to use with "
            "--ucx-gda-auto-device, for example cuda0-mlx5_5:1. The script "
            "still adds the ordinary HCA for active-message/control metadata. "
            f"Defaults to ${GDA_DEVICE_ENV}."
        ),
    )
    parser.add_argument(
        "--ucx-gda-device-candidates",
        default=os.getenv(GDA_DEVICE_CANDIDATES_ENV),
        help=(
            "Optional comma-separated preferred full UCX rc_gda devices for "
            "--ucx-gda-auto-device. The first discovered candidate matching "
            "the active CUDA device is selected. Defaults to "
            f"${GDA_DEVICE_CANDIDATES_ENV}."
        ),
    )
    return parser.parse_args()


def remove_cli_option(argv: list[str], option: str, takes_value: bool) -> list[str]:
    result: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            skip_next = takes_value
            continue
        if arg.startswith(f"{option}="):
            continue
        result.append(arg)
    return result


def has_cli_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def local_spawn_cuda_visible_devices(num_ranks: int) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",")]
        devices = [device for device in devices if device]
        if len(devices) < num_ranks:
            raise ValueError(
                "--spawn-local-ranks requested "
                f"{num_ranks} ranks, but CUDA_VISIBLE_DEVICES exposes only "
                f"{len(devices)} device(s): {visible_devices!r}"
            )
        return ",".join(devices[:num_ranks])

    return ",".join(str(rank) for rank in range(num_ranks))


def resolve_local_spawn_config(args: argparse.Namespace) -> LocalSpawnConfig:
    local_world_size = args.spawn_local_ranks
    if local_world_size < 1:
        raise ValueError("--spawn-local-ranks requires at least 1 rank")
    if args.force_ucx_tcp:
        raise ValueError("--spawn-local-ranks conflicts with --force-ucx-tcp")

    global_world_size = (
        local_world_size
        if args.global_world_size is None
        else args.global_world_size
    )
    if global_world_size < 2:
        raise ValueError("Real multi-rank validation requires WORLD_SIZE >= 2")
    if local_world_size > global_world_size:
        raise ValueError(
            "--spawn-local-ranks cannot exceed --global-world-size: "
            f"local={local_world_size}, global={global_world_size}"
        )

    if args.rank_base is not None:
        rank_base = args.rank_base
    elif args.node_rank is not None:
        rank_base = args.node_rank * local_world_size
    else:
        rank_base = 0

    if rank_base < 0:
        raise ValueError("--rank-base must be non-negative")
    if rank_base + local_world_size > global_world_size:
        raise ValueError(
            "Local rank range exceeds global world size: "
            f"rank_base={rank_base}, local_world_size={local_world_size}, "
            f"global_world_size={global_world_size}"
        )

    has_remote_ranks = global_world_size > local_world_size
    if args.ucx_intranode and has_remote_ranks:
        raise ValueError(
            "--ucx-intranode is only for pure intrapod intranode validation "
            "where all ranks are spawned in one pod. For multi-pod runs, use "
            "--ucx-gda-auto-device for pure IBGDA or "
            "--ucx-mixed-gda-intranode for local CUDA IPC/NVL plus remote "
            "IBGDA."
        )
    if args.ucx_intranode and rank_base != 0:
        raise ValueError(
            "--ucx-intranode is pure intrapod intranode mode and requires "
            "--rank-base 0"
        )

    if has_remote_ranks:
        if not args.external_store:
            raise ValueError(
                "--spawn-local-ranks with remote ranks requires "
                "--external-store and a shared TCPStore reachable from every "
                "pod"
            )
        if args.store_master_addr is None:
            raise ValueError(
                "--spawn-local-ranks with remote ranks requires "
                "--store-master-addr or MASTER_ADDR for the shared external "
                "TCPStore"
            )

    if args.ucx_mixed_gda_intranode:
        if local_world_size < 2:
            raise ValueError(
                "--ucx-mixed-gda-intranode requires at least two local ranks "
                "per pod so local CUDA IPC/NVL peers are exercised"
            )
        if not has_remote_ranks:
            raise ValueError(
                "--ucx-mixed-gda-intranode requires remote ranks; for pure "
                "single-pod intranode use --ucx-intranode"
            )
        if not args.ucx_gda_auto_device:
            raise ValueError(
                "--ucx-mixed-gda-intranode requires --ucx-gda-auto-device "
                "for remote IBGDA peers"
            )

    if args.ucx_gda_auto_device and local_world_size > 1 and args.ucx_gda_device:
        raise ValueError(
            "--ucx-gda-device names one rc_gda device for one active CUDA "
            "ordinal. With multiple local child ranks, use "
            "--ucx-gda-device-candidates so each child can select the "
            "candidate matching its LOCAL_RANK/CUDA device."
        )

    return LocalSpawnConfig(
        local_world_size=local_world_size,
        global_world_size=global_world_size,
        rank_base=rank_base,
        cuda_visible_devices=local_spawn_cuda_visible_devices(local_world_size),
    )


def stream_child_output(
    global_rank: int,
    local_rank: int,
    proc: subprocess.Popen[str],
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(
            f"[global-rank {global_rank} local-rank {local_rank}] {line}",
            end="",
            flush=True,
        )


def terminate_child_processes(
    procs: list[tuple[int, subprocess.Popen[str]]],
    grace_sec: float = 10.0,
) -> None:
    for _rank, proc in procs:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if all(proc.poll() is not None for _rank, proc in procs):
            return
        time.sleep(0.1)

    for _rank, proc in procs:
        if proc.poll() is None:
            proc.kill()


def run_spawn_local_ranks(args: argparse.Namespace) -> None:
    if env_flag(LOCAL_SPAWN_CHILD_ENV):
        raise RuntimeError(
            f"{LOCAL_SPAWN_CHILD_ENV}=1 child process must not spawn ranks"
        )
    spawn_config = resolve_local_spawn_config(args)

    device_count = torch.cuda.device_count()
    if device_count < spawn_config.local_world_size:
        raise RuntimeError(
            "--spawn-local-ranks requested "
            f"{spawn_config.local_world_size} ranks, but torch sees only "
            f"{device_count} CUDA device(s)"
        )

    base_child_args = remove_cli_option(
        sys.argv[1:], "--spawn-local-ranks", takes_value=True
    )
    base_child_args = remove_cli_option(
        base_child_args, "--global-world-size", takes_value=True
    )
    base_child_args = remove_cli_option(
        base_child_args, "--rank-base", takes_value=True
    )
    base_child_args = remove_cli_option(
        base_child_args, "--global-rank-base", takes_value=True
    )
    base_child_args = remove_cli_option(
        base_child_args, "--node-rank", takes_value=True
    )
    if not args.external_store:
        base_child_args = remove_cli_option(
            base_child_args, "--store-master-addr", takes_value=True
        )
        base_child_args.extend(
            ["--store-master-addr", LOCAL_SPAWN_MASTER_ADDR]
        )
    if (
        spawn_config.global_world_size == spawn_config.local_world_size
        and not args.ucx_gda_auto_device
        and not args.ucx_mixed_gda_intranode
        and not args.ucx_intranode
        and not has_cli_option(base_child_args, "--ucx-intranode")
    ):
        base_child_args.append("--ucx-intranode")

    child_store_master_addr = (
        LOCAL_SPAWN_MASTER_ADDR
        if not args.external_store
        else args.store_master_addr
    )
    print(
        "[local-spawn] launching "
        f"{spawn_config.local_world_size} local ranks in one container with "
        f"global ranks "
        f"{spawn_config.rank_base}.."
        f"{spawn_config.rank_base + spawn_config.local_world_size - 1}, "
        f"WORLD_SIZE={spawn_config.global_world_size}, "
        f"CUDA_VISIBLE_DEVICES={spawn_config.cuda_visible_devices}, "
        f"store_master_addr={child_store_master_addr}, "
        f"store_port={args.store_port}",
        flush=True,
    )

    procs: list[tuple[int, subprocess.Popen[str]]] = []
    threads: list[threading.Thread] = []
    script_path = Path(__file__).resolve()
    try:
        for local_rank in range(spawn_config.local_world_size):
            global_rank = spawn_config.rank_base + local_rank
            child_env = os.environ.copy()
            child_env[LOCAL_SPAWN_CHILD_ENV] = "1"
            child_env[GLOBAL_WORLD_SIZE_ENV] = str(
                spawn_config.global_world_size
            )
            child_env[RANK_BASE_ENV] = str(spawn_config.rank_base)
            child_env[LOCAL_WORLD_SIZE_ENV] = str(
                spawn_config.local_world_size
            )
            child_env["RANK"] = str(global_rank)
            child_env["WORLD_SIZE"] = str(spawn_config.global_world_size)
            child_env["LOCAL_RANK"] = str(local_rank)
            child_env["LOCAL_WORLD_SIZE"] = str(spawn_config.local_world_size)
            child_env["CUDA_VISIBLE_DEVICES"] = (
                spawn_config.cuda_visible_devices
            )
            if not args.external_store:
                child_env["MASTER_ADDR"] = LOCAL_SPAWN_MASTER_ADDR
            child_env["MASTER_PORT"] = str(args.store_port)
            child_env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                [sys.executable, str(script_path), *base_child_args],
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            procs.append((global_rank, proc))
            thread = threading.Thread(
                target=stream_child_output,
                args=(global_rank, local_rank, proc),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        exit_codes: dict[int, int] = {}
        while len(exit_codes) < len(procs):
            for rank, proc in procs:
                if rank in exit_codes:
                    continue
                returncode = proc.poll()
                if returncode is None:
                    continue
                exit_codes[rank] = returncode
                if returncode != 0:
                    terminate_child_processes(procs)
            time.sleep(0.1)

        for thread in threads:
            thread.join(timeout=1.0)

        failures = {
            rank: returncode
            for rank, returncode in exit_codes.items()
            if returncode != 0
        }
        if failures:
            raise RuntimeError(
                "local rank subprocesses failed with exit codes "
                f"{failures}"
            )

        print(
            "PASS local_spawn_ranks="
            f"{spawn_config.local_world_size} "
            f"rank_base={spawn_config.rank_base} "
            f"global_world_size={spawn_config.global_world_size}",
            flush=True,
        )
    except BaseException:
        terminate_child_processes(procs)
        raise


def read_rank_env() -> RankEnv:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(
        os.environ.get(
            LOCAL_WORLD_SIZE_ENV,
            os.environ.get("LOCAL_WORLD_SIZE", "1"),
        )
    )
    return RankEnv(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )


def validate_runtime_topology(
    args: argparse.Namespace,
    env: RankEnv,
) -> None:
    if env.rank < 0 or env.rank >= env.world_size:
        raise ValueError(
            f"RANK must be in [0, WORLD_SIZE); got rank={env.rank}, "
            f"world_size={env.world_size}"
        )
    if env.local_world_size < 1:
        raise ValueError(
            f"LOCAL_WORLD_SIZE must be at least 1; got {env.local_world_size}"
        )
    if env.local_rank < 0 or env.local_rank >= env.local_world_size:
        raise ValueError(
            "LOCAL_RANK must be in [0, LOCAL_WORLD_SIZE); got "
            f"local_rank={env.local_rank}, "
            f"local_world_size={env.local_world_size}"
        )

    if args.ucx_intranode and env.local_world_size != env.world_size:
        raise ValueError(
            "--ucx-intranode supports only the clarified single-pod intranode "
            "topology where every rank is in one pod/container. "
            f"Got WORLD_SIZE={env.world_size} and "
            f"LOCAL_WORLD_SIZE={env.local_world_size}. Same-node ranks split "
            "across separate Kubernetes pods are not supported because UCX "
            "shared-memory active messages require compatible shared "
            "PID/IPC namespaces. Use --spawn-local-ranks in one multi-GPU pod."
        )

    if not args.ucx_mixed_gda_intranode:
        return

    if env.local_world_size < 2:
        raise ValueError(
            "--ucx-mixed-gda-intranode requires at least two ranks in this "
            f"pod; got LOCAL_WORLD_SIZE={env.local_world_size}"
        )
    if env.world_size <= env.local_world_size:
        raise ValueError(
            "--ucx-mixed-gda-intranode requires both local and remote peers; "
            f"got WORLD_SIZE={env.world_size}, "
            f"LOCAL_WORLD_SIZE={env.local_world_size}. For pure single-pod "
            "intranode use --ucx-intranode."
        )
    if not args.external_store:
        raise ValueError(
            "--ucx-mixed-gda-intranode requires --external-store so all pods "
            "coordinate through one shared TCPStore"
        )
    if not args.ucx_gda_auto_device:
        raise ValueError(
            "--ucx-mixed-gda-intranode requires --ucx-gda-auto-device for "
            "remote IBGDA peers"
        )


def create_store(args: argparse.Namespace, env: RankEnv) -> dist.TCPStore:
    master_addr = args.store_master_addr
    if master_addr is None:
        raise ValueError(
            "--store-master-addr or MASTER_ADDR is required for multi-rank runs"
        )

    is_master = env.rank == 0 and not args.external_store
    host_name = "0.0.0.0" if is_master else master_addr
    return dist.TCPStore(
        host_name=host_name,
        port=args.store_port,
        is_master=is_master,
        wait_for_workers=False,
        timeout=timedelta(seconds=args.barrier_timeout_sec),
    )


def parse_ucx_tls(ucx_tls: str) -> set[str]:
    return {
        item.strip().split(":", maxsplit=1)[0].lower()
        for item in ucx_tls.split(",")
        if item.strip()
    }


def validate_intranode_ucx_tls(ucx_tls: str) -> None:
    tls = parse_ucx_tls(ucx_tls)
    if "all" in tls:
        raise ValueError(
            "--ucx-intranode requires an explicit UCX_TLS allow-list; "
            f"got {ucx_tls!r}"
        )
    if any(item.startswith("^") for item in tls):
        raise ValueError(
            "--ucx-intranode requires an explicit UCX_TLS allow-list; "
            f"got exclusion list {ucx_tls!r}"
        )
    if "cuda_ipc" not in tls:
        raise ValueError(
            "--ucx-intranode requires UCX_TLS to include cuda_ipc for "
            f"same-node GPU memory views; got {ucx_tls!r}"
        )
    if tls.isdisjoint(INTRANODE_SHARED_MEMORY_TLS):
        allowed = ",".join(sorted(INTRANODE_SHARED_MEMORY_TLS))
        raise ValueError(
            "--ucx-intranode requires UCX_TLS to include a shared-memory "
            "active-message transport for metadata/control "
            f"({allowed}); got {ucx_tls!r}"
        )


def validate_mixed_gda_intranode_ucx_tls(ucx_tls: str) -> None:
    tls = parse_ucx_tls(ucx_tls)
    if "all" in tls:
        raise ValueError(
            "--ucx-mixed-gda-intranode requires an explicit UCX_TLS "
            f"allow-list; got {ucx_tls!r}"
        )
    if any(item.startswith("^") for item in tls):
        raise ValueError(
            "--ucx-mixed-gda-intranode requires an explicit UCX_TLS "
            f"allow-list; got exclusion list {ucx_tls!r}"
        )
    if "rc_gda" not in tls:
        raise ValueError(
            "--ucx-mixed-gda-intranode requires UCX_TLS to include rc_gda "
            f"for remote IBGDA peers; got {ucx_tls!r}"
        )
    if tls.isdisjoint(GDA_CONTROL_TLS):
        control = ",".join(sorted(GDA_CONTROL_TLS))
        raise ValueError(
            "--ucx-mixed-gda-intranode requires UCX_TLS to include rc or ud "
            "for active-message/control metadata "
            f"({control}); got {ucx_tls!r}"
        )
    validate_intranode_ucx_tls(ucx_tls)


def configure_ucx_intranode_transport(args: argparse.Namespace) -> None:
    ucx_tls = args.ucx_tls or INTRANODE_DEFAULT_UCX_TLS
    validate_intranode_ucx_tls(ucx_tls)

    previous_tls = os.environ.get("UCX_TLS")
    os.environ["UCX_TLS"] = ucx_tls
    print(
        f"[transport] set UCX_TLS={ucx_tls} for intranode CUDA IPC/NVL"
        + (f" (overrode {previous_tls})" if previous_tls else ""),
        flush=True,
    )
    print(
        "[transport] intranode mode requires ranks to share PID/IPC "
        "namespaces; use --spawn-local-ranks in one pod/container or "
        "host/shared PID/IPC-compatible pod settings for Kubernetes tests",
        flush=True,
    )

    previous_devices = os.environ.get("UCX_NET_DEVICES")
    os.environ["UCX_NET_DEVICES"] = "all"
    print(
        "[transport] set UCX_NET_DEVICES=all for intranode CUDA IPC/NVL"
        + (f" (overrode {previous_devices})" if previous_devices else ""),
        flush=True,
    )


def configure_ucx_mixed_gda_intranode_transport(
    args: argparse.Namespace,
) -> None:
    ucx_tls = args.ucx_tls or MIXED_GDA_INTRANODE_DEFAULT_UCX_TLS
    validate_mixed_gda_intranode_ucx_tls(ucx_tls)

    previous_tls = os.environ.get("UCX_TLS")
    os.environ["UCX_TLS"] = ucx_tls
    print(
        "[transport] set "
        f"UCX_TLS={ucx_tls} for mixed local CUDA IPC/NVL plus remote IBGDA"
        + (f" (overrode {previous_tls})" if previous_tls else ""),
        flush=True,
    )
    print(
        "[transport] mixed mode requires one multi-GPU pod per node. Local "
        "ranks must share PID/IPC namespaces through --spawn-local-ranks; "
        "remote ranks must use a shared external TCPStore and reachable "
        "IBGDA/HCA ports.",
        flush=True,
    )


def configure_ucx_transport(args: argparse.Namespace) -> None:
    if args.ucx_gda_auto_device and args.force_ucx_tcp:
        raise ValueError("--ucx-gda-auto-device conflicts with --force-ucx-tcp")
    if args.ucx_intranode and args.force_ucx_tcp:
        raise ValueError("--ucx-intranode conflicts with --force-ucx-tcp")
    if args.ucx_mixed_gda_intranode and args.force_ucx_tcp:
        raise ValueError(
            "--ucx-mixed-gda-intranode conflicts with --force-ucx-tcp"
        )
    if args.ucx_intranode and args.ucx_gda_auto_device:
        raise ValueError(
            "--ucx-intranode configures pure same-node CUDA IPC/NVL and "
            "conflicts with --ucx-gda-auto-device. Use --ucx-gda-auto-device "
            "alone for a same-node IBGDA/RDMA fallback."
        )
    if args.ucx_intranode and args.ucx_mixed_gda_intranode:
        raise ValueError(
            "--ucx-intranode configures pure intrapod CUDA IPC/NVL and "
            "conflicts with --ucx-mixed-gda-intranode"
        )
    if args.ucx_mixed_gda_intranode and not args.ucx_gda_auto_device:
        raise ValueError(
            "--ucx-mixed-gda-intranode requires --ucx-gda-auto-device"
        )

    if args.ucx_gda_auto_device:
        configure_ucx_gda_pre_import()

    if args.ucx_mixed_gda_intranode:
        configure_ucx_mixed_gda_intranode_transport(args)
        return

    if args.ucx_intranode:
        configure_ucx_intranode_transport(args)
        return

    if args.ucx_tls:
        os.environ["UCX_TLS"] = args.ucx_tls
        print(f"[transport] set UCX_TLS={args.ucx_tls}", flush=True)
        return

    if args.force_ucx_tcp:
        os.environ["UCX_TLS"] = "tcp,cuda_copy,self"
        print("[transport] forced UCX_TLS=tcp,cuda_copy,self", flush=True)
        return

    if args.ucx_gda_auto_device:
        if "UCX_TLS" in os.environ:
            print(
                f"[transport] using existing UCX_TLS={os.environ['UCX_TLS']}",
                flush=True,
            )
        else:
            os.environ["UCX_TLS"] = GDA_DEFAULT_UCX_TLS
            print(f"[transport] set UCX_TLS={GDA_DEFAULT_UCX_TLS}", flush=True)


def configure_ucx_gda_pre_import() -> None:
    for name in GDA_RETAIN_ENV_VARS:
        value = os.getenv(name)
        if value is not None and not env_flag(name):
            raise ValueError(
                f"{name} must be set to y when --ucx-gda-auto-device is used; "
                f"got {value!r}"
            )
        os.environ[name] = "y"
        print(f"[transport] set {name}=y", flush=True)


def tail_text(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return f"... <truncated> ...\n{text[-max_chars:]}"


def run_ucx_info_d(env: dict[str, str]) -> str:
    ucx_info = shutil.which("ucx_info")
    if ucx_info is None:
        raise RuntimeError(
            "IBGDA preflight requires ucx_info in PATH. Install UCX tools or "
            "disable --ucx-gda-auto-device."
        )

    result = subprocess.run(
        [ucx_info, "-d"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"IBGDA preflight command failed: ucx_info -d exited with "
            f"{result.returncode}\n{tail_text(result.stdout)}"
        )
    return result.stdout


def extract_ucx_rc_gda_devices(ucx_info_output: str) -> list[str]:
    devices: list[str] = []
    active_transport = None

    for line in ucx_info_output.splitlines():
        transport_match = re.search(r"\bTransport:\s+(\S+)", line)
        if transport_match:
            active_transport = transport_match.group(1)
            continue

        if active_transport != "rc_gda":
            continue

        device_match = re.search(r"\bDevice:\s+(\S+)", line)
        if device_match:
            device = device_match.group(1)
            if device not in devices:
                devices.append(device)

    return devices


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_ucx_rc_gda_device_name(device: str) -> None:
    if not re.fullmatch(r"cuda\d+-mlx5_[^:\s]+:\d+", device):
        raise ValueError(
            "Expected full UCX rc_gda device name like cuda0-mlx5_5:1; "
            f"got {device!r}"
        )


def select_ucx_rc_gda_device(
    devices: list[str],
    cuda_device: int,
    explicit_device: str | None = None,
    candidate_devices: list[str] | None = None,
) -> str:
    full_device_pattern = re.compile(r"cuda\d+-mlx5_[^:\s]+:\d+")
    full_devices = [
        device for device in devices if full_device_pattern.fullmatch(device)
    ]
    if not full_devices:
        raise RuntimeError(
            "IBGDA preflight found Transport: rc_gda but no full CUDA/HCA "
            "device names. Expected names like cuda0-mlx5_5:1; found "
            f"{devices}."
        )

    cuda_prefix = f"cuda{cuda_device}-"
    matching_devices = [
        device for device in full_devices if device.startswith(cuda_prefix)
    ]

    if explicit_device:
        validate_ucx_rc_gda_device_name(explicit_device)
        if not explicit_device.startswith(cuda_prefix):
            raise RuntimeError(
                "IBGDA preflight explicit device does not match the active "
                f"CUDA device cuda{cuda_device}: {explicit_device}. Set "
                "CUDA_VISIBLE_DEVICES/LOCAL_RANK consistently or choose a "
                f"{GDA_DEVICE_ENV} value with prefix {cuda_prefix}."
            )
        if explicit_device not in full_devices:
            raise RuntimeError(
                "IBGDA preflight explicit device was not discovered by "
                f"ucx_info -d: {explicit_device}. Found {full_devices}."
            )
        return explicit_device

    if candidate_devices:
        for device in candidate_devices:
            validate_ucx_rc_gda_device_name(device)
        for device in candidate_devices:
            if device in matching_devices:
                return device
        matching_candidates = [
            device for device in candidate_devices if device.startswith(cuda_prefix)
        ]
        raise RuntimeError(
            "IBGDA preflight found rc_gda devices, but none of the preferred "
            "candidates for the active CUDA device were discovered. "
            f"cuda_device=cuda{cuda_device}, candidates={matching_candidates}, "
            f"discovered={matching_devices}."
        )

    if not matching_devices:
        raise RuntimeError(
            "IBGDA preflight found rc_gda devices, but none match the active "
            f"CUDA device cuda{cuda_device}. Found {full_devices}. Set "
            "CUDA_VISIBLE_DEVICES/LOCAL_RANK consistently or set "
            "UCX_NET_DEVICES manually and disable --ucx-gda-auto-device."
        )

    return matching_devices[0]


def ordinary_hca_from_ucx_rc_gda_device(full_gda_device: str) -> str:
    match = re.fullmatch(
        r"cuda\d+-(?P<hca>mlx5_[^:\s]+:\d+)", full_gda_device
    )
    if match is None:
        raise ValueError(
            "Expected full UCX rc_gda device name like cuda0-mlx5_5:1; "
            f"got {full_gda_device!r}"
        )
    return match.group("hca")


def format_ucx_gda_net_devices(full_gda_device: str) -> str:
    hca_device = ordinary_hca_from_ucx_rc_gda_device(full_gda_device)
    return f"{full_gda_device},{hca_device}"


def configure_ucx_gda_auto_device(
    args: argparse.Namespace,
    env: RankEnv,
) -> None:
    if not args.ucx_gda_auto_device:
        return

    cuda_device = torch.cuda.current_device()
    print(
        f"[transport] running IBGDA preflight for rank {env.rank} on "
        f"cuda{cuda_device}",
        flush=True,
    )

    discovery_env = os.environ.copy()
    discovery_env["UCX_NET_DEVICES"] = "all"
    discovery_output = run_ucx_info_d(discovery_env)
    devices = extract_ucx_rc_gda_devices(discovery_output)
    candidate_devices = parse_csv(args.ucx_gda_device_candidates)
    print(
        f"[transport] discovered rc_gda devices={devices}; "
        f"explicit={args.ucx_gda_device}; candidates={candidate_devices}",
        flush=True,
    )
    if not devices:
        retain_settings = ", ".join(
            f"{name}={os.getenv(name)}" for name in GDA_RETAIN_ENV_VARS
        )
        raise RuntimeError(
            "IBGDA preflight failed: ucx_info -d did not expose "
            "Transport: rc_gda with UCX_NET_DEVICES=all. Ensure the image's "
            "UCX build includes rc_gda, the mlx5 GDA module loads, and "
            f"{retain_settings}. ucx_info output tail:\n"
            f"{tail_text(discovery_output)}"
        )

    selected_device = select_ucx_rc_gda_device(
        devices,
        cuda_device,
        explicit_device=args.ucx_gda_device,
        candidate_devices=candidate_devices,
    )
    selected_devices = format_ucx_gda_net_devices(selected_device)
    previous_devices = os.environ.get("UCX_NET_DEVICES")
    os.environ["UCX_NET_DEVICES"] = selected_devices
    print(
        "[transport] selected UCX_NET_DEVICES="
        f"{selected_devices} for rc_gda plus AM/control metadata"
        + (f" from candidates {candidate_devices}" if candidate_devices else "")
        + (f" (overrode {previous_devices})" if previous_devices else ""),
        flush=True,
    )

    validation_env = os.environ.copy()
    validation_output = run_ucx_info_d(validation_env)
    validated_devices = extract_ucx_rc_gda_devices(validation_output)
    if selected_device not in validated_devices:
        raise RuntimeError(
            "IBGDA preflight failed: selected UCX_NET_DEVICES="
            f"{selected_devices} did not keep Transport: rc_gda visible "
            f"for {selected_device}. "
            f"ucx_info reported rc_gda devices {validated_devices}. Output "
            f"tail:\n{tail_text(validation_output)}"
        )


def initialize_cuda_runtime(env: RankEnv) -> None:
    torch.cuda.set_device(env.local_rank)
    torch.empty((), device="cuda")
    torch.cuda.synchronize()
    print(
        f"[cuda] initialized context on local_rank={env.local_rank} "
        f"current_device={torch.cuda.current_device()}",
        flush=True,
    )


def import_nixl_ep() -> None:
    global nixl_ep
    nixl_ep = importlib.import_module("nixl_ep")


def store_key_prefix(args: argparse.Namespace) -> str:
    namespace = args.metadata_namespace.strip("/")
    generation = args.metadata_generation.strip("/")
    if not namespace:
        raise ValueError("--metadata-namespace must not be empty")
    if not generation:
        raise ValueError("--metadata-generation must not be empty")
    return f"checkpoint_preserve_va/{namespace}/{generation}"


def metadata_prefix(args: argparse.Namespace, phase: str) -> str:
    return f"{store_key_prefix(args)}/NIXL_EP/{phase}"


def store_barrier(
    store: dist.TCPStore,
    env: RankEnv,
    key_prefix: str,
    name: str,
    timeout_sec: float,
) -> None:
    prefix = f"{key_prefix}/{name}"
    store.set(f"{prefix}/{env.rank}", str(time.time()).encode())
    store.wait(
        [f"{prefix}/{rank}" for rank in range(env.world_size)],
        timedelta(seconds=timeout_sec),
    )


def gather_store_values(
    store: dist.TCPStore,
    env: RankEnv,
    key_prefix: str,
    name: str,
    value: str,
    timeout_sec: float,
) -> list[str]:
    prefix = f"{key_prefix}/{name}"
    store.set(f"{prefix}/{env.rank}", value.encode())
    keys = [f"{prefix}/{rank}" for rank in range(env.world_size)]
    store.wait(keys, timedelta(seconds=timeout_sec))
    return [raw.decode() for raw in store.multi_get(keys)]


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def touch_sentinel(path: Path) -> None:
    write_text_atomic(path, f"{time.time()}\n")


def cleanup_snapshot_control_dir(control_dir: Path) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    for name in (READY_FOR_CHECKPOINT, SNAPSHOT_COMPLETE, RESTORE_COMPLETE):
        try:
            (control_dir / name).unlink()
        except FileNotFoundError:
            pass


def wait_for_snapshot_control_event(
    control_dir: Path,
    timeout_sec: float,
) -> str:
    snapshot_complete = control_dir / SNAPSHOT_COMPLETE
    restore_complete = control_dir / RESTORE_COMPLETE
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        if snapshot_complete.exists():
            print(
                f"[snapshot] observed {SNAPSHOT_COMPLETE}: {snapshot_complete}",
                flush=True,
            )
            return "checkpoint"
        if restore_complete.exists():
            print(
                f"[snapshot] observed {RESTORE_COMPLETE}: {restore_complete}",
                flush=True,
            )
            return "restore"
        time.sleep(0.1)

    raise TimeoutError(
        "timed out waiting for Dynamo snapshot-control sentinel "
        f"({SNAPSHOT_COMPLETE} or {RESTORE_COMPLETE}) in {control_dir}"
    )


def make_inputs(
    env: RankEnv,
    num_tokens: int,
    hidden_dim: int,
    num_experts_per_rank: int,
    num_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = env.world_size * num_experts_per_rank
    if num_topk != 1:
        raise ValueError("This focused reproducer currently expects --num-topk 1")
    if hidden_dim not in (2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192):
        raise ValueError(f"Unsupported NIXL EP hidden dimension: {hidden_dim}")
    if hidden_dim % 128 != 0:
        raise ValueError("--hidden-dim must be a multiple of 128")
    if (env.world_size * num_tokens) % 4 != 0:
        raise ValueError(
            "--num-tokens * WORLD_SIZE must be divisible by 4 for NIXL EP"
        )

    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    x = torch.full(
        (num_tokens, hidden_dim),
        float(env.rank + 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    x[:, -1] = (token_ids + 1).to(torch.bfloat16)

    owner_rank = token_ids % env.world_size
    expert_offset = (token_ids // env.world_size) % num_experts_per_rank
    topk_idx = (owner_rank * num_experts_per_rank + expert_offset).view(
        num_tokens, num_topk
    )
    topk_idx = topk_idx.to(nixl_ep.topk_idx_t).contiguous()
    topk_weights = torch.ones(
        (num_tokens, num_topk), device="cuda", dtype=torch.float32
    )
    expected = x.clone()

    assert int(topk_idx.max().item()) < num_experts
    return x, topk_idx, topk_weights, expected


def run_iteration(
    buffer: nixl_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    use_recv_hooks: bool = False,
) -> IterationState:
    recv_x, recv_count, handle, _event, recv_hook = buffer.dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=False,
        async_finish=False,
        return_recv_hook=use_recv_hooks,
    )
    if recv_hook is not None:
        recv_hook()
    combined_x, _event, recv_hook = buffer.combine(
        recv_x,
        topk_idx,
        topk_weights,
        handle,
        async_finish=False,
        return_recv_hook=use_recv_hooks,
    )
    if recv_hook is not None:
        recv_hook()
    return IterationState(
        recv_x=recv_x,
        recv_count=recv_count,
        handle=handle,
        combined_x=combined_x,
    )


def assert_expected(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    torch.cuda.synchronize()
    if not torch.equal(actual, expected):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{label} mismatch: max_abs_diff={max_diff}")


def warmup_for_capture(
    args: argparse.Namespace,
    env: RankEnv,
    store: dist.TCPStore,
    key_prefix: str,
    buffer: nixl_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    store_barrier(
        store,
        env,
        key_prefix,
        "capture_warmup_start",
        args.barrier_timeout_sec,
    )
    with torch.cuda.stream(stream):
        for _ in range(args.capture_warmups):
            state = run_iteration(
                buffer,
                x,
                topk_idx,
                topk_weights,
                args.num_tokens,
                num_experts,
                use_recv_hooks=True,
            )
            state.combined_x.record_stream(stream)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    store_barrier(
        store,
        env,
        key_prefix,
        "capture_warmup_done",
        args.barrier_timeout_sec,
    )


def capture_iteration(
    args: argparse.Namespace,
    env: RankEnv,
    store: dist.TCPStore,
    key_prefix: str,
    buffer: nixl_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> CapturedIteration:
    warmup_for_capture(
        args,
        env,
        store,
        key_prefix,
        buffer,
        x,
        topk_idx,
        topk_weights,
        num_experts,
    )
    graph = torch.cuda.CUDAGraph()
    store_barrier(store, env, key_prefix, "capture_start", args.barrier_timeout_sec)
    with torch.cuda.graph(graph):
        state = run_iteration(
            buffer,
            x,
            topk_idx,
            topk_weights,
            args.num_tokens,
            num_experts,
            use_recv_hooks=True,
        )
    torch.cuda.synchronize()
    return CapturedIteration(graph=graph, state=state)


def write_pause_hook(
    args: argparse.Namespace,
    env: RankEnv,
    addresses: dict[str, int],
) -> None:
    if args.hook_dir is None:
        return
    args.hook_dir.mkdir(parents=True, exist_ok=True)
    pause_file = args.hook_dir / f"rank_{env.rank}.paused.json"
    pause_file.write_text(
        json.dumps(
            {
                "rank": env.rank,
                "world_size": env.world_size,
                "addresses": addresses,
            },
            indent=2,
            sort_keys=True,
        )
    )


def wait_for_external_resume(args: argparse.Namespace, env: RankEnv) -> None:
    if args.pause_seconds > 0:
        time.sleep(args.pause_seconds)

    if not args.wait_for_resume_file:
        return
    if args.hook_dir is None:
        raise ValueError("--wait-for-resume-file requires --hook-dir")

    resume_file = args.hook_dir / f"rank_{env.rank}.resume"
    print(
        f"[rank {env.rank}] waiting for external resume file {resume_file}",
        flush=True,
    )
    while not resume_file.exists():
        time.sleep(0.5)


def main() -> None:
    args = parse_args()
    if env_flag(LOCAL_SPAWN_CHILD_ENV) and args.spawn_local_ranks:
        raise ValueError(
            f"{LOCAL_SPAWN_CHILD_ENV}=1 child received recursive "
            "--spawn-local-ranks"
        )
    if args.spawn_local_ranks:
        run_spawn_local_ranks(args)
        return

    env = read_rank_env()
    validate_runtime_topology(args, env)
    configure_ucx_transport(args)

    if env.world_size < 2:
        raise ValueError("Real multi-rank validation requires WORLD_SIZE >= 2")
    if args.snapshot_control_dir is not None and not args.external_store:
        raise ValueError(
            "--snapshot-control-dir requires --external-store for multi-node "
            "checkpoint/restore so restored ranks do not depend on a "
            "checkpointed rank-local TCPStore server"
        )
    if args.metadata_generation is None:
        if args.external_store:
            raise ValueError(
                "--external-store requires --metadata-generation or "
                "NIXL_EP_METADATA_GENERATION to avoid stale TCPStore metadata"
            )
        args.metadata_generation = "inprocess"
    else:
        args.metadata_generation = args.metadata_generation.strip()
        if not args.metadata_generation:
            raise ValueError("--metadata-generation must not be empty")
        print(
            f"[store] metadata namespace={args.metadata_namespace} "
            f"generation={args.metadata_generation}",
            flush=True,
        )

    coord_prefix = f"{store_key_prefix(args)}/coord"

    if args.ucx_gda_auto_device:
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
        initialize_cuda_runtime(env)
        configure_ucx_gda_auto_device(args, env)
        import_nixl_ep()
    else:
        import_nixl_ep()
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
        torch.cuda.set_device(env.local_rank)

    store = create_store(args, env)
    store_barrier(store, env, coord_prefix, "store_ready", args.barrier_timeout_sec)

    num_experts = env.world_size * args.num_experts_per_rank
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens,
        args.hidden_dim,
        env.world_size,
        num_experts,
    )
    buffer = nixl_ep.Buffer(
        rank=env.rank,
        explicitly_destroy=True,
        tcp_store_group=store,
        tcp_store_metadata_prefix=metadata_prefix(args, "initial"),
        timeout_ms=args.timeout_ms,
    )
    try:
        buffer.update_memory_buffers(
            num_ranks=env.world_size,
            num_experts_per_rank=args.num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )
        all_ranks = list(range(env.world_size))
        buffer.connect_ranks(all_ranks)
        store_barrier(store, env, coord_prefix, "connected", args.barrier_timeout_sec)

        x, topk_idx, topk_weights, expected = make_inputs(
            env,
            args.num_tokens,
            args.hidden_dim,
            args.num_experts_per_rank,
            args.num_topk,
        )

        eager_state = run_iteration(
            buffer,
            x,
            topk_idx,
            topk_weights,
            args.num_tokens,
            num_experts,
        )
        assert_expected("eager pre-pause", eager_state.combined_x, expected)
        store_barrier(
            store,
            env,
            coord_prefix,
            "eager_pre_pause_done",
            args.barrier_timeout_sec,
        )

        captured: CapturedIteration | None = None
        graph_capture_error: str | None = None
        if not args.skip_cuda_graph:
            try:
                captured = capture_iteration(
                    args,
                    env,
                    store,
                    coord_prefix,
                    buffer,
                    x,
                    topk_idx,
                    topk_weights,
                    num_experts,
                )
            except Exception as exc:
                graph_capture_error = repr(exc)

            statuses = gather_store_values(
                store,
                env,
                coord_prefix,
                "graph_capture_status",
                "ok" if graph_capture_error is None else graph_capture_error,
                args.barrier_timeout_sec,
            )
            graph_errors = [status for status in statuses if status != "ok"]
            if graph_errors:
                graph_capture_error = graph_capture_error or graph_errors[0]
                captured = None
                if not args.allow_eager_fallback:
                    raise RuntimeError(
                        "CUDA graph capture failed on at least one rank: "
                        f"{graph_errors}"
                    )
                print(
                    f"[rank {env.rank}] CUDA graph capture failed on at least "
                    "one rank; continuing eager-only because "
                    f"--allow-eager-fallback was set: {graph_errors}",
                    flush=True,
                )
            else:
                store_barrier(
                    store,
                    env,
                    coord_prefix,
                    "graph_pre_pause_replay_start",
                    args.barrier_timeout_sec,
                )
                assert captured is not None
                captured.graph.replay()
                assert_expected(
                    "cuda graph pre-pause replay",
                    captured.state.combined_x,
                    expected,
                )
                store_barrier(
                    store,
                    env,
                    coord_prefix,
                    "graph_pre_pause_replay_done",
                    args.barrier_timeout_sec,
                )

        addresses_before_pause = buffer.get_graph_visible_addresses()
        pause_snapshot = buffer.checkpoint_pause_preserve_va()
        if pause_snapshot != addresses_before_pause:
            raise AssertionError(
                "checkpoint_pause_preserve_va returned a different address "
                "snapshot than get_graph_visible_addresses()"
            )
        write_pause_hook(args, env, pause_snapshot)
        store_barrier(store, env, coord_prefix, "paused", args.barrier_timeout_sec)

        if env.rank == 0:
            print(
                "[rank 0] all ranks called checkpoint_pause_preserve_va(); "
                "external CRIU/Dynamo checkpoint can be taken now",
                flush=True,
            )

        if args.snapshot_control_dir is not None:
            cleanup_snapshot_control_dir(args.snapshot_control_dir)
            store_barrier(
                store,
                env,
                coord_prefix,
                "snapshot_control_cleaned",
                args.barrier_timeout_sec,
            )
            buffer.set_tcp_store_group(None)
            del store
            gc.collect()
            touch_sentinel(args.snapshot_control_dir / READY_FOR_CHECKPOINT)
            print(
                "[snapshot] wrote "
                f"{args.snapshot_control_dir / READY_FOR_CHECKPOINT}",
                flush=True,
            )
            snapshot_event = wait_for_snapshot_control_event(
                args.snapshot_control_dir,
                args.snapshot_timeout_sec,
            )
            if snapshot_event == "checkpoint":
                cleanup_snapshot_control_dir(args.snapshot_control_dir)
                print(
                    "[snapshot] checkpoint completed in original process; "
                    "exiting without NIXL EP resume",
                    flush=True,
                )
                return

            cleanup_snapshot_control_dir(args.snapshot_control_dir)
            store = create_store(args, env)
            buffer.set_tcp_store_group(
                store,
                tcp_store_metadata_prefix=metadata_prefix(args, "resume"),
            )
            store_barrier(
                store,
                env,
                coord_prefix,
                "restored_store_ready",
                args.barrier_timeout_sec,
            )
        else:
            wait_for_external_resume(args, env)
            store_barrier(
                store,
                env,
                coord_prefix,
                "resume_start",
                args.barrier_timeout_sec,
            )
            buffer.set_tcp_store_group(
                store,
                tcp_store_metadata_prefix=metadata_prefix(args, "resume"),
            )

        buffer.checkpoint_resume_preserve_va(
            all_ranks,
            activate=False,
            expected_addresses=pause_snapshot,
        )
        for rank in all_ranks:
            buffer.update_mask_buffer(rank, mask=False)
        if not buffer.validate_graph_visible_addresses(pause_snapshot):
            raise AssertionError("graph-visible CUDA VAs changed after resume")
        store_barrier(store, env, coord_prefix, "resumed", args.barrier_timeout_sec)

        if captured is not None:
            store_barrier(
                store,
                env,
                coord_prefix,
                "graph_post_resume_replay_start",
                args.barrier_timeout_sec,
            )
            captured.graph.replay()
            assert_expected(
                "cuda graph post-resume replay without recapture",
                captured.state.combined_x,
                expected,
            )
            validation_mode = "cuda_graph_replay"
        else:
            store_barrier(
                store,
                env,
                coord_prefix,
                "eager_post_resume_start",
                args.barrier_timeout_sec,
            )
            post_resume_state = run_iteration(
                buffer,
                x,
                topk_idx,
                topk_weights,
                args.num_tokens,
                num_experts,
            )
            assert_expected(
                "eager post-resume",
                post_resume_state.combined_x,
                expected,
            )
            validation_mode = "eager"

        store_barrier(
            store,
            env,
            coord_prefix,
            "post_resume_done",
            args.barrier_timeout_sec,
        )
        summary = {
            "rank": env.rank,
            "world_size": env.world_size,
            "num_rdma_bytes": num_rdma_bytes,
            "validation_mode": validation_mode,
            "graph_capture_error": graph_capture_error,
            "addresses": pause_snapshot,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        buffer.destroy()


if __name__ == "__main__":
    main()
