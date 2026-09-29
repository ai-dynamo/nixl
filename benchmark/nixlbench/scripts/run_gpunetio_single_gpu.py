#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch two local GPUNETIO workers on one allocated GPU, with private MPS."""

import argparse
import ipaddress
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


def checked(argv, env=None, stdin=None):
    """Run a bounded setup/control command without a shell."""
    result = subprocess.run(
        argv, env=env, input=stdin, capture_output=True, text=True, timeout=10
    )
    control_error = argv[0] == "nvidia-cuda-mps-control" and re.search(
        r"^\s*(?:error|fatal)\b", result.stdout, re.IGNORECASE | re.MULTILINE
    )
    if result.returncode or control_error:
        raise RuntimeError(
            f"{argv[0]} failed: {(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def worker_command(command, group, interface):
    """Keep topology, backend and validation under launcher control."""
    reserved = {
        "backend",
        "worker_type",
        "runtime_type",
        "benchmark_group",
        "mode",
        "scheme",
        "gpunetio_device_list",
        "gpunetio_oob_list",
        "check_consistency",
        "nocheck_consistency",
        "num_initiator_dev",
        "num_target_dev",
        "config_file",
        "flagfile",
        "fromenv",
        "tryfromenv",
        "use_device_api",
    }
    for argument in command[1:]:
        if (
            argument.startswith("-")
            and argument.lstrip("-").split("=", 1)[0] in reserved
        ):
            raise ValueError(f"Launcher owns this option: {argument}")
    return command + [
        "--backend=GPUNETIO",
        "--worker_type=nixl",
        "--runtime_type=ETCD",
        f"--benchmark_group={group}",
        "--scheme=pairwise",
        "--mode=SG",
        "--num_initiator_dev=1",
        "--num_target_dev=1",
        "--gpunetio_device_list=0",
        f"--gpunetio_oob_list={interface}",
        "--check_consistency=true",
    ]


def mps_clients(env, records):
    """Return client PIDs reported by this private control daemon."""
    servers = checked(["nvidia-cuda-mps-control"], env, "get_server_list\n")
    clients = set()
    for server in servers.split():
        if server.isdecimal():
            output = checked(
                ["nvidia-cuda-mps-control"], env, f"get_client_list {server}\n"
            )
            pids = {int(pid) for pid in output.split() if pid.isdecimal()}
            clients.update(pids)
            if pids:
                records.append({"server": int(server), "clients": sorted(pids)})
    return clients


def stop_workers(workers):
    """Ask only our workers to stop; never force-kill live CUDA work."""
    for worker in workers:
        if worker.poll() is None:
            worker.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 15
    for worker in workers:
        try:
            worker.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return False
    return True


def run(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not shutil.which(command[0]):
        raise ValueError("Supply an executable nixlbench after --")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu):
        raise ValueError("--gpu must be a full GPU UUID from the allocated device")
    if args.timeout <= 0 or args.interfaces[0] == args.interfaces[1]:
        raise ValueError("Use a positive timeout and two distinct OOB interfaces")
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise ValueError(
            "Existing MPS configuration: use your managed launcher instead"
        )
    processes = checked(["ps", "-eo", "stat=,comm="])
    if any(
        not state.startswith("Z") and Path(name).name.startswith("nvidia-cuda-mps")
        for line in processes.splitlines()
        for state, name in [line.strip().split(maxsplit=1)]
    ):
        raise RuntimeError(
            "Existing MPS process visible; use the managed launcher instead"
        )
    addresses = []
    for interface in args.interfaces:
        devices = json.loads(
            checked(["ip", "-j", "-4", "addr", "show", "dev", interface])
        )
        found = [
            entry["local"]
            for device in devices
            for entry in device.get("addr_info", [])
            if entry.get("family") == "inet"
        ]
        if len(found) != 1 or not all(
            "UP" in device.get("flags", []) for device in devices
        ):
            raise ValueError(
                "Each OOB interface must be up with exactly one IPv4 address"
            )
        address = ipaddress.IPv4Address(found[0])
        if address.is_unspecified or address.is_multicast:
            raise ValueError("OOB requires a concrete unicast IPv4 address")
        addresses.append(str(address))
    if addresses[0] == addresses[1]:
        raise ValueError("OOB interfaces resolve to the same IPv4 address")
    group = "single-gpu-" + uuid.uuid4().hex
    commands = [worker_command(command, group, name) for name in args.interfaces]
    actual = checked(
        ["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"]
    )
    if actual.lower() != args.gpu.lower():
        raise ValueError("GPU identity did not resolve uniquely")
    occupants = checked(
        [
            "nvidia-smi",
            "-i",
            args.gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    if occupants:
        raise RuntimeError(
            "Selected GPU already has compute/MPS processes; not taking ownership"
        )
    if args.mps == "auto" and not shutil.which("nvidia-cuda-mps-control"):
        raise RuntimeError("MPS control tool missing; no silent non-MPS fallback")

    out = (
        Path(args.output)
        if args.output
        else Path(tempfile.mkdtemp(prefix="nixlbench-mps-"))
    )
    if args.output:
        out.mkdir(mode=0o700, parents=False, exist_ok=False)
    out = out.resolve()
    pipe, logs = out / "pipe", out / "mps-log"
    pipe.mkdir(mode=0o700)
    logs.mkdir(mode=0o700)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        CUDA_MPS_PIPE_DIRECTORY=str(pipe),
        CUDA_MPS_LOG_DIRECTORY=str(logs),
    )
    # Keep a single UUID visible to daemon AND clients. Never expose other GPUs
    # if MPS is unavailable; UUID selection avoids ambiguous numeric remapping.
    # In off mode the private empty pipe also prevents joining a global daemon.
    workers, files, membership = [], [], []
    started, verified = False, False
    status, failure = "FAIL", None
    print(f"Results: {out}", flush=True)
    print(
        f"GPUNETIO: two local workers share GPU {args.gpu}; MPS={args.mps}. "
        + (
            "Enabling private MPS to avoid cross-process CUDA scheduling delays."
            if args.mps == "auto"
            else "MPS disabled explicitly for the control run."
        ),
        flush=True,
    )
    try:
        if args.mps == "auto":
            started = True
            checked(["nvidia-cuda-mps-control", "-d"], env)
        for index, argv in enumerate(commands):
            log = (out / f"worker{index}.log").open("w")
            files.append(log)
            workers.append(
                subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        expected = {worker.pid for worker in workers}
        deadline = time.monotonic() + args.timeout
        while True:
            if started and not verified:
                verified = expected <= mps_clients(env, membership)
            codes = [worker.poll() for worker in workers]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Worker failed: {codes}")
            if all(code is not None for code in codes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Benchmark timed out")
            time.sleep(0.02)
        if started and not verified:
            raise RuntimeError(
                "Both worker PIDs were not observed in MPS; increase run length"
            )
        for index in range(2):
            output = (out / f"worker{index}.log").read_text()
            if re.search(
                r"Consistency check failed|error CQE|NIXL_ERR|Segmentation fault",
                output,
                re.IGNORECASE,
            ):
                raise RuntimeError(
                    f"Worker {index} reported a validation/transfer error"
                )
        # Initiator rank is assigned by etcd, not by local launch order.
        if not any(
            re.search(
                r"^\s*\d+\s+\d+\s+\d+\.\d+",
                (out / f"worker{i}.log").read_text(),
                re.MULTILINE,
            )
            for i in range(2)
        ):
            raise RuntimeError("No benchmark result row; refusing an empty success")
        status = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        failure = str(error) or "Interrupted"
    finally:
        # Ignore further termination signals only during bounded cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        drained = stop_workers(workers)
        if not drained:
            status = "FAIL"
            failure = f"{failure or ''}; live worker remains; private daemon retained for recovery"
        elif started:
            try:
                checked(["nvidia-cuda-mps-control"], env, "quit\n")
            except Exception as error:
                status, failure = "FAIL", f"MPS cleanup failed: {error}"
        for log in files:
            log.close()
        result = {
            "status": status,
            "error": failure,
            "gpu": args.gpu,
            "mps_requested": args.mps,
            "mps_clients_verified": verified,
            "worker_pids": [worker.pid for worker in workers],
            "exit_codes": [worker.poll() for worker in workers],
            "commands": commands,
            "oob_addresses": addresses,
            "membership": membership,
            "private_pipe": str(pipe),
            "workers_drained": drained,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{status}: {failure or 'both workers exited and validation was enabled'}",
        flush=True,
    )
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", required=True, help="full UUID of an allocated, idle GPU"
    )
    parser.add_argument(
        "--oob-interfaces",
        dest="interfaces",
        nargs=2,
        required=True,
        help="two local interfaces with distinct reachable IPv4 addresses",
    )
    parser.add_argument("--mps", choices=("auto", "off"), default="auto")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--output", help="new private output directory (must not exist)"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args)
    except (Exception, KeyboardInterrupt) as error:
        print(f"FAIL: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
