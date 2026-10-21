#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run fresh, alternating UCX SGL / compact GPUNETIO process pairs.

Configuration is trusted command data. Raw outputs contain machine paths;
share summary.json, not private configuration or raw /proc/self/maps logs.
"""
import argparse
import json
import math
import shlex
import statistics
import subprocess
import time
from pathlib import Path


def remote(endpoint, args):
    return endpoint["shell_prefix"] + [shlex.join(args)]


def samples(case, backend, measured, warmup):
    for role in ("sender", "receiver"):
        log = (case / f"{role}.log").read_text()
        if f"REPLAY_PASS {measured + warmup} {backend}" not in log:
            raise RuntimeError(
                f"missing {role} validation/teardown verdict: {case.name}"
            )
        workload = "WORKLOAD 27 64 576 4 reposts=1 standalone=0 layer_mrs=0 cpu="
        if not any(line.startswith(workload) for line in log.splitlines()):
            raise RuntimeError(f"wrong {role} workload: {case.name}")
        if "got completion with err" in log or "error CQE" in log:
            raise RuntimeError(f"CQ error: {case.name}")
    result = []
    for line in (case / "sender.log").read_text().splitlines():
        if not line.startswith("SAMPLE "):
            continue
        fields = line.split()
        if len(fields) != 8 or int(fields[1]) != len(result):
            raise ValueError(f"malformed sample: {case.name}")
        row = list(map(float, fields[2:]))
        if any(not math.isfinite(x) or x < 0 for x in row):
            raise ValueError("invalid elapsed time")
        if not math.isclose(sum(row[:4]), row[4], rel_tol=1e-4, abs_tol=0.02):
            raise ValueError("phase sum does not match request time")
        result.append(row)
    if len(result) != measured:
        raise ValueError(f"wrong measured count: {case.name}")
    return result


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def check_environment(env):
    allowed_ucx = {"UCX_MODULE_DIR", "UCX_NET_DEVICES", "UCX_IB_GID_INDEX"}
    forbidden = {"LD_PRELOAD", "PD_LAG_POLICY", "PD_REPLAY_LAYER_MRS"} & env.keys()
    forbidden.update(
        key for key in env if key.startswith("UCX_") and key not in allowed_ucx
    )
    if forbidden:
        raise ValueError(
            "campaign rejects environment overrides: " + ", ".join(sorted(forbidden))
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, help="new private evidence directory")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    owners = cfg.get("progress_owners", 4)
    ordered = cfg.get("ordered_notifications", True)
    if type(owners) is not int or not 1 <= owners <= 4 or type(ordered) is not bool:
        raise ValueError(
            "progress_owners must be 1..4; ordered_notifications must be boolean"
        )
    measured, warmup, pairs = (
        cfg.get("iterations", 600),
        cfg.get("warmup", 20),
        cfg.get("pairs", 5),
    )
    if measured < 1 or warmup < 0 or pairs < 1:
        raise ValueError("invalid run counts")
    min_free_mib = cfg.get("min_free_mib", 4096)
    if (
        not isinstance(min_free_mib, int)
        or isinstance(min_free_mib, bool)
        or min_free_mib <= 0
    ):
        raise ValueError("min_free_mib must be a positive integer")
    for role in ("sender", "receiver"):
        check_environment(cfg[role]["env"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.private.json").write_text(json.dumps(cfg, indent=2))
    rows = {arm: [] for arm in ("UCX", "GPUNETIO")}
    per_run = []
    identities = {}
    common_core = None
    for pair in range(pairs):
        order = ("UCX", "GPUNETIO") if pair % 2 == 0 else ("GPUNETIO", "UCX")
        paired = {}
        for backend in order:
            case = args.output / f"r{pair}-{backend.lower()}"
            case.mkdir()
            commands = {}
            for role in ("sender", "receiver"):
                ep = cfg[role]
                gpu = str(ep["gpu"])
                query = [
                    "nvidia-smi",
                    "-i",
                    gpu,
                    "--query-gpu=index,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
                history = []
                for attempt in range(6):
                    before = subprocess.check_output(remote(ep, query), text=True)
                    history.append(before)
                    _, free, util = map(int, before.strip().split(","))
                    if free >= min_free_mib and util <= 5:
                        break
                    # NVML utilization can still cover the previous, exited worker.
                    time.sleep(1)
                (case / f"{role}-preflight-history.csv").write_text("".join(history))
                (case / f"{role}-before.csv").write_text(before)
                if free < min_free_mib or util > 5:
                    raise RuntimeError(f"insufficient idle GPU headroom: {role}")
                identity = subprocess.check_output(
                    remote(
                        ep, ["sha256sum", "--", ep["executable"]] + ep["identity_files"]
                    ),
                    text=True,
                )
                (case / f"{role}-hashes.private.txt").write_text(identity)
                pins = {
                    Path(line.split(maxsplit=1)[1].lstrip("*")).name: line.split()[0]
                    for line in identity.strip().splitlines()
                }
                if role in identities and identities[role] != pins:
                    raise RuntimeError(f"binaries changed during campaign: {role}")
                identities[role] = pins
                if "libnixl.so" not in pins:
                    raise ValueError("identity_files must include libnixl.so")
                if common_core is not None and common_core != pins["libnixl.so"]:
                    raise RuntimeError(
                        "comparison does not use one identical NIXL core"
                    )
                common_core = pins["libnixl.so"]
                env = dict(ep["env"])
                env.update(
                    CUDA_VISIBLE_DEVICES=gpu,
                    CUDA_MODULE_LOADING="EAGER",
                    NIXL_LOG_LEVEL="ERROR",
                    NIXL_UCX_SGL_ENABLE="1",
                    PD_NOTIFICATION_PIPELINE="0",
                    PD_REPLAY_ROWS="64",
                    PD_REPLAY_LAYERS="27",
                    PD_REPLAY_REPOSTS="1",
                    PD_REPLAY_STANDALONE="0",
                    PD_REPLAY_PROGRESS_OWNERS=str(owners),
                    PD_REPLAY_ORDERED_NOTIFICATIONS=str(int(ordered)),
                )
                command = ["env"] + [f"{k}={v}" for k, v in env.items()]
                command += [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=5s",
                    "90s",
                    "taskset",
                    "-c",
                    str(ep["cpu"]),
                    ep["executable"],
                    role,
                    backend,
                    cfg["receiver_ip"],
                    str(cfg["control_port"]),
                    str(ep["oob_port"]),
                    cfg["device"],
                    cfg["interface"],
                    str(cfg["gid_index"]),
                    str(measured),
                    str(warmup),
                ]
                if backend == "GPUNETIO":
                    command.append("strided")
                commands[role] = remote(ep, command)
            (case / "commands.private.json").write_text(json.dumps(commands, indent=2))
            with (case / "receiver.log").open("w") as receiver_log:
                receiver = subprocess.Popen(
                    commands["receiver"], stdout=receiver_log, stderr=subprocess.STDOUT
                )
                try:
                    deadline = time.monotonic() + 45
                    while "\nREADY\n" not in "\n" + (case / "receiver.log").read_text():
                        if receiver.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(f"receiver not ready: {case}")
                        time.sleep(0.2)
                    with (case / "sender.log").open("w") as sender_log:
                        sender = subprocess.Popen(
                            commands["sender"],
                            stdout=sender_log,
                            stderr=subprocess.STDOUT,
                        )
                        sender_code = sender.wait(timeout=100)
                    receiver_code = receiver.wait(timeout=100)
                finally:
                    if receiver.poll() is None:
                        receiver.wait(timeout=100)
            (case / "exit.json").write_text(
                json.dumps({"sender": sender_code, "receiver": receiver_code})
            )
            if sender_code or receiver_code:
                raise RuntimeError(f"worker failure: {case}")
            arm = samples(case, backend, measured, warmup)
            rows[backend].extend(arm)
            paired[backend] = {
                "p50_us": statistics.median(x[4] for x in arm),
                "p99_us": percentile([x[4] for x in arm], 0.99),
            }
            print(
                f"PASS pair={pair} arm={backend} p50={paired[backend]['p50_us']:.3f}us",
                flush=True,
            )
        improvement = 100 * (1 - paired["GPUNETIO"]["p50_us"] / paired["UCX"]["p50_us"])
        per_run.append({"pair": pair, "arms": paired, "p50_reduction_pct": improvement})
    pooled = {}
    names = ("plan", "prepare", "post", "post_to_done", "request", "validated_ack")
    for backend, arm in rows.items():
        pooled[backend] = {
            name: {
                "p50_us": statistics.median(x[i] for x in arm),
                "p99_us": percentile([x[i] for x in arm], 0.99),
                "max_us": max(x[i] for x in arm),
                "gte_1ms": sum(x[i] >= 1000 for x in arm),
            }
            for i, name in enumerate(names)
        }
    gain = statistics.median(x["p50_reduction_pct"] for x in per_run)
    passed = (
        pairs >= 5
        and measured >= 600
        and all(x["p50_reduction_pct"] > 0 for x in per_run)
        and gain >= 5
        and pooled["GPUNETIO"]["request"]["p99_us"]
        <= pooled["UCX"]["request"]["p99_us"]
    )
    summary = {
        "pairs": pairs,
        "measured_per_arm_per_pair": measured,
        "per_run": per_run,
        "pooled": pooled,
        "median_paired_p50_reduction_pct": gain,
        "gate_pass": passed,
        "common_core_sha256": common_core,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"Gate={'PASS' if passed else 'NOT_PASSED'} median paired p50 reduction={gain:.2f}%"
    )


if __name__ == "__main__":
    main()
