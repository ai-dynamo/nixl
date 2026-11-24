# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright afdEP-comm contributors (formerly afdEP).
# Adapted from afdEP-comm benchmarks/screen.py at 575d2f9. See NOTICE.
r"""Alternate single-arm benchmark processes and pair them by repeat.

Each repeat launches one fresh distributed job per arm, rotating the order.
The medians from each repeat form one paired comparison against the first arm.

Run from runtime/ep.
    python -m benchmarks.afd.screen --repeats 4 --output /tmp/screen.json -- \
        --payload bf16 --execution eager --hybrid off --tokens 1 32 128 2048
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

# The first arm is the control. A custom arm label can select a backend with "_baseline".
DEFAULT_ARMS: dict[str, dict] = {"nixlep": {}, "afdep": {}, "fastafd": {}}
ENV_KEY = "_env"
# "_tree" runs the arm from another source tree, imported alone (PYTHONSAFEPATH);
# "_digest" is the native source digest every rank of the arm must report.
TREE_KEY, DIGEST_KEY = "_tree", "_digest"
LAUNCH = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nproc-per-node",
]
PROCESS_TIMEOUT_S = 900


def run_arm(name, overrides, repeat, args, passthrough):
    """Run one fresh process; an arm's optional "_env" applies to that process."""
    overrides = dict(overrides)
    env = os.environ | overrides.pop(ENV_KEY, {})
    tree, digest = overrides.pop(TREE_KEY, None), overrides.pop(DIGEST_KEY, None)
    cwd = None
    if tree is not None:
        cwd = str(Path(tree).resolve())
        env |= {"PYTHONPATH": cwd, "PYTHONSAFEPATH": "1"}
    # Resolved: a "_tree" arm runs from another working directory.
    output = args.output.resolve().parent / f"{args.output.stem}-{name}-r{repeat}.json"
    command = [
        *LAUNCH,
        str(args.nproc),
        "-m",
        args.module,
        *passthrough,
        "--arms",
        json.dumps({name: overrides}),
        "--output",
        str(output),
    ]
    log = output.with_suffix(".log")
    with log.open("w") as stream:
        code = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=PROCESS_TIMEOUT_S,
            env=env,
            cwd=cwd,
        ).returncode
    if code:
        raise SystemExit(f"{name} repeat {repeat} failed ({code}); see {log}")
    try:
        result = json.loads(output.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"{name} repeat {repeat} could not read {output}: {exc}"
        ) from exc
    ranks = sorted(result["metadata"], key=lambda rank: rank["rank"])
    if [rank["rank"] for rank in ranks] != list(range(args.nproc)):
        raise SystemExit(
            f"{name} repeat {repeat}: incomplete rank metadata in {output}"
        )
    digests = {rank["native_source_sha256"] for rank in ranks}
    if len(digests) != 1 or (digest is not None and digests != {digest}):
        raise SystemExit(
            f"{name} repeat {repeat} has inconsistent native digests {digests}"
        )
    binaries = {rank["backend_binary_sha256"] for rank in ranks}
    if len(binaries) != 1 or not all(binaries):
        raise SystemExit(
            f"{name} repeat {repeat}: inconsistent binary digest in {output}"
        )
    placement = [
        {
            key: rank[key]
            for key in ("rank", "role", "role_rank", "device", "device_uuid")
        }
        for rank in ranks
    ]
    layouts = {rank["receive_layout"] for rank in ranks if rank["role"] == "ffn"}
    if (
        not layouts
        or not all(layouts)
        or not all(rank["device_uuid"] for rank in placement)
    ):
        raise SystemExit(
            f"{name} repeat {repeat}: missing layout or device UUID in {output}"
        )
    medians = {}
    work = {}
    for cell in result["cells"]:
        tokens = cell["tokens"]
        if tokens in work:
            raise SystemExit(f"{name} repeat {repeat}: duplicate token extent {tokens}")
        value = cell["summary"][name]
        for key in args.metric.split("."):
            value = value[key]
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(
                f"{name} repeat {repeat}: invalid {args.metric} for {tokens} tokens"
            )
        medians[tokens] = value
        work[tokens] = cell["work"]
        if not isinstance(work[tokens], dict) or not work[tokens]:
            raise SystemExit(
                f"{name} repeat {repeat} has no logical work record for {tokens} tokens"
            )
    if not work:
        raise SystemExit(f"{name} repeat {repeat}: no benchmark cells in {output}")
    return {
        "median_us": medians,
        "native_source_sha256": digests.pop(),
        "backend_binary_sha256": binaries.pop(),
        "receive_layout": sorted(layouts),
        "placement": placement,
        "work": work,
        "raw_json": output.name,
    }


def summarize(processes, names):
    """Per token extent: median process value and paired change vs names[0]."""
    control = names[0]
    table = []
    for n in sorted(processes[0]["median_us"][control]):
        row = {"tokens": n}
        for name in names:
            values = [p["median_us"][name][n] for p in processes]
            changes = [
                100 * (p["median_us"][name][n] / p["median_us"][control][n] - 1)
                for p in processes
            ]
            row[name] = {
                "median_us": statistics.median(values),
                "paired_change_pct": changes,
                "geomean_change_pct": 100
                * (
                    math.exp(statistics.fmean(math.log1p(c / 100) for c in changes)) - 1
                ),
                "repeats_faster": sum(c < 0 for c in changes),
            }
        table.append(row)
    return table


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arms", type=json.loads, default=DEFAULT_ARMS)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--module", default="benchmarks.afd.comm", help="per-arm benchmark"
    )
    parser.add_argument(
        "--metric", default="median_us", help="dot path into each arm's cell summary"
    )
    args, passthrough = parser.parse_known_args()
    passthrough = [value for value in passthrough if value != "--"]
    if (
        args.repeats < 1
        or args.nproc < 2
        or not isinstance(args.arms, dict)
        or not args.arms
    ):
        parser.error(
            "repeats must be positive, nproc at least two, and --arms a nonempty JSON object"
        )
    if any(
        not name or Path(name).name != name or not isinstance(overrides, dict)
        for name, overrides in args.arms.items()
    ):
        parser.error(
            "arm names must be file name components and overrides JSON objects"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = list(args.arms)
    processes = []
    reference = None
    backends = {}
    for repeat in range(args.repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        runs = {}
        for name in order:
            run = run_arm(name, args.arms[name], repeat, args, passthrough)
            if reference is None:
                reference = run
            elif run["placement"] != reference["placement"]:
                raise SystemExit(
                    f"{name} repeat {repeat}: device or role placement differs from the reference"
                )
            elif run["work"] != reference["work"]:
                raise SystemExit(
                    f"{name} repeat {repeat}: logical work differs from the reference"
                )
            backend = {
                key: run[key]
                for key in (
                    "native_source_sha256",
                    "backend_binary_sha256",
                    "receive_layout",
                )
            }
            if backends.setdefault(name, backend) != backend:
                raise SystemExit(
                    f"{name} repeat {repeat}: backend build or receive layout changed"
                )
            runs[name] = run
        results = {name: run["median_us"] for name, run in runs.items()}
        processes.append(
            {
                "order": order,
                "median_us": results,
                "raw_json": {name: run["raw_json"] for name, run in runs.items()},
            }
        )
        print(json.dumps({"repeat": repeat} | results), flush=True)
    table = summarize(processes, names)
    args.output.write_text(
        json.dumps(
            {
                "arms": args.arms,
                "passthrough": passthrough,
                "placement": reference["placement"],
                "work": reference["work"],
                "backends": backends,
                "processes": processes,
                "table": table,
                "pod": os.environ.get("HOSTNAME"),
                "python": sys.version,
            },
            indent=1,
        )
        + "\n"
    )
    for row in table:
        print(
            json.dumps(
                {"tokens": row["tokens"]}
                | {
                    name: [
                        round(row[name]["median_us"], 2),
                        round(row[name]["geomean_change_pct"], 2),
                        row[name]["repeats_faster"],
                    ]
                    for name in names
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
