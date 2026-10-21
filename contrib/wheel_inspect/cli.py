# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line interface: inspect, fetch, resolve-reference, publish-report, release-run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from wheel_inspect import report
from wheel_inspect.checks import load_side, run_checks
from wheel_inspect.inventory import InventoryError
from wheel_inspect.wheelset import WheelSetError

EXIT_OK, EXIT_USAGE, EXIT_NO_REPORT = 0, 1, 2


def _fail(message: str) -> int:
    print(f"wheel_inspect: {message}", file=sys.stderr)
    return EXIT_NO_REPORT


def _inspect_dirs(
    candidate: Path,
    reference: Path,
    out_dir: Path,
    candidate_manifest: Optional[Path] = None,
    reference_manifest: Optional[Path] = None,
    candidate_label: Optional[str] = None,
    reference_label: Optional[str] = None,
    cuobjdump: Optional[str] = None,
) -> int:
    try:
        cand = load_side(candidate, candidate_manifest, candidate_label)
        ref = load_side(reference, reference_manifest, reference_label)
    except (WheelSetError, InventoryError, OSError) as exc:
        return _fail(str(exc))
    findings = run_checks(cand, ref, cuobjdump)
    doc = report.build_document(cand, ref, findings, report.tool_version())
    report.write_reports(doc, out_dir)
    print(report.results_table(doc))
    print(f"\n{report.summary_line(doc)}; reports in {out_dir}")
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace) -> int:
    return _inspect_dirs(
        Path(args.candidate),
        Path(args.reference),
        Path(args.out_dir),
        Path(args.candidate_manifest) if args.candidate_manifest else None,
        Path(args.reference_manifest) if args.reference_manifest else None,
        args.candidate_label,
        args.reference_label,
        args.cuobjdump,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wheel_inspect", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser(
        "inspect", help="compare two local wheel directories and write the report"
    )
    inspect.add_argument(
        "--candidate", required=True, help="directory with the candidate wheel set"
    )
    inspect.add_argument(
        "--reference", required=True, help="directory with the reference wheel set"
    )
    inspect.add_argument("--out-dir", required=True)
    inspect.add_argument(
        "--candidate-manifest",
        help="default: <candidate>/build-manifest.json when present",
    )
    inspect.add_argument("--reference-manifest")
    inspect.add_argument("--candidate-label", help="default: the wheel-set version")
    inspect.add_argument("--reference-label")
    inspect.add_argument(
        "--cuobjdump",
        help="path to cuobjdump; default: from PATH, else the fatbin check is skipped",
    )
    inspect.set_defaults(func=_cmd_inspect)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    return int(args.func(args))
