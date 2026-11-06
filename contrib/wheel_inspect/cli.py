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

from wheel_inspect import fetch as fetch_mod
from wheel_inspect import report
from wheel_inspect.checks import load_side, run_checks
from wheel_inspect.fetch import FetchError
from wheel_inspect.inventory import InventoryError
from wheel_inspect.wheelset import WheelSetError, load_wheelset

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
    except (WheelSetError, InventoryError, OSError, ValueError) as exc:
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


def default_opener() -> fetch_mod.Opener:
    return fetch_mod.default_opener()


def fetch_source(
    source: str,
    dest: Path,
    opener: Optional[fetch_mod.Opener] = None,
    auth: fetch_mod.Auth = None,
) -> Path:
    return fetch_mod.fetch(source, dest, opener, auth)


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        path = fetch_source(
            args.source, Path(args.dest), default_opener(), fetch_mod.env_auth()
        )
    except FetchError as exc:
        return _fail(str(exc))
    print(path)
    return EXIT_OK


def _cmd_resolve_reference(args: argparse.Namespace) -> int:
    try:
        print(
            fetch_mod.resolve_reference(
                args.candidate_version, default_opener(), args.index_url
            )
        )
    except FetchError as exc:
        return _fail(str(exc))
    return EXIT_OK


def _cmd_publish_report(args: argparse.Namespace) -> int:
    try:
        fetch_mod.publish_report(
            args.folder, Path(args.report_dir), fetch_mod.env_auth(), default_opener()
        )
    except (FetchError, OSError) as exc:
        return _fail(str(exc))
    return EXIT_OK


def _cmd_release_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    opener = default_opener()
    auth = fetch_mod.env_auth()
    try:
        candidate = fetch_source(args.candidate, out_dir / "candidate", opener, auth)
        reference_source = args.reference
        if reference_source == "auto":
            version = load_wheelset(candidate).version
            reference_source = "pypi:" + fetch_mod.resolve_reference(version, opener)
            print(f"reference: {reference_source}")
        reference = fetch_source(reference_source, out_dir / "reference", opener, auth)
    except (FetchError, WheelSetError, OSError) as exc:
        return _fail(str(exc))
    rc = _inspect_dirs(candidate, reference, out_dir, cuobjdump=args.cuobjdump)
    if rc != EXIT_OK:
        return rc
    kind, value = fetch_mod.parse_source(args.candidate)
    if kind == "artifactory" and not args.no_publish:
        doc = report.load_document(out_dir / f"{report.REPORT_BASENAME}.json") or {}
        try:
            fetch_mod.publish_report(value, out_dir, auth, opener)
            doc["publish"] = {"ok": True, "folder": value}
            print(f"report published to {value}")
        except (FetchError, OSError) as exc:
            doc["publish"] = {"ok": False, "folder": value, "error": str(exc)}
            print(f"wheel_inspect: report not published: {exc}", file=sys.stderr)
        report.write_reports(doc, out_dir)
    return EXIT_OK


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

    fetch = sub.add_parser(
        "fetch",
        help="download a wheel set: pypi:<version>, artifactory:<folder URL>, dir:<path>",
    )
    fetch.add_argument("--source", required=True)
    fetch.add_argument("--dest", required=True)
    fetch.set_defaults(func=_cmd_fetch)

    resolve = sub.add_parser(
        "resolve-reference",
        help="print the highest released version on PyPI below the given one",
    )
    resolve.add_argument("--candidate-version", required=True)
    resolve.add_argument("--index-url", default=fetch_mod.PYPI_INDEX)
    resolve.set_defaults(func=_cmd_resolve_reference)

    publish = sub.add_parser(
        "publish-report",
        help="upload wheel-inspection.{json,md,html} into an Artifactory folder",
    )
    publish.add_argument("--folder", required=True)
    publish.add_argument("--report-dir", required=True)
    publish.set_defaults(func=_cmd_publish_report)

    release = sub.add_parser(
        "release-run",
        help="fetch both sides, inspect, and publish the report (the nightly's step)",
    )
    release.add_argument(
        "--candidate",
        required=True,
        help="pypi:<version> | artifactory:<folder URL> | dir:<path>",
    )
    release.add_argument(
        "--reference",
        required=True,
        help="same forms, or 'auto' for the previous PyPI release",
    )
    release.add_argument("--out-dir", required=True)
    release.add_argument("--no-publish", action="store_true")
    release.add_argument("--cuobjdump")
    release.set_defaults(func=_cmd_release_run)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    try:
        return int(args.func(args))
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")
