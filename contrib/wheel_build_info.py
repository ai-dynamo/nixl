#!/usr/bin/env python3

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

"""Record what a NIXL wheel was built from.

``venv-toolchain`` writes the versions of the build tools installed in the
interpreter running it (meant to run with a build venv's python before the
venv is destroyed). ``sidecar`` writes ``<wheel>.build-info.json`` next to a
built wheel: the wheel's identity and sha256, the UCX it bundles, the torch
versions built and skipped, and the resolved toolchain. Every value that
cannot be determined is ``null``; the keys are always present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1
TOOLCHAIN_PACKAGES = (
    "meson",
    "meson-python",
    "pybind11",
    "patchelf",
    "setuptools",
    "pyyaml",
    "torch",
)
REPAIR_PACKAGES = ("auditwheel", "patchelf")

_UCX_FIELDS = {
    "release": re.compile(r"^# Library version: (\S+)", re.M),
    "api_version": re.compile(r"^# API headers version: (\S+)", re.M),
    "configured_with": re.compile(r"^# Configured with: (.*)$", re.M),
}
_UCX_GIT = re.compile(r"Git branch '([^']+)', revision ([0-9a-f]+)")
_NVCC_RELEASE = re.compile(r"release (\d+\.\d+)")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def parse_wheel_filename(name: str) -> dict[str, Optional[str]]:
    if not name.endswith(".whl"):
        raise ValueError(f"not a wheel filename: {name}")
    parts = name[: -len(".whl")].split("-")
    if len(parts) == 5:
        package, version, python_tag, abi_tag, platform_tag = parts
        build_tag: Optional[str] = None
    elif len(parts) == 6:
        package, version, build_tag, python_tag, abi_tag, platform_tag = parts
    else:
        raise ValueError(f"not a wheel filename: {name}")
    return {
        "package": package,
        "version": version,
        "build_tag": build_tag,
        "python_tag": python_tag,
        "abi_tag": abi_tag,
        "platform_tag": platform_tag,
    }


def parse_ucx_info(text: Optional[str]) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {
        "release": None,
        "api_version": None,
        "branch": None,
        "revision": None,
        "configured_with": None,
        "raw": None,
    }
    if text is None:
        return out
    out["raw"] = text
    for key, pattern in _UCX_FIELDS.items():
        match = pattern.search(text)
        out[key] = match.group(1).strip() if match else None
    git = _UCX_GIT.search(text)
    if git:
        out["branch"], out["revision"] = git.group(1), git.group(2)
    return out


def parse_nvcc_version(text: Optional[str]) -> Optional[str]:
    match = _NVCC_RELEASE.search(text or "")
    return match.group(1) if match else None


def package_versions(names: Iterable[str]) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(argv: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _load_toolchain(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _packages_without_torch(toolchain: dict[str, Any]) -> dict[str, Optional[str]]:
    packages = toolchain.get("packages") or {}
    return {k: v for k, v in packages.items() if k != "torch"}


def build_sidecar(
    *,
    wheel_path: Path,
    python_version: str,
    platform: str,
    cu_tag: str,
    torch_requested: list[str],
    torch_built: list[tuple[str, Optional[Path]]],
    torch_skipped: list[str],
    skip_reason: Optional[str],
    toolchain_json: Optional[Path],
    ucx_info_text: Optional[str],
    nvcc_text: Optional[str],
    git_commit: Optional[str],
    repair_packages: dict[str, Optional[str]],
    generated_at: str,
) -> dict[str, Any]:
    identity = parse_wheel_filename(wheel_path.name)

    base = _load_toolchain(toolchain_json)
    base_packages = _packages_without_torch(base) if base else {}
    toolchain: dict[str, Optional[str]] = {
        name: base_packages.get(name) for name in TOOLCHAIN_PACKAGES if name != "torch"
    }
    toolchain["auditwheel"] = repair_packages.get("auditwheel")
    toolchain["patchelf_repair"] = repair_packages.get("patchelf")

    built: list[dict[str, Any]] = []
    for version, path in torch_built:
        entry = _load_toolchain(path)
        if entry is None:
            built.append(
                {
                    "version": version,
                    "channel": None,
                    "resolved": None,
                    "toolchain": None,
                }
            )
            continue
        built.append(
            {
                "version": version,
                "channel": entry.get("channel"),
                "resolved": (entry.get("packages") or {}).get("torch"),
                "toolchain": _packages_without_torch(entry),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "wheel": {
            "name": wheel_path.name,
            **identity,
            "sha256": sha256_of(wheel_path),
            "size": wheel_path.stat().st_size,
        },
        "python_version": python_version,
        "platform": platform,
        "cuda": {"cu_tag": cu_tag, "toolkit_version": parse_nvcc_version(nvcc_text)},
        "nixl": {"commit": git_commit},
        "ucx": parse_ucx_info(ucx_info_text),
        "torch": {
            "requested": list(torch_requested),
            "built": built,
            "skipped": [{"version": v, "reason": skip_reason} for v in torch_skipped],
        },
        "toolchain": toolchain,
    }


def _csv(value: Optional[str]) -> list[str]:
    return [item for item in (value or "").split(",") if item]


def _git_commit() -> Optional[str]:
    out = run_capture(["git", "rev-parse", "HEAD"])
    commit = (out or "").strip()
    return commit if _COMMIT.match(commit) else None


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _parse_torch_built(items: list[str]) -> list[tuple[str, Optional[Path]]]:
    built: list[tuple[str, Optional[Path]]] = []
    for item in items:
        version, sep, path = item.partition("=")
        if not version or not sep or not path:
            raise ValueError(f"--torch-built expects VERSION=FILE, got {item!r}")
        built.append((version, Path(path)))
    return built


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    venv = sub.add_parser(
        "venv-toolchain", help="record this interpreter's build tools"
    )
    venv.add_argument("--out", required=True, type=Path)
    venv.add_argument("--channel", help="torch channel this venv was provisioned from")

    side = sub.add_parser("sidecar", help="write <wheel>.build-info.json")
    side.add_argument("--wheel", required=True, type=Path)
    side.add_argument("--out", required=True, type=Path)
    side.add_argument("--python-version", required=True)
    side.add_argument("--platform", required=True)
    side.add_argument("--cu-tag", required=True)
    side.add_argument("--toolchain-json", type=Path)
    side.add_argument("--torch-requested", help="comma-separated")
    side.add_argument(
        "--torch-built", action="append", default=[], metavar="VERSION=FILE"
    )
    side.add_argument("--torch-skipped", help="comma-separated")
    side.add_argument("--skip-reason")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "venv-toolchain":
            _write_json(
                args.out,
                {
                    "schema_version": SCHEMA_VERSION,
                    "channel": args.channel,
                    "packages": package_versions(TOOLCHAIN_PACKAGES),
                },
            )
            return 0
        doc = build_sidecar(
            wheel_path=args.wheel,
            python_version=args.python_version,
            platform=args.platform,
            cu_tag=args.cu_tag,
            torch_requested=_csv(args.torch_requested),
            torch_built=_parse_torch_built(args.torch_built),
            torch_skipped=_csv(args.torch_skipped),
            skip_reason=args.skip_reason,
            toolchain_json=args.toolchain_json,
            ucx_info_text=run_capture(["ucx_info", "-v"]),
            nvcc_text=run_capture(["nvcc", "--version"]),
            git_commit=_git_commit(),
            repair_packages=package_versions(REPAIR_PACKAGES),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _write_json(args.out, doc)
    except (OSError, ValueError) as exc:
        print(f"wheel_build_info: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
