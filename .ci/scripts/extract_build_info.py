#!/usr/bin/env python3
"""Write <wheel>.build-info.json for one built NIXL wheel, after the build.

Run by .ci/scripts/wheel-nightly/publish-wheel.sh inside the wheel image, so
the same environment that built the wheel answers: ``ucx_info -v`` for the
UCX in the image, ``git rev-parse`` for the NIXL commit, ``uv pip list`` for
the environment auditwheel ran in. The per-venv toolchain comes from the
``uv pip list`` dumps contrib/build-wheel.sh leaves in build-info/
(``pip-py<ver>-torch<X.Y|base>.json``); refs that predate the dump get
``null`` there and nothing else degrades. Torch versions built are read off
the wheel itself (the nixl_ep modules). Every value that cannot be
determined is ``null``; the keys are always present. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1
TOOLCHAIN_PACKAGES = (
    "meson",
    "meson-python",
    "pybind11",
    "patchelf",
    "setuptools",
    "pyyaml",
)
_UCX_FIELDS = {
    "release": re.compile(r"^# Library version: (\S+)", re.M),
    "api_version": re.compile(r"^# API headers version: (\S+)", re.M),
    "configured_with": re.compile(r"^# Configured with: (.*)$", re.M),
}
_UCX_GIT = re.compile(r"Git branch '([^']+)', revision ([0-9a-f]+)")
_EP_MODULE = re.compile(r"^nixl_ep_cu\d+/nixl_ep_cpp_torch(\d)(\d+)\.[^/]+\.so$")
_DUMP = re.compile(r"^pip-py(?P<py>[0-9.]+)-torch(?P<torch>base|[0-9.]+)\.json$")


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
    }
    if text is None:
        return out
    for key, pattern in _UCX_FIELDS.items():
        match = pattern.search(text)
        out[key] = match.group(1).strip() if match else None
    git = _UCX_GIT.search(text)
    if git:
        out["branch"], out["revision"] = git.group(1), git.group(2)
    return out


def run_capture(argv: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pip_list_to_versions(text_or_obj: Any) -> dict[str, str]:
    """``uv pip list --format json`` (a list of {name, version}) -> {normalised name: version}."""
    items = json.loads(text_or_obj) if isinstance(text_or_obj, str) else text_or_obj
    out: dict[str, str] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("name") and item.get("version"):
            out[str(item["name"]).lower().replace("_", "-")] = str(item["version"])
    return out


def load_dumps(directory: Optional[Path]) -> dict[str, dict[str, str]]:
    """torch key ('base' or 'X.Y') -> package versions, from the dumps in ``directory``."""
    dumps: dict[str, dict[str, str]] = {}
    if directory is None or not directory.is_dir():
        return dumps
    for path in sorted(directory.iterdir()):
        match = _DUMP.match(path.name)
        if not match:
            continue
        try:
            dumps[match.group("torch")] = pip_list_to_versions(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
    return dumps


def torch_built_in(wheel: zipfile.ZipFile) -> list[str]:
    """Torch versions whose nixl_ep module is in the wheel, e.g. torch214 -> 2.14."""
    found = set()
    for name in wheel.namelist():
        match = _EP_MODULE.match(name)
        if match:
            found.add(f"{match.group(1)}.{match.group(2)}")
    return sorted(found, key=lambda v: tuple(int(p) for p in v.split(".")))


def build_sidecar(
    *,
    wheel_path: Path,
    python_version: str,
    platform: str,
    cu_tag: str,
    torch_requested: list[str],
    dumps: dict[str, dict[str, str]],
    repair_env: Optional[dict[str, str]],
    ucx_info_text: Optional[str],
    git_commit: Optional[str],
    generated_at: str,
) -> dict[str, Any]:
    identity = parse_wheel_filename(wheel_path.name)
    with zipfile.ZipFile(wheel_path) as zf:
        built_versions = torch_built_in(zf)

    # The main wheel is built in the first torch venv (build-wheel.sh);
    # the plain venv only exists for a torch-less build.
    main_venv = dumps.get(built_versions[0]) if built_versions else dumps.get("base")
    toolchain: dict[str, Optional[str]] = {
        name: (main_venv or {}).get(name) for name in TOOLCHAIN_PACKAGES
    }
    toolchain["auditwheel"] = (repair_env or {}).get("auditwheel")
    toolchain["patchelf_repair"] = (repair_env or {}).get("patchelf")

    built = []
    for version in built_versions:
        venv = dumps.get(version) or {}
        resolved = venv.get("torch")
        built.append(
            {
                "version": version,
                "channel": (
                    ("nightly" if ".dev" in resolved else "stable")
                    if resolved
                    else None
                ),
                "resolved": resolved,
                "toolchain": (
                    {name: venv.get(name) for name in TOOLCHAIN_PACKAGES}
                    if venv
                    else None
                ),
            }
        )
    skipped = [
        {
            "version": v,
            "reason": f"no torch {v} wheel on the stable or nightly {cu_tag} index for Python {python_version}",
        }
        for v in torch_requested
        if v not in built_versions
    ]
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
        "cuda": {"cu_tag": cu_tag},
        "nixl": {"commit": git_commit},
        "ucx": parse_ucx_info(ucx_info_text),
        "torch": {
            "requested": list(torch_requested),
            "built": built,
            "skipped": skipped,
        },
        "toolchain": toolchain,
        "toolchain_dumps": sorted(dumps),
    }


def _csv(value: Optional[str]) -> list[str]:
    return [v for v in (value or "").split(",") if v]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--cu-tag", required=True)
    parser.add_argument("--torch-requested", default="", help="comma-separated")
    parser.add_argument(
        "--pip-dumps", type=Path, help="build-info dir with the per-venv pip list dumps"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("."),
        help="NIXL checkout the wheel was built from",
    )
    args = parser.parse_args(argv)

    repair_text = run_capture(["uv", "pip", "list", "--format", "json"])
    doc = build_sidecar(
        wheel_path=args.wheel,
        python_version=args.python_version,
        platform=args.platform,
        cu_tag=args.cu_tag,
        torch_requested=_csv(args.torch_requested),
        dumps=load_dumps(args.pip_dumps),
        repair_env=pip_list_to_versions(repair_text) if repair_text else None,
        ucx_info_text=run_capture(["ucx_info", "-v"]),
        git_commit=(
            run_capture(["git", "-C", str(args.source_dir), "rev-parse", "HEAD"]) or ""
        ).strip()
        or None,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
