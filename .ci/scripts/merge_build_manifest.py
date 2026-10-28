#!/usr/bin/env python3
"""Merge the nightly wheel build's per-cell fragments into build-manifest.json.

Run by .ci/scripts/wheel-nightly/publish-manifest.sh in a plain Python image
at the end of nixl-ci-build-wheel-nightly. Reads the fragment directories the
cells deposited (cell.env, wheel_build_ids.env, build_options.env and the
wheel's build-info sidecar from extract_build_info.py), cross-checks them against each other and against
the Artifactory folder listing, writes the manifest and uploads it next to
the wheels. The manifest file is always written, complete or not; it is
uploaded, and the exit status is 0, only when it is complete (--dry-run
skips the upload).

The fragments come off a PVC other jobs can write, so every name is validated
before it is used and nothing is ever handed to a shell.

Env: EXPECTED_WHEEL_CELLS, WHEEL_REPO_URL, WHEEL_REPO_API, WHEEL_REPO_UI,
JOB_NAME, BUILD_NUMBER, BUILD_URL, NIXL_VERSION, ci_refspec, PUBLISH_DIR,
ARTIFACTORY_USER, ARTIFACTORY_TOKEN. Stdlib only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

CELL_KEY = re.compile(r"^[A-Za-z0-9._-]+$")
FILE_NAME = re.compile(r"^[A-Za-z0-9._+-]+$")
# Publish's allowlist plus the folder component.
FOLDER = re.compile(
    r"^(verification/g[0-9a-f]{8}\.ucx[0-9a-f]{8}|release/[A-Za-z0-9][A-Za-z0-9._-]*/[0-9]+)$"
)
MANIFEST_NAME = "build-manifest.json"
HTTP_ATTEMPTS = 3


class ManifestError(Exception):
    pass


# --- fragments -----------------------------------------------------------------


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE or KEY="VALUE" lines; blank and # lines are skipped."""
    out: dict[str, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key:
            continue
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        out[key] = value
    return out


def _read_regular(path: Path) -> Optional[str]:
    if path.is_symlink() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def read_cells(fragments: Path) -> list[dict[str, Any]]:
    cells = []
    for entry in sorted(fragments.iterdir(), key=lambda p: p.name):
        cell: dict[str, Any] = {
            "key": entry.name,
            "cell": {},
            "ids": {},
            "opts": {},
            "sidecar": None,
            "sidecars": 0,
            "sidecar_error": None,
            "valid": bool(CELL_KEY.match(entry.name))
            and entry.is_dir()
            and not entry.is_symlink(),
        }
        cells.append(cell)
        if not cell["valid"]:
            continue
        for name, field in (
            ("cell.env", "cell"),
            ("wheel_build_ids.env", "ids"),
            ("build_options.env", "opts"),
        ):
            text = _read_regular(entry / name)
            if text is not None:
                cell[field] = parse_env(text)
        sidecars = sorted(
            p
            for p in entry.glob("*.build-info.json")
            if not p.is_symlink() and p.is_file()
        )
        cell["sidecars"] = len(sidecars)
        if sidecars:
            try:
                loaded = json.loads(sidecars[0].read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("not a JSON object")
                cell["sidecar"] = loaded
            except ValueError as exc:
                cell["sidecar_error"] = f"{sidecars[0].name}: {exc}"
    return cells


# --- Artifactory ---------------------------------------------------------------


def _auth_header() -> dict[str, str]:
    user, token = os.environ.get("ARTIFACTORY_USER"), os.environ.get(
        "ARTIFACTORY_TOKEN"
    )
    if not user or not token:
        raise ManifestError("ARTIFACTORY_USER / ARTIFACTORY_TOKEN not set")
    return {
        "Authorization": "Basic "
        + base64.b64encode(f"{user}:{token}".encode()).decode()
    }


def _http(method: str, url: str, data: Optional[bytes] = None) -> tuple[int, bytes]:
    """Returns (status, body); retries connection errors and 5xx like curl --retry."""
    last = ""
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        req = urllib.request.Request(
            url, data=data, method=method, headers=_auth_header()
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return exc.code, b""
            last = f"{exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            last = str(exc)
        if attempt < HTTP_ATTEMPTS:
            time.sleep(2 * attempt)
    raise ManifestError(f"{method} {url} failed: {last}")


def _get_json(url: str) -> Any:
    status, body = _http("GET", url)
    if status != 200:
        raise ManifestError(f"GET {url} returned {status}")
    try:
        return json.loads(body)
    except ValueError as exc:
        raise ManifestError(f"GET {url}: not JSON") from exc


def artifactory_files(folder: str) -> dict[str, dict[str, Any]]:
    """name -> {sha256, size} for every file in the folder; only plain names are ever put in a URL."""
    api = os.environ["WHEEL_REPO_API"]
    listing = _get_json(f"{api}/{folder}?list&deep=0&listFolders=0")
    files: dict[str, dict[str, Any]] = {}
    for item in listing.get("files") or []:
        name = str(item.get("uri", "")).lstrip("/")
        files[name] = {"sha256": item.get("sha2") or None, "size": item.get("size")}
    for name, info in files.items():
        if not info["sha256"] and FILE_NAME.match(name):
            one = _get_json(f"{api}/{folder}/{name}")
            info["sha256"] = (one.get("checksums") or {}).get("sha256")
    return files


def upload(folder: str, text: str) -> str:
    url = f"{os.environ['WHEEL_REPO_URL']}/{folder}/{MANIFEST_NAME}"
    status, _ = _http("PUT", url, text.encode())
    if status not in (200, 201):
        raise ManifestError(f"PUT {MANIFEST_NAME} returned {status}")
    return f"{os.environ.get('WHEEL_REPO_UI', os.environ['WHEEL_REPO_URL'])}/{folder}/{MANIFEST_NAME}"


# --- assembly ------------------------------------------------------------------


def _tokens(text: Optional[str]) -> list[str]:
    return (text or "").split()


def assemble(
    all_cells: list[dict[str, Any]], files: dict[str, dict[str, Any]], errors: list[str]
) -> tuple[Optional[str], dict[str, Any]]:
    """Validates the fragments and builds the manifest document; every problem goes onto errors."""
    for c in all_cells:
        if not c["valid"]:
            errors.append(f"staging entry '{c['key']}' is not a cell key - ignored")
    cells = [c for c in all_cells if c["valid"]]
    expected = int(os.environ.get("EXPECTED_WHEEL_CELLS") or 20)
    if len(cells) != expected:
        errors.append(f"expected {expected} cell fragments, found {len(cells)}")

    def agree(
        what: str, get: Callable[[dict[str, Any]], Any], group: Optional[list] = None
    ) -> Optional[str]:
        """A value every cell must agree on; None when absent or in dispute."""
        seen: dict[str, str] = {}
        for c in cells if group is None else group:
            v = get(c)
            if v is not None and v != "":
                seen[str(v)] = c["key"]
        if len(seen) > 1:
            errors.append(f"{what} differs across cells: {seen}")
        return next(iter(seen)) if len(seen) == 1 else None

    def per_cuda(
        what: str, get: Callable[[dict[str, Any]], Any]
    ) -> dict[str, Optional[str]]:
        return {
            cu: agree(
                f"{what} (cu{cu})",
                get,
                [c for c in cells if c["cell"].get("CUDA") == cu],
            )
            for cu in ("12", "13")
        }

    def side(c: dict[str, Any], *path: str) -> Any:
        node: Any = c["sidecar"]
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
        return node

    for c in cells:
        if not c["cell"].get("WHEELS") or not c["cell"].get("PUBLISH_FOLDER"):
            errors.append(f"cell {c['key']}: cell.env incomplete")
        if c["sidecars"] != 1:
            errors.append(
                f"cell {c['key']}: expected 1 build-info sidecar, found {c['sidecars']}"
            )
        if c["sidecar_error"]:
            errors.append(f"cell {c['key']}: sidecar unreadable ({c['sidecar_error']})")
        if c["sidecar"] and side(c, "wheel", "name") not in _tokens(
            c["cell"].get("WHEELS")
        ):
            errors.append(
                f"cell {c['key']}: sidecar describes {side(c, 'wheel', 'name')}, not one of its wheels"
            )

    folder = agree("PUBLISH_FOLDER", lambda c: c["cell"].get("PUBLISH_FOLDER"))
    if not folder or not FOLDER.match(folder):
        errors.append(f"publish folder '{folder}' is not an allowed path")
    nixl_sha = agree("NIXL_SHA", lambda c: c["ids"].get("NIXL_SHA"))
    nixl_commit = agree("NIXL_COMMIT", lambda c: c["ids"].get("NIXL_COMMIT"))
    ucx_repo = agree("UCX_REPO", lambda c: c["opts"].get("UCX_REPO"))
    ucx_ref = agree("UCX_REF", lambda c: c["opts"].get("UCX_REF"))
    ucx_sha = agree("UCX_SHA", lambda c: c["opts"].get("UCX_SHA"))
    spcx_on = agree(
        "BUILD_UCX_SPCX_PLUGIN", lambda c: c["opts"].get("BUILD_UCX_SPCX_PLUGIN")
    )
    spcx_ref = agree(
        "UCX_SPCX_PLUGIN_REF", lambda c: c["cell"].get("UCX_SPCX_PLUGIN_REF")
    )
    spcx_commit = agree(
        "UCX_SPCX_PLUGIN_COMMIT", lambda c: c["cell"].get("UCX_SPCX_PLUGIN_COMMIT")
    )
    inf_on = agree("BUILD_INFINIA", lambda c: c["opts"].get("BUILD_INFINIA"))
    inf_image = agree(
        "INFINIA_LIBS_IMAGE", lambda c: c["opts"].get("INFINIA_LIBS_IMAGE")
    )
    inf_digest = agree(
        "INFINIA_LIBS_DIGEST", lambda c: c["opts"].get("INFINIA_LIBS_DIGEST")
    )
    ucx_release = agree("UCX release (ucx_info)", lambda c: side(c, "ucx", "release"))
    ucx_branch = agree("UCX branch (ucx_info)", lambda c: side(c, "ucx", "branch"))
    ucx_revision = agree(
        "UCX revision (ucx_info)", lambda c: side(c, "ucx", "revision")
    )
    image_commit = agree("NIXL commit (image)", lambda c: side(c, "nixl", "commit"))
    base_image = per_cuda("BASE_IMAGE", lambda c: c["opts"].get("BASE_IMAGE"))
    base_tag = per_cuda("BASE_IMAGE_TAG", lambda c: c["opts"].get("BASE_IMAGE_TAG"))
    cuda_version = per_cuda("CUDA_VERSION", lambda c: c["opts"].get("CUDA_VERSION"))
    torch_requested = per_cuda(
        "TORCH_VERSIONS_REQUESTED", lambda c: c["cell"].get("TORCH_VERSIONS_REQUESTED")
    )

    # Declared (build options) vs measured (ucx_info in the image, git in the image).
    # A tag or sha UCX_REF is checked out detached, so ucx_info reports 'HEAD'; the revision check covers it.
    if ucx_branch and ucx_branch != "HEAD" and ucx_ref and ucx_branch != ucx_ref:
        errors.append(
            f"UCX in the image is branch '{ucx_branch}' but the build declared '{ucx_ref}'"
        )
    if (
        ucx_revision
        and ucx_sha
        and not (ucx_sha.startswith(ucx_revision) or ucx_revision.startswith(ucx_sha))
    ):
        errors.append(
            f"UCX revision in the image ({ucx_revision}) does not match the declared commit ({ucx_sha})"
        )
    if image_commit and nixl_sha and not image_commit.startswith(nixl_sha):
        errors.append(
            f"NIXL commit in the image ({image_commit}) does not start with NIXL_SHA ({nixl_sha})"
        )

    # One binary wheel per cell, one meta wheel for the build (uploaded by the cu12 cells).
    binary_names: list[str] = []
    meta_cells: dict[str, list[str]] = {}
    for c in cells:
        for name in _tokens(c["cell"].get("WHEELS")):
            if name.startswith("nixl-"):
                meta_cells.setdefault(name, []).append(c["key"])
            elif name in binary_names:
                errors.append(
                    f"binary wheel {name} was published by more than one cell"
                )
            else:
                binary_names.append(name)
    if len(binary_names) != expected:
        errors.append(f"expected {expected} binary wheels, found {len(binary_names)}")
    if len(meta_cells) != 1:
        errors.append(f"expected exactly one meta wheel, found {sorted(meta_cells)}")
    all_names = binary_names + list(meta_cells)
    for name in all_names:
        if name not in files:
            errors.append(f"{name} is not in Artifactory folder {folder}")
    unexpected = sorted(n for n in files if n not in all_names and n != MANIFEST_NAME)

    wheels: list[dict[str, Any]] = []
    built_all: list[str] = []
    skipped_all: list[dict[str, Any]] = []
    tool_sets: dict[str, list[str]] = {}
    for c in cells:
        s = c["sidecar"]
        binaries = [
            n for n in _tokens(c["cell"].get("WHEELS")) if not n.startswith("nixl-")
        ]
        wheel_name = (
            side(c, "wheel", "name") if s else (binaries[0] if binaries else None)
        )
        art = files.get(wheel_name or "") or {}
        if not s:
            build_info = "missing"
        elif side(c, "toolchain_dumps"):
            build_info = "available"
        else:
            build_info = (
                "available; toolchain not recorded (built ref predates the pip dump)"
            )
        entry: dict[str, Any] = {
            "name": wheel_name,
            "kind": "binary",
            "sha256": art.get("sha256"),
            "size": art.get("size"),
            "local_sha256_matches": (
                (side(c, "wheel", "sha256") == art.get("sha256")) if s else None
            ),
            "cuda": c["cell"].get("CUDA"),
            "python": c["cell"].get("PYTHON_VERSION"),
            "arch": c["cell"].get("ARCH"),
            "platform": side(c, "wheel", "platform_tag"),
            "wheel_base_image": c["cell"].get("WHEEL_BASE_IMAGE"),
            "cell": c["key"],
            "build_info": build_info,
            "torch": None,
            "toolchain": None,
            "ucx": None,
        }
        if s:
            if art.get("sha256") and side(c, "wheel", "sha256") != art.get("sha256"):
                errors.append(
                    f"{wheel_name}: sha256 in Artifactory differs from the file this build produced"
                )
            built = side(c, "torch", "built") or []
            skipped = side(c, "torch", "skipped") or []
            entry["torch"] = {
                "built": [
                    {
                        "version": b.get("version"),
                        "channel": b.get("channel"),
                        "resolved": b.get("resolved"),
                    }
                    for b in built
                ],
                "skipped": skipped,
            }
            entry["toolchain"] = side(c, "toolchain")
            entry["ucx"] = {
                "release": side(c, "ucx", "release"),
                "branch": side(c, "ucx", "branch"),
                "revision": side(c, "ucx", "revision"),
            }
            built_all.extend(str(b.get("version")) for b in built)
            skipped_all.extend(
                {
                    "version": k.get("version"),
                    "python": c["cell"].get("PYTHON_VERSION"),
                    "cuda": c["cell"].get("CUDA"),
                    "reason": k.get("reason"),
                }
                for k in skipped
            )
            for tool, value in (side(c, "toolchain") or {}).items():
                tool_sets.setdefault(tool, [])
                if value is not None:
                    tool_sets[tool].append(str(value))
        wheels.append(entry)
    for name in meta_cells:
        art = files.get(name) or {}
        wheels.append(
            {
                "name": name,
                "kind": "meta",
                "sha256": art.get("sha256"),
                "size": art.get("size"),
                "uploaded_by_cells": sorted(meta_cells[name]),
            }
        )

    publish_dir = os.environ.get("PUBLISH_DIR") or "verification"
    doc = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": not errors,
        "errors": errors,
        "nixl": {
            "commit": nixl_commit,
            "sha8": nixl_sha,
            "ref": os.environ.get("NIXL_VERSION"),
            "version": binary_names[0].split("-")[1] if binary_names else None,
        },
        "ucx": {
            "repo": ucx_repo,
            "declared_ref": ucx_ref,
            "declared_commit": ucx_sha,
            "release": ucx_release,
            "branch": ucx_branch,
            "revision": ucx_revision,
        },
        "spcx": {"enabled": spcx_on == "true", "ref": spcx_ref, "commit": spcx_commit},
        "infinia": {
            "enabled": inf_on == "true",
            "image": inf_image,
            "digest": inf_digest,
        },
        "base_image": {
            cu: {
                "image": base_image[cu],
                "tag": base_tag[cu],
                "cuda_version": cuda_version[cu],
                "torch_requested": [
                    t for t in (torch_requested[cu] or "").split(",") if t
                ],
            }
            for cu in ("12", "13")
        },
        "torch": {"built": sorted(set(built_all)), "skipped": skipped_all},
        "toolchain": {
            tool: sorted(set(vals)) for tool, vals in sorted(tool_sets.items())
        },
        "jenkins": {
            "job": os.environ.get("JOB_NAME"),
            "build": (
                int(os.environ["BUILD_NUMBER"])
                if os.environ.get("BUILD_NUMBER", "").isdigit()
                else None
            ),
            "url": os.environ.get("BUILD_URL"),
            "ci_refspec": os.environ.get("ci_refspec"),
            "nixl_version": os.environ.get("NIXL_VERSION"),
            "publish_dir": publish_dir,
            "folder": folder,
        },
        "wheels": wheels,
        "unexpected_files": unexpected,
    }
    return folder, doc


# --- main ----------------------------------------------------------------------


def run(fragments: Path, out: Path, dry_run: bool) -> str:
    errors: list[str] = []
    cells = read_cells(fragments) if fragments.is_dir() else []
    folders = [
        c["cell"]["PUBLISH_FOLDER"]
        for c in cells
        if c["valid"] and c["cell"].get("PUBLISH_FOLDER")
    ]
    first = folders[0] if folders else None
    files: dict[str, dict[str, Any]] = {}
    if first and FOLDER.match(first):
        try:
            files = artifactory_files(first)
        except ManifestError as exc:
            errors.append(f"Artifactory listing of {first} failed: {exc}")
    folder, doc = assemble(cells, files, errors)
    text = json.dumps(doc, indent=2) + "\n"
    out.write_text(text, encoding="utf-8")
    if errors:
        raise ManifestError("; ".join(errors))
    if dry_run:
        return f"dry run: {out} written, not uploaded"
    return "Published " + upload(str(folder), text)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fragments",
        required=True,
        type=Path,
        help="directory of per-cell fragment dirs",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="where to write the manifest"
    )
    parser.add_argument(
        "--status-file", type=Path, help="one-line outcome for the caller"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read Artifactory and write the manifest, but do not upload it",
    )
    args = parser.parse_args(argv)
    try:
        status, rc = run(args.fragments, args.out, args.dry_run), 0
    except ManifestError as exc:
        status, rc = str(exc), 1
    except (
        Exception
    ) as exc:  # a bug here must still leave a readable reason for the build
        status, rc = f"{type(exc).__name__}: {exc}", 1
    print(
        status if rc == 0 else f"ERROR: {status}",
        file=sys.stdout if rc == 0 else sys.stderr,
    )
    if args.status_file:
        args.status_file.write_text(status + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
