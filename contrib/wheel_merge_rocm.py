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

"""Merge per-ROCm NIXL backend wheels into one rocm-nixl wheel.

Takes the nixl_rocm<X>_<Y> backend wheels built for ONE Python tag and writes
rocm_nixl-<version>-<python>-<abi>-<platform>.whl holding every backend plus
the ``nixl`` dispatcher, which selects the backend matching the installed ROCm
at import time.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import sys
import zipfile
from typing import NamedTuple

DIST_NAME = "rocm-nixl"
WHEEL_PREFIX = "rocm_nixl"
README_NAME = "README.rocm.md"
GENERATOR = "wheel_merge_rocm"


class RocmPytorch(NamedTuple):
    torch: str
    index_url: str


# AMD's official ROCm PyTorch build for each ROCm X.Y, pinned exactly: the
# "+rocm" local version exists only on AMD's index, so pip can never satisfy
# the extra with the CUDA build of torch from PyPI. Each pin must exist for
# every Python the wheels are built for. The 7.14 pin is a ROCm 7.14.0 build
# and brings the 7.14.0 pip ROCm runtime, while CI builds the backend against
# ROCm 7.14.1.
ROCM_PYTORCH = {
    "7.14": RocmPytorch(
        torch="2.13.0+rocm7.14.0",
        index_url="https://repo.amd.com/rocm/whl-multi-arch/",
    ),
    "10.1": RocmPytorch(
        torch="2.14.0+rocm10.1.0",
        index_url="https://stable.repo.amd.com/rocm/whl-next/",
    ),
}

_BACKEND_WHEEL_RE = re.compile(
    r"nixl_rocm(?P<major>\d+)_(?P<minor>\d+)-(?P<version>[^-]+)"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl"
)
_GENERATED_TIME = (1980, 1, 1, 0, 0, 0)
_REGULAR_FILE = 0o100644 << 16


class MergeError(Exception):
    """A backend wheel cannot be merged."""


class Backend(NamedTuple):
    path: str
    rocm: tuple[int, int]
    version: str
    python: str
    abi: str
    platform: str

    @property
    def package(self) -> str:
        return "nixl_rocm{}_{}".format(*self.rocm)

    @property
    def rocm_version(self) -> str:
        return "{}.{}".format(*self.rocm)

    @property
    def tags(self) -> tuple[str, str, str]:
        return (self.python, self.abi, self.platform)


def _sha256_b64(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_metadata(text: str) -> tuple[list[tuple[str, str]], str]:
    """Split METADATA into its (field, value) headers and its body."""
    head, _, body = text.partition("\n\n")
    headers: list[tuple[str, str]] = []
    for line in head.splitlines():
        if line[:1] in (" ", "\t") and headers:
            field, value = headers[-1]
            headers[-1] = (field, value + "\n" + line)
            continue
        field, sep, value = line.partition(":")
        if not sep:
            raise MergeError(f"malformed METADATA header line: {line!r}")
        headers.append((field.strip(), value.strip()))
    return headers, body


def _field(headers: list[tuple[str, str]], name: str) -> list[str]:
    return [value for field, value in headers if field.lower() == name.lower()]


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return _normalize(match.group(1)) if match else ""


def parse_backend(path: str) -> Backend:
    """Describe a backend wheel from its file name."""
    name = os.path.basename(path)
    match = _BACKEND_WHEEL_RE.fullmatch(name)
    if match is None:
        raise MergeError(
            f"{name}: not a nixl_rocm<X>_<Y> backend wheel (expected "
            "nixl_rocm<X>_<Y>-<version>-<python>-<abi>-<platform>.whl)"
        )
    return Backend(
        path=path,
        rocm=(int(match["major"]), int(match["minor"])),
        version=match["version"],
        python=match["python"],
        abi=match["abi"],
        platform=match["platform"],
    )


def check_backends(backends: list[Backend]) -> None:
    """Refuse backends that cannot share one wheel."""
    by_rocm: dict[tuple[int, int], Backend] = {}
    for backend in backends:
        other = by_rocm.setdefault(backend.rocm, backend)
        if other is not backend:
            raise MergeError(
                f"duplicate backend for ROCm {backend.rocm_version}: "
                f"{os.path.basename(other.path)} and {os.path.basename(backend.path)}"
            )
        if backend.rocm_version not in ROCM_PYTORCH:
            raise MergeError(
                f"{os.path.basename(backend.path)}: no PyTorch pin for ROCm "
                f"{backend.rocm_version} in ROCM_PYTORCH (known: "
                f"{', '.join(ROCM_PYTORCH)})"
            )

    first = backends[0]
    for backend in backends[1:]:
        for what, mine, theirs in (
            ("version", backend.version, first.version),
            (
                "Python/ABI tag",
                f"{backend.python}-{backend.abi}",
                f"{first.python}-{first.abi}",
            ),
            ("platform tag", backend.platform, first.platform),
        ):
            if mine != theirs:
                raise MergeError(
                    f"backends disagree on {what}: "
                    f"{os.path.basename(first.path)} has {theirs}, "
                    f"{os.path.basename(backend.path)} has {mine}"
                )


def read_backend_metadata(
    backend: Backend, whl: zipfile.ZipFile
) -> tuple[str, list[tuple[str, str]]]:
    """Return the backend's dist-info directory and its METADATA headers."""
    name = os.path.basename(backend.path)
    dist_infos = sorted(
        {
            entry.split("/", 1)[0]
            for entry in whl.namelist()
            if entry.split("/", 1)[0].endswith(".dist-info")
        }
    )
    if len(dist_infos) != 1:
        raise MergeError(
            f"{name}: expected one .dist-info directory, found "
            f"{', '.join(dist_infos) or 'none'}"
        )
    dist_info = dist_infos[0]
    try:
        text = whl.read(f"{dist_info}/METADATA").decode("utf-8")
    except KeyError:
        raise MergeError(f"{name}: no {dist_info}/METADATA") from None
    headers, _ = _split_metadata(text)

    expected = _normalize(f"nixl-rocm{backend.rocm_version}")
    names = _field(headers, "Name")
    if len(names) != 1 or _normalize(names[0]) != expected:
        raise MergeError(
            f"{name}: not a nixl_rocm<X>_<Y> backend wheel: METADATA Name is "
            f"{', '.join(names) or 'missing'}, expected nixl-rocm{backend.rocm_version}"
        )
    versions = _field(headers, "Version")
    if versions != [backend.version]:
        raise MergeError(
            f"{name}: METADATA Version {', '.join(versions) or 'missing'} does "
            f"not match the file name ({backend.version})"
        )
    if not any(e.startswith(f"{backend.package}/") for e in whl.namelist()):
        raise MergeError(f"{name}: import package {backend.package} is missing")
    return dist_info, headers


def license_files(
    name: str, whl: zipfile.ZipFile, dist_info: str, headers: list[tuple[str, str]]
) -> dict[str, bytes]:
    """The backend's license files, keyed by path inside its dist-info."""
    entries = set(whl.namelist())
    found: dict[str, bytes] = {}
    for license_file in _field(headers, "License-File"):
        # Metadata 2.4 keeps them under licenses/, older versions at the top.
        for relative in (f"licenses/{license_file}", license_file):
            if f"{dist_info}/{relative}" in entries:
                found[relative] = whl.read(f"{dist_info}/{relative}")
                break
        else:
            raise MergeError(f"{name}: license file {license_file} is missing")
    return found


def build_metadata(
    backends: list[Backend],
    all_headers: list[list[tuple[str, str]]],
    readme: str,
) -> str:
    """METADATA of the merged wheel, derived from the backends' METADATA."""
    first, headers = backends[0], all_headers[0]
    for field in ("Requires-Python", "License", "License-Expression", "License-File"):
        for backend, other in zip(backends[1:], all_headers[1:]):
            if _field(other, field) != _field(headers, field):
                raise MergeError(
                    f"backends disagree on {field}: "
                    f"{os.path.basename(first.path)} has "
                    f"{_field(headers, field) or 'none'}, "
                    f"{os.path.basename(backend.path)} has "
                    f"{_field(other, field) or 'none'}"
                )

    metadata_version = (_field(headers, "Metadata-Version") or ["2.1"])[0]
    # Metadata 2.3 and later require normalized extra names (PEP 685).
    normalized_extras = tuple(int(n) for n in metadata_version.split(".")[:2]) >= (2, 3)

    numpy = [
        requirement
        for requirement in _field(headers, "Requires-Dist")
        if _requirement_name(requirement) == "numpy" and "extra" not in requirement
    ] or ["numpy"]

    dropped = {
        "requires-dist",
        "provides-extra",
        "description",
        "description-content-type",
    }
    lines = []
    for field, value in headers:
        if field.lower() in dropped:
            continue
        if field.lower() == "name":
            value = DIST_NAME
        lines.append(f"{field}: {value}")
    lines += [f"Requires-Dist: {requirement}" for requirement in numpy]
    for backend in backends:
        extra = f"rocm{backend.rocm_version}"
        if normalized_extras:
            extra = _normalize(extra)
        pin = ROCM_PYTORCH[backend.rocm_version].torch
        lines.append(f"Provides-Extra: {extra}")
        lines.append(f'Requires-Dist: torch=={pin}; extra == "{extra}"')
    lines.append("Description-Content-Type: text/markdown")
    return "\n".join(lines) + "\n\n" + readme


def build_wheel_file(backend: Backend) -> str:
    lines = [
        "Wheel-Version: 1.0",
        f"Generator: {GENERATOR}",
        "Root-Is-Purelib: false",
    ]
    # A compressed tag set expands to one Tag line per combination.
    for python in backend.python.split("."):
        for abi in backend.abi.split("."):
            for platform in backend.platform.split("."):
                lines.append(f"Tag: {python}-{abi}-{platform}")
    return "\n".join(lines) + "\n"


def dispatcher_files(dispatcher_dir: str) -> dict[str, bytes]:
    """The nixl dispatcher package and nixl_meta_utils.py, keyed by wheel path."""
    package_dir = os.path.join(dispatcher_dir, "nixl")
    utils = os.path.join(dispatcher_dir, "nixl_meta_utils.py")
    if not os.path.isfile(os.path.join(package_dir, "__init__.py")):
        raise MergeError(f"no nixl/__init__.py in dispatcher dir {dispatcher_dir}")
    if not os.path.isfile(utils):
        raise MergeError(f"no nixl_meta_utils.py in dispatcher dir {dispatcher_dir}")
    paths = {
        f"nixl/{name}": os.path.join(package_dir, name)
        for name in sorted(os.listdir(package_dir))
        if name.endswith(".py")
    }
    paths["nixl_meta_utils.py"] = utils
    files = {}
    for name, path in paths.items():
        with open(path, "rb") as f:
            files[name] = f.read()
    return files


def _generated(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_GENERATED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = _REGULAR_FILE
    return info


def merge(backend_wheels: list[str], output_dir: str, dispatcher_dir: str) -> str:
    """Merge the backend wheels into one rocm_nixl wheel; return its path."""
    backends = sorted(
        (parse_backend(path) for path in backend_wheels), key=lambda b: b.rocm
    )
    if not backends:
        raise MergeError("no backend wheels given")
    check_backends(backends)
    first = backends[0]

    readme_path = os.path.join(dispatcher_dir, README_NAME)
    try:
        with open(readme_path, encoding="utf-8") as f:
            readme = f.read()
    except OSError as e:
        raise MergeError(f"cannot read {readme_path}: {e}") from e

    dist_info = f"{WHEEL_PREFIX}-{first.version}.dist-info"
    generated = dispatcher_files(dispatcher_dir)
    # Wheel path -> where it comes from, to report collisions.
    owners = {name: "the nixl dispatcher" for name in generated}

    wheels = [zipfile.ZipFile(backend.path, "r") for backend in backends]
    try:
        all_headers = []
        licenses: dict[str, bytes] = {}
        copies: list[tuple[zipfile.ZipFile, zipfile.ZipInfo]] = []
        for backend, whl in zip(backends, wheels):
            name = os.path.basename(backend.path)
            backend_dist_info, headers = read_backend_metadata(backend, whl)
            all_headers.append(headers)
            for relative, data in license_files(
                name, whl, backend_dist_info, headers
            ).items():
                if licenses.setdefault(relative, data) != data:
                    raise MergeError(
                        f"{name}: license file {relative} differs from "
                        f"{os.path.basename(first.path)}"
                    )
            for info in whl.infolist():
                top = info.filename.split("/", 1)[0]
                if info.is_dir() or top == backend_dist_info:
                    continue
                # Left out: the asio headers and pkg-config file leaked by a
                # meson subproject, and the example programs, whose wrappers
                # have the same name in every backend but exec different files.
                if top.endswith(".data") or top == f"{backend.package}.scripts":
                    continue
                owner = owners.setdefault(info.filename, name)
                if owner != name:
                    raise MergeError(
                        f"path collision: {info.filename} is in both {owner} and {name}"
                    )
                copies.append((whl, info))

        generated.update(
            {f"{dist_info}/{relative}": data for relative, data in licenses.items()}
        )
        generated[f"{dist_info}/METADATA"] = build_metadata(
            backends, all_headers, readme
        ).encode("utf-8")
        generated[f"{dist_info}/WHEEL"] = build_wheel_file(first).encode("utf-8")
        top_level = ["nixl", "nixl_meta_utils"] + [b.package for b in backends]
        generated[f"{dist_info}/top_level.txt"] = (
            "\n".join(sorted(top_level)) + "\n"
        ).encode("utf-8")
        for name in generated:
            if owners.setdefault(name, "the merged wheel") not in (
                "the merged wheel",
                "the nixl dispatcher",
            ):
                raise MergeError(
                    f"path collision: {name} is in both {owners[name]} and "
                    "the merged wheel"
                )

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir,
            f"{WHEEL_PREFIX}-{first.version}-{'-'.join(first.tags)}.whl",
        )
        record_path = f"{dist_info}/RECORD"
        rows = []
        tmp_path = f"{out_path}.tmp"
        with zipfile.ZipFile(
            tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zout:
            for whl, info in sorted(copies, key=lambda copy: copy[1].filename):
                data = whl.read(info)
                zout.writestr(info, data)
                rows.append(
                    [info.filename, f"sha256={_sha256_b64(data)}", str(len(data))]
                )
            # The dist-info directory goes last, RECORD at the very end.
            for name in sorted(generated, key=lambda n: (n.startswith(dist_info), n)):
                data = generated[name]
                zout.writestr(_generated(name), data)
                rows.append([name, f"sha256={_sha256_b64(data)}", str(len(data))])
            rows.append([record_path, "", ""])
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows(rows)
            zout.writestr(_generated(record_path), buf.getvalue().encode("utf-8"))
        os.replace(tmp_path, out_path)
    finally:
        for whl in wheels:
            whl.close()

    print(f"wrote {out_path}")
    print("backends included:")
    for backend in backends:
        pytorch = ROCM_PYTORCH[backend.rocm_version]
        print(
            f"  ROCm {backend.rocm_version}: {backend.package} "
            f"({os.path.basename(backend.path)}); extra rocm{backend.rocm_version} "
            f"-> torch=={pytorch.torch} from {pytorch.index_url}"
        )
    size = os.path.getsize(out_path)
    print(f"wheel size: {size} bytes ({size / (1024 * 1024):.1f} MiB)")
    return out_path


def main() -> int:
    default_dispatcher_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "bindings",
        "python",
        "nixl-meta",
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory to write the rocm_nixl wheel to",
    )
    parser.add_argument(
        "--dispatcher-dir",
        default=default_dispatcher_dir,
        help="directory holding the nixl dispatcher package, nixl_meta_utils.py "
        f"and {README_NAME} (default: %(default)s)",
    )
    parser.add_argument(
        "backend_wheels",
        nargs="+",
        metavar="BACKEND_WHEEL",
        help="nixl_rocm<X>_<Y> backend wheels, all for the same Python tag",
    )
    args = parser.parse_args()

    try:
        merge(args.backend_wheels, args.output_dir, args.dispatcher_dir)
    except (MergeError, OSError, zipfile.BadZipFile) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
