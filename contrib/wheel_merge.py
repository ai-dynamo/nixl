#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Merge file(s) from one wheel into another, regenerating RECORD."""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import io
import os
import sys
import zipfile


def _sha256_b64(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _record_bytes(entries: list[tuple[str, bytes]], record_path: str) -> bytes:
    rows = [
        [name, f"sha256={_sha256_b64(data)}", str(len(data))]
        for name, data in entries
        if name != record_path
    ]
    rows.append([record_path, "", ""])
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue().encode("utf-8")


def merge(
    base_wheel: str,
    source_wheel: str,
    pattern: str,
    target_dir: str,
) -> list[str]:
    """Merge files matching `pattern` from `source_wheel` into `base_wheel`.

    `pattern` is a fnmatch glob applied to the basename of each entry in
    the source wheel; matches are placed under `target_dir/` inside the
    base wheel. `base_wheel` is rewritten atomically with a regenerated
    RECORD. Returns the list of merged entry names (relative to the
    wheel root).
    """
    target_dir = target_dir.rstrip("/")

    # Pull matching files out of the source wheel, rewriting their path so
    # they land under target_dir/.
    merged: dict[str, tuple[zipfile.ZipInfo, bytes]] = {}
    with zipfile.ZipFile(source_wheel, "r") as zsrc:
        for info in zsrc.infolist():
            name = os.path.basename(info.filename)
            if not fnmatch.fnmatch(name, pattern):
                continue
            new_name = f"{target_dir}/{name}"
            new_info = zipfile.ZipInfo(filename=new_name)
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            new_info.date_time = info.date_time
            merged[new_name] = (new_info, zsrc.read(info))

    if not merged:
        raise SystemExit(
            f"no files matched {pattern!r} in {source_wheel}"
        )

    # Read base wheel; pull out RECORD path so we can regenerate it.
    by_name: dict[str, tuple[zipfile.ZipInfo, bytes]] = {}
    record_path: str | None = None
    with zipfile.ZipFile(base_wheel, "r") as zin:
        for info in zin.infolist():
            if info.filename.endswith(".dist-info/RECORD"):
                record_path = info.filename
                continue
            by_name[info.filename] = (info, zin.read(info))
    if record_path is None:
        raise SystemExit(f"no .dist-info/RECORD found in {base_wheel}")

    # Apply merge (source overrides base if a name collides).
    by_name.update(merged)

    # Sort for stable output; nice for diffing two wheels.
    ordered = sorted(by_name)
    record = _record_bytes([(n, by_name[n][1]) for n in ordered], record_path)

    record_info = zipfile.ZipInfo(filename=record_path)
    record_info.compress_type = zipfile.ZIP_DEFLATED

    tmp_path = f"{base_wheel}.tmp"
    with zipfile.ZipFile(
        tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for name in ordered:
            info, data = by_name[name]
            zout.writestr(info, data)
        zout.writestr(record_info, record)
    os.replace(tmp_path, base_wheel)

    return sorted(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-wheel",
        required=True,
        help="wheel to merge into (rewritten in place)",
    )
    parser.add_argument(
        "--source-wheel",
        required=True,
        help="wheel to extract files from",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="basename glob of files to merge (e.g. 'nixl_ep_cpp_torch212.*')",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="directory inside the base wheel for the merged files "
        "(e.g. 'nixl_ep_cu13')",
    )
    args = parser.parse_args()

    merged = merge(
        base_wheel=args.base_wheel,
        source_wheel=args.source_wheel,
        pattern=args.pattern,
        target_dir=args.target_dir,
    )
    print(f"merged {len(merged)} file(s) into {args.base_wheel}:")
    for name in merged:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
