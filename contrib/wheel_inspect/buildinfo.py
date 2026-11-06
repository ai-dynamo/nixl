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

"""Facts read out of a wheel's metadata: the README build-info block, SBOM, and the UCX string in libucp."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Lines inside the block that do not start with one of these continue the previous line
# (the Dockerfile pipes the block through `fold -s -w 96`).
_LINE_STARTS = ("NIXL commit:", "UCX", "# ", "SPCX Plugin:", "Infinia Plugin:")
_GIT_LINE = re.compile(r"Git branch '([^']+)', revision ([0-9a-f]+)")
# NUL-terminated x.y.z not glued to a longer token (GLIBC_2.2.5 in .dynstr, GCC version in .comment).
_VERSION_STRING = re.compile(rb"(?<![0-9._])(\d+\.\d+\.\d+)(?=\x00)")
_LIBUCP = re.compile(r"^nixl_cu\d+\.libs/libucp-[0-9a-f]{8}\.so")


@dataclass
class BuildInfoBlock:
    nixl_commit: Optional[str] = None
    ucx_release: Optional[str] = None
    ucx_api_version: Optional[str] = None
    ucx_branch: Optional[str] = None
    ucx_revision: Optional[str] = None
    ucx_configured_with: Optional[str] = None
    spcx_plugin: Optional[str] = None
    infinia_plugin: Optional[str] = None


def _value(line: str, prefix: str) -> Optional[str]:
    value = line[len(prefix) :].strip()
    return value if value and value != "unavailable" else None


def parse_build_info(readme: str) -> Optional[BuildInfoBlock]:
    start = readme.find("## Build info")
    if start < 0:
        return None
    unfolded: list[str] = []
    for raw in readme[start + len("## Build info") :].splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            break
        if not line or line.startswith("```"):
            continue
        if unfolded and not line.startswith(_LINE_STARTS):
            unfolded[-1] += " " + line.strip()
        else:
            unfolded.append(line)

    block = BuildInfoBlock()
    for line in unfolded:
        if line.startswith("NIXL commit:"):
            block.nixl_commit = _value(line, "NIXL commit:")
        elif line.startswith("# Library version:"):
            block.ucx_release = _value(line, "# Library version:")
        elif line.startswith("# API headers version:"):
            block.ucx_api_version = _value(line, "# API headers version:")
        elif line.startswith("# Git branch"):
            match = _GIT_LINE.search(line)
            if match:
                block.ucx_branch, block.ucx_revision = match.group(1), match.group(2)
        elif line.startswith("# Configured with:"):
            block.ucx_configured_with = _value(line, "# Configured with:")
        elif line.startswith("SPCX Plugin:"):
            block.spcx_plugin = _value(line, "SPCX Plugin:")
        elif line.startswith("Infinia Plugin:"):
            block.infinia_plugin = _value(line, "Infinia Plugin:")
    return block


def scan_ucx_release(data: bytes) -> Optional[str]:
    """The UCX release string in libucp: a NUL-terminated x.y.z, anchored near 'ucp'/'ucx' when ambiguous.

    ucp_get_version_string() returns a static "1.23.0" in .rodata; the bytes before it are data,
    so only the terminator and the neighbourhood are trusted.
    """
    found: dict[str, list[int]] = {}
    for match in _VERSION_STRING.finditer(data):
        found.setdefault(match.group(1).decode(), []).append(match.start())
    if not found:
        return None
    if len(found) == 1:
        return next(iter(found))
    anchored = []
    for value, offsets in found.items():
        for offset in offsets:
            window = data[max(0, offset - 512) : offset + 512].lower()
            if b"ucp" in window or b"ucx" in window:
                anchored.append(value)
                break
    return anchored[0] if len(set(anchored)) == 1 else None


@dataclass
class WheelFacts:
    metadata_version: Optional[str] = None
    requires_dist: list[str] = field(default_factory=list)
    wheel_generator: Optional[str] = None
    sbom_tools: dict[str, str] = field(default_factory=dict)
    sbom_components: dict[str, str] = field(default_factory=dict)
    build_info: Optional[BuildInfoBlock] = None
    ucx_release_scanned: Optional[str] = None


def _read_text(zf: zipfile.ZipFile, name: str) -> Optional[str]:
    try:
        return zf.read(name).decode("utf-8", "replace")
    except KeyError:
        return None


def _sbom_versions(items: object) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(items, dict):
        items = items.get("components", [])
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("name"):
                out[str(item["name"])] = str(item.get("version", ""))
    return out


def wheel_facts(wheel: Path, package: str, version: str) -> WheelFacts:
    facts = WheelFacts()
    prefix = f"{package}-{version}.dist-info/"
    with zipfile.ZipFile(wheel) as zf:
        metadata = _read_text(zf, prefix + "METADATA")
        if metadata is not None:
            headers, _, body = metadata.partition("\n\n")
            for line in headers.splitlines():
                if line.startswith("Version:"):
                    facts.metadata_version = line[len("Version:") :].strip()
                elif line.startswith("Requires-Dist:"):
                    facts.requires_dist.append(line[len("Requires-Dist:") :].strip())
            facts.build_info = parse_build_info(body)
        wheel_text = _read_text(zf, prefix + "WHEEL")
        if wheel_text is not None:
            for line in wheel_text.splitlines():
                if line.startswith("Generator:"):
                    facts.wheel_generator = line[len("Generator:") :].strip()
        sbom_text = _read_text(zf, prefix + "sboms/auditwheel.cdx.json")
        if sbom_text is not None:
            try:
                sbom = json.loads(sbom_text)
            except ValueError:
                sbom = {}
            if isinstance(sbom, dict):
                facts.sbom_tools = _sbom_versions(
                    (sbom.get("metadata") or {}).get("tools")
                )
                facts.sbom_components = _sbom_versions(sbom.get("components"))
        libucp = sorted(n for n in zf.namelist() if _LIBUCP.match(n))
        if libucp:
            facts.ucx_release_scanned = scan_ucx_release(zf.read(libucp[0]))
    return facts
