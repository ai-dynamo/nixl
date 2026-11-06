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

import json

from wheel_inspect import buildinfo
from wheel_inspect_fixtures import make_wheel

BLOCK = (
    "Intro text\n\n## Build info\n\n```text\n"
    "NIXL commit: 7379e112cde171bc662a827763992c46db8d3a26\n"
    "UCX (ucx_info -v):\n"
    "# Library version: 1.23.0\n"
    "# Library path: /usr/local/lib/libucs.so.0\n"
    "# API headers version: 1.23.0\n"
    "# Git branch 'v1.23.x', revision e6f37ab\n"
    "# Configured with: --disable-logging --disable-debug --disable-assertions\n"
    "--disable-params-check --enable-mt --enable-shared --with-cuda=/usr/local/cuda\n"
    "--with-verbs --with-dm\n"
    "SPCX Plugin: 14571103c6e8f6b1a0790a837636674f737899ea\n"
    "Infinia Plugin: v2.5.0-rcx.1@sha256:abc\n"
    "```\n"
)


def test_build_info_folded_lines():
    b = buildinfo.parse_build_info(BLOCK)
    assert b is not None
    assert b.ucx_release == "1.23.0" and b.ucx_api_version == "1.23.0"
    assert b.ucx_revision == "e6f37ab" and b.ucx_branch == "v1.23.x"
    assert b.ucx_configured_with is not None
    assert (
        b.ucx_configured_with.endswith("--with-verbs --with-dm")
        and "--enable-mt" in b.ucx_configured_with
    )
    assert b.nixl_commit is not None and b.nixl_commit.startswith("7379e112")
    assert b.spcx_plugin is not None and b.spcx_plugin.startswith("14571103")
    assert b.infinia_plugin == "v2.5.0-rcx.1@sha256:abc"


def test_build_info_absent_and_unavailable():
    assert buildinfo.parse_build_info("# NIXL\n\nno block here\n") is None
    b = buildinfo.parse_build_info(
        "## Build info\n\n```text\nNIXL commit: unavailable\nUCX: ucx_info unavailable\n```\n"
    )
    assert b is not None
    assert b.nixl_commit is None and b.ucx_release is None and b.spcx_plugin is None


def test_scan_ucx_release_unique_and_anchored():
    assert buildinfo.scan_ucx_release(b"\0junk\0" + b"1.23.0\0" + b"more\0") == "1.23.0"
    data = (
        b"\0" * 1024 + b"2.28.1\0" + b"\0" * 2048 + b"ucp_get_version\0" + b"1.23.0\0"
    )
    assert buildinfo.scan_ucx_release(data) == "1.23.0"
    assert buildinfo.scan_ucx_release(b"1.2.3\0" + b"4.5.6\0") is None
    assert buildinfo.scan_ucx_release(b"nothing here\0") is None
    assert buildinfo.scan_ucx_release(b"\xcd\xcdA1.23.0\0ucp_rkey\0") == "1.23.0"
    assert buildinfo.scan_ucx_release(b"GLIBC_2.2.5\0ucp\0" + b"1.23.0\0") == "1.23.0"
    assert (
        buildinfo.scan_ucx_release(b"GCC: (GNU) 14.2.1 2025\0" + b"\x00ucp\x001.23.0\0")
        == "1.23.0"
    )


def test_wheel_facts_reads_metadata_sbom_and_scan(tmp_path):
    sbom = {
        "metadata": {"tools": [{"name": "auditwheel", "version": "6.8.1"}]},
        "components": [{"name": "libibverbs", "version": "48.0-1.el8"}],
    }
    w = make_wheel(
        tmp_path / "nixl_cu13-1.6.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        {
            "nixl_cu13-1.6.0.dist-info/METADATA": (
                "Metadata-Version: 2.1\nName: nixl-cu13\nVersion: 1.6.0\nRequires-Dist: numpy\n\n"
                + BLOCK
            ).encode(),
            "nixl_cu13-1.6.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: meson\n",
            "nixl_cu13-1.6.0.dist-info/sboms/auditwheel.cdx.json": json.dumps(
                sbom
            ).encode(),
            "nixl_cu13.libs/libucp-72ad654d.so.0.0.0": b"\0ucp\0" + b"1.23.0\0",
        },
    )
    f = buildinfo.wheel_facts(w, "nixl_cu13", "1.6.0")
    assert (
        f.metadata_version == "1.6.0"
        and f.requires_dist == ["numpy"]
        and f.wheel_generator == "meson"
    )
    assert f.sbom_tools == {"auditwheel": "6.8.1"} and f.sbom_components == {
        "libibverbs": "48.0-1.el8"
    }
    assert f.build_info is not None and f.build_info.ucx_release == "1.23.0"
    assert f.ucx_release_scanned == "1.23.0"


def test_wheel_facts_tolerates_missing_pieces(tmp_path):
    w = make_wheel(
        tmp_path / "nixl-1.6.0-py3-none-any.whl",
        {"nixl-1.6.0.dist-info/METADATA": b"Version: 1.6.0\n"},
    )
    f = buildinfo.wheel_facts(w, "nixl", "1.6.0")
    assert f.metadata_version == "1.6.0" and f.requires_dist == []
    assert f.build_info is None and f.ucx_release_scanned is None
    assert f.sbom_tools == {} and f.sbom_components == {} and f.wheel_generator is None
