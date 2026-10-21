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

"""Unit tests for contrib/wheel_build_info.py (stdlib-only build-info sidecar)."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "contrib" / "wheel_build_info.py"

UCX_INFO = (
    "# Library version: 1.23.0\n"
    "# Library path: /usr/local/lib/libucs.so.0\n"
    "# API headers version: 1.23.0\n"
    "# Git branch 'v1.23.x', revision e6f37ab\n"
    "# Configured with: --disable-logging --enable-mt\n"
)

BINARY_WHL = "nixl_cu13-1.6.0-cp312-cp312-manylinux_2_28_x86_64.whl"


@pytest.fixture(scope="module")
def wbi():
    spec = importlib.util.spec_from_file_location("wheel_build_info", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_wheel_filename_binary(wbi):
    r = wbi.parse_wheel_filename(BINARY_WHL)
    assert r == {
        "package": "nixl_cu13",
        "version": "1.6.0",
        "build_tag": None,
        "python_tag": "cp312",
        "abi_tag": "cp312",
        "platform_tag": "manylinux_2_28_x86_64",
    }


def test_parse_wheel_filename_meta(wbi):
    assert (
        wbi.parse_wheel_filename("nixl-1.6.0-py3-none-any.whl")["platform_tag"] == "any"
    )


def test_parse_wheel_filename_build_tag(wbi):
    r = wbi.parse_wheel_filename(
        "nixl_rocm-1.6.0-1-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    assert (r["package"], r["build_tag"], r["python_tag"]) == (
        "nixl_rocm",
        "1",
        "cp312",
    )


def test_parse_wheel_filename_rejects_garbage(wbi):
    with pytest.raises(ValueError):
        wbi.parse_wheel_filename("not-a-wheel.txt")


def test_parse_ucx_info_full(wbi):
    r = wbi.parse_ucx_info(UCX_INFO)
    assert (r["release"], r["api_version"], r["branch"], r["revision"]) == (
        "1.23.0",
        "1.23.0",
        "v1.23.x",
        "e6f37ab",
    )
    assert r["configured_with"] == "--disable-logging --enable-mt"
    assert r["raw"] == UCX_INFO


def test_parse_ucx_info_without_git_line(wbi):
    r = wbi.parse_ucx_info("# Library version: 1.23.0\n")
    assert r["release"] == "1.23.0"
    assert r["branch"] is None and r["revision"] is None


def test_parse_ucx_info_none(wbi):
    assert set(wbi.parse_ucx_info(None).values()) == {None}


def test_parse_nvcc_version(wbi):
    assert (
        wbi.parse_nvcc_version("Cuda compilation tools, release 13.0, V13.0.88\n")
        == "13.0"
    )
    assert wbi.parse_nvcc_version(None) is None


def _sidecar(wbi, tmp_path, **overrides):
    whl = tmp_path / BINARY_WHL
    whl.write_bytes(b"x")
    kwargs = dict(
        wheel_path=whl,
        python_version="3.12",
        platform="manylinux_2_28_x86_64",
        cu_tag="cu130",
        torch_requested=[],
        torch_built=[],
        torch_skipped=[],
        skip_reason=None,
        toolchain_json=None,
        ucx_info_text=None,
        nvcc_text=None,
        git_commit=None,
        repair_packages={"auditwheel": None, "patchelf": None},
        generated_at="2026-10-19T00:00:00Z",
    )
    kwargs.update(overrides)
    return whl, wbi.build_sidecar(**kwargs)


def test_build_sidecar_nulls_when_tools_missing(wbi, tmp_path):
    whl, doc = _sidecar(wbi, tmp_path)
    assert doc["schema_version"] == 1
    assert doc["wheel"]["sha256"] == wbi.sha256_of(whl) and doc["wheel"]["size"] == 1
    assert doc["cuda"] == {"cu_tag": "cu130", "toolkit_version": None}
    assert doc["ucx"]["release"] is None and doc["nixl"]["commit"] is None
    assert doc["torch"] == {"requested": [], "built": [], "skipped": []}
    assert set(doc["toolchain"]) == {
        "meson",
        "meson-python",
        "pybind11",
        "patchelf",
        "setuptools",
        "pyyaml",
        "auditwheel",
        "patchelf_repair",
    }
    assert set(doc["toolchain"].values()) == {None}


def test_build_sidecar_torch_lists(wbi, tmp_path):
    tc = tmp_path / "toolchain-2.12.json"
    packages = {
        "meson": "1.9.1",
        "meson-python": "0.18.0",
        "pybind11": "3.0.1",
        "patchelf": "0.19.1.0",
        "setuptools": "80.9.0",
        "pyyaml": "6.0.3",
        "torch": "2.12.1+cu130",
    }
    tc.write_text(
        json.dumps({"schema_version": 1, "channel": "stable", "packages": packages})
    )
    reason = "no wheel on index for Python 3.12 + cu130 on x86_64"
    _, doc = _sidecar(
        wbi,
        tmp_path,
        torch_requested=["2.12", "2.15"],
        torch_built=[("2.12", tc), ("2.13", tmp_path / "missing.json")],
        torch_skipped=["2.15"],
        skip_reason=reason,
        toolchain_json=tc,
        ucx_info_text=UCX_INFO,
        nvcc_text="release 13.0, V13.0.88",
        git_commit="7379e112cde171bc662a827763992c46db8d3a26",
        repair_packages={"auditwheel": "6.8.1", "patchelf": "0.19.1.0"},
    )
    assert doc["torch"]["requested"] == ["2.12", "2.15"]
    expected_tc = {k: v for k, v in packages.items() if k != "torch"}
    assert doc["torch"]["built"][0] == {
        "version": "2.12",
        "channel": "stable",
        "resolved": "2.12.1+cu130",
        "toolchain": expected_tc,
    }
    assert doc["torch"]["built"][1] == {
        "version": "2.13",
        "channel": None,
        "resolved": None,
        "toolchain": None,
    }
    assert doc["torch"]["skipped"] == [{"version": "2.15", "reason": reason}]
    assert doc["toolchain"]["auditwheel"] == "6.8.1"
    assert doc["toolchain"]["patchelf_repair"] == "0.19.1.0"
    assert "torch" not in doc["toolchain"]
    assert (
        doc["ucx"]["revision"] == "e6f37ab" and doc["cuda"]["toolkit_version"] == "13.0"
    )
    assert doc["nixl"]["commit"] == "7379e112cde171bc662a827763992c46db8d3a26"


def test_cli_venv_toolchain_writes_file(wbi, tmp_path):
    out = tmp_path / "tc.json"
    assert wbi.main(["venv-toolchain", "--out", str(out), "--channel", "nightly"]) == 0
    d = json.loads(out.read_text())
    assert d["channel"] == "nightly"
    assert set(d["packages"]) == set(wbi.TOOLCHAIN_PACKAGES)


def test_cli_sidecar_writes_file(wbi, tmp_path, monkeypatch):
    whl = tmp_path / "nixl_cu12-1.6.0-cp310-cp310-manylinux_2_28_aarch64.whl"
    whl.write_bytes(b"xy")
    out = tmp_path / "out.json"
    monkeypatch.setattr(wbi, "run_capture", lambda argv: None)
    rc = wbi.main(
        [
            "sidecar",
            "--wheel",
            str(whl),
            "--out",
            str(out),
            "--python-version",
            "3.10",
            "--platform",
            "manylinux_2_28_aarch64",
            "--cu-tag",
            "cu129",
        ]
    )
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["wheel"]["name"] == whl.name
    assert d["python_version"] == "3.10"
    assert d["nixl"]["commit"] is None


def test_cli_sidecar_bad_wheel_name_exits_1(wbi, tmp_path):
    bad = tmp_path / "junk.whl"
    bad.write_bytes(b"x")
    rc = wbi.main(
        [
            "sidecar",
            "--wheel",
            str(bad),
            "--out",
            str(tmp_path / "o.json"),
            "--python-version",
            "3.10",
            "--platform",
            "p",
            "--cu-tag",
            "cu129",
        ]
    )
    assert rc == 1
