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

from pathlib import Path

import pytest
from wheel_inspect import wheelset
from wheel_inspect.wheelset import Cell


def touch_set(
    root: Path,
    version: str,
    cudas=("cu12", "cu13"),
    pys=("cp310", "cp311"),
    arches=("x86_64", "aarch64"),
    meta=True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for cu in cudas:
        for py in pys:
            for arch in arches:
                (
                    root / f"nixl_{cu}-{version}-{py}-{py}-manylinux_2_28_{arch}.whl"
                ).touch()
    if meta:
        (root / f"nixl-{version}-py3-none-any.whl").touch()
    return root


def test_load_full_set(tmp_path):
    ws = wheelset.load_wheelset(touch_set(tmp_path, "1.6.0"))
    assert ws.version == "1.6.0"
    assert len(ws.binary) == 8
    assert ws.meta.name == "nixl-1.6.0-py3-none-any.whl"
    assert ws.cells[-1].is_meta
    assert ws.cells[0] == Cell("cu12", "cp310", "x86_64")


def test_cell_of_and_labels():
    assert wheelset.cell_of("nixl_cu13", "cp312", "manylinux_2_28_aarch64") == Cell(
        "cu13", "cp312", "aarch64"
    )
    assert wheelset.cell_of("nixl", "py3", "any").is_meta
    assert Cell("cu13", "cp312", "x86_64").label == "cu13/cp312/x86_64"
    assert Cell.META.label == "meta"
    with pytest.raises(ValueError):
        wheelset.cell_of("nixl_rocm", "cp312", "manylinux_2_28_x86_64")
    with pytest.raises(ValueError):
        wheelset.cell_of("nixl_cu13", "cp312", "win_amd64")


def test_sort_cells_orders_cuda_py_arch_meta_last():
    cells = [
        Cell.META,
        Cell("cu13", "cp310", "aarch64"),
        Cell("cu12", "cp311", "x86_64"),
        Cell("cu12", "cp310", "aarch64"),
    ]
    assert [c.label for c in wheelset.sort_cells(cells)] == [
        "cu12/cp310/aarch64",
        "cu12/cp311/x86_64",
        "cu13/cp310/aarch64",
        "meta",
    ]


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (
            lambda d: (
                d / "nixl_cu13-1.6.0-cp311-cp311-manylinux_2_28_aarch64.whl"
            ).unlink(),
            "not a full product",
        ),
        (lambda d: (d / "nixl-1.6.0-py3-none-any.whl").unlink(), "meta wheel"),
        (lambda d: (d / "nixl-1.6.1-py3-none-any.whl").touch(), "meta wheel"),
        (
            lambda d: (
                d / "nixl_cu13-1.6.1-cp311-cp311-manylinux_2_28_aarch64.whl"
            ).touch(),
            "version",
        ),
        (
            lambda d: (
                d / "nixl_rocm-1.6.0-cp311-cp311-manylinux_2_28_x86_64.whl"
            ).touch(),
            "nixl_rocm",
        ),
        (lambda d: (d / "junk.whl").touch(), "junk.whl"),
    ],
)
def test_load_rejects_malformed_sets(tmp_path, mutate, needle):
    d = touch_set(tmp_path, "1.6.0")
    mutate(d)
    with pytest.raises(wheelset.WheelSetError, match=needle):
        wheelset.load_wheelset(d)


def test_load_empty_dir_rejected(tmp_path):
    with pytest.raises(wheelset.WheelSetError):
        wheelset.load_wheelset(tmp_path)


def test_pairs_with_reports_asymmetry(tmp_path):
    a = wheelset.load_wheelset(
        touch_set(tmp_path / "a", "1.6.0", pys=("cp310", "cp311", "cp312"))
    )
    b = wheelset.load_wheelset(
        touch_set(tmp_path / "b", "1.5.0", pys=("cp310", "cp311"))
    )
    shared, only_a, only_b = a.pairs_with(b)
    assert len(shared) == 8
    assert [c.py for c in only_a] == ["cp312"] * 4
    assert only_b == []
