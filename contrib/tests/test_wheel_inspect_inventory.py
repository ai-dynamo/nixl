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

import hashlib

import pytest
from wheel_inspect import inventory
from wheel_inspect_fixtures import make_wheel


@pytest.mark.parametrize(
    "path,key",
    [
        ("nixl_cu12-1.4.1.dist-info/METADATA", "nixl_cu12-VER.dist-info/METADATA"),
        (
            "nixl_cu12-1.4.1.data/headers/asio.hpp",
            "nixl_cu12-VER.data/headers/asio.hpp",
        ),
        ("nixl-1.4.1.dist-info/RECORD", "nixl-VER.dist-info/RECORD"),
        (
            "nixl_cu12.libs/libucp-72ad654d.so.0.0.0",
            "nixl_cu12.libs/libucp-HASH.so.0.0.0",
        ),
        (
            "nixl_cu12.libs/libgusli_clnt-7ab00740.so",
            "nixl_cu12.libs/libgusli_clnt-HASH.so",
        ),
        (
            "nixl_cu12.libs/ucx/libuct_ib.so.0.0.0",
            "nixl_cu12.libs/ucx/libuct_ib.so.0.0.0",
        ),
        (".nixl_cu12.mesonpy.libs/libnixl.so", ".nixl_cu12.mesonpy.libs/libnixl.so"),
        (
            "nixl_ep_cu13/nixl_ep_cpp_torch214.cpython-312-x86_64-linux-gnu.so",
            "nixl_ep_cu13/nixl_ep_cpp_torch214.cpython-312-x86_64-linux-gnu.so",
        ),
    ],
)
def test_key_of(path, key):
    assert inventory.key_of(path) == key


def test_key_of_hash_suffix_exact_width():
    assert (
        inventory.key_of("nixl_cu12.libs/libfoo.so.1") == "nixl_cu12.libs/libfoo.so.1"
    )
    assert (
        inventory.key_of("nixl_cu12.libs/libfoo-abcdef1.so")
        == "nixl_cu12.libs/libfoo-abcdef1.so"
    )
    assert (
        inventory.key_of("nixl_cu12.libs/libfoo-abcdef123.so")
        == "nixl_cu12.libs/libfoo-abcdef123.so"
    )


def test_inventory_hashes_and_sizes(tmp_path):
    w = make_wheel(
        tmp_path / "a.whl",
        {
            "nixl_cu12/__init__.py": b"x" * 10,
            "nixl_cu12-1.0.0.dist-info/METADATA": b"Version: 1.0.0\n",
        },
    )
    inv = inventory.inventory(w)
    m = inv["nixl_cu12/__init__.py"]
    assert m.digest == hashlib.sha256(b"x" * 10).hexdigest()
    assert m.size == 10 and m.compressed > 0 and m.path == "nixl_cu12/__init__.py"
    assert "nixl_cu12-VER.dist-info/METADATA" in inv


def test_inventory_symlink_member(tmp_path):
    w = make_wheel(
        tmp_path / "a.whl",
        {"nixl_cu12.libs/ucx/libuct_ib.so": ("symlink", "libuct_ib.so.0.0.0")},
    )
    assert (
        inventory.inventory(w)["nixl_cu12.libs/ucx/libuct_ib.so"].digest
        == "symlink:libuct_ib.so.0.0.0"
    )


def test_inventory_key_collision_raises(tmp_path):
    w = make_wheel(
        tmp_path / "a.whl",
        {
            "nixl_cu12.libs/libucp-11111111.so.0.0.0": b"a",
            "nixl_cu12.libs/libucp-22222222.so.0.0.0": b"b",
        },
    )
    with pytest.raises(inventory.InventoryError, match="libucp-HASH"):
        inventory.inventory(w)


def test_diff_pair_classes(tmp_path):
    ref = inventory.inventory(
        make_wheel(
            tmp_path / "r.whl",
            {
                "nixl_cu12-1.4.0.dist-info/METADATA": b"Version: 1.4.0",
                "nixl_cu12-1.4.0.data/headers/asio.hpp": b"same",
                "nixl_cu12.libs/libucp-5c599099.so.0.0.0": b"old ucp",
                "nixl_cu12.libs/libucs-11111111.so.0.0.0": b"ucs",
                ".nixl_cu12.mesonpy.libs/libnixl.so": b"v1",
                "gone.txt": b"bye",
            },
        )
    )
    cand = inventory.inventory(
        make_wheel(
            tmp_path / "c.whl",
            {
                "nixl_cu12-1.4.1.dist-info/METADATA": b"Version: 1.4.1",
                "nixl_cu12-1.4.1.data/headers/asio.hpp": b"same",
                "nixl_cu12.libs/libucp-72ad654d.so.0.0.0": b"new ucp",
                "nixl_cu12.libs/libucs-11111111.so.0.0.0": b"ucs patched",
                ".nixl_cu12.mesonpy.libs/libnixl.so": b"v2",
                "nixl_ep_cu12/new.so": b"new",
            },
        )
    )
    d = inventory.diff_pair(ref, cand)
    assert set(d.new) == {"nixl_ep_cu12/new.so"}
    assert set(d.deleted) == {"gone.txt"}
    assert set(d.rehashed) == {"nixl_cu12.libs/libucp-HASH.so.0.0.0"}
    assert set(d.changed) == {
        "nixl_cu12-VER.dist-info/METADATA",
        "nixl_cu12.libs/libucs-HASH.so.0.0.0",
        ".nixl_cu12.mesonpy.libs/libnixl.so",
    }
    assert "nixl_cu12-VER.data/headers/asio.hpp" not in d.changed
    old, new = d.rehashed["nixl_cu12.libs/libucp-HASH.so.0.0.0"]
    assert old.path.endswith("5c599099.so.0.0.0") and new.path.endswith(
        "72ad654d.so.0.0.0"
    )
