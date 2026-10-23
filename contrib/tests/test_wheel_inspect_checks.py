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

import pytest
from wheel_inspect import checks
from wheel_inspect.wheelset import Cell
from wheel_inspect_fixtures import make_set

U = [
    Cell(c, p, a)
    for c in ("cu12", "cu13")
    for p in ("cp310", "cp311")
    for a in ("x86_64", "aarch64")
]


def sides(tmp_path, **cand_kwargs):
    c = checks.load_side(make_set(tmp_path / "c", "1.6.1", **cand_kwargs))
    r = checks.load_side(make_set(tmp_path / "r", "1.6.0"))
    return c, r


def test_fold_scope_strings():
    assert checks.fold_scope(U, U) == "all"
    assert checks.fold_scope([c for c in U if c.cuda == "cu13"], U) == "cu13: all"
    assert (
        checks.fold_scope([Cell("cu12", "cp311", "x86_64")], U)
        == "cu12: cp311 x x86_64"
    )
    assert checks.fold_scope(
        [Cell("cu12", "cp310", "x86_64"), Cell("cu12", "cp311", "aarch64")], U
    ) == ("cu12: cp310-x86_64, cp311-aarch64")
    assert checks.fold_scope(
        [Cell("cu12", "cp310", "x86_64"), Cell("cu13", "cp310", "x86_64")], U
    ) == ("cu12: cp310 x x86_64; cu13: cp310 x x86_64")
    assert checks.fold_scope([], U) == "none"
    assert checks.fold_scope(U + [Cell.META], U) == "all; meta"
    assert checks.fold_scope([Cell.META], U) == "meta"


def test_all_pass_on_identical_sets(tmp_path):
    c, r = sides(tmp_path)
    f = {x.id: x for x in checks.run_checks(c, r)}
    assert [f[i].state for i in (1, 2, 6, 7)] == ["PASS"] * 4
    assert f[3].state == "SKIPPED" and f[4].state == "PASS" and "1.23.0" in f[4].finding
    assert f[5].state == "REVIEW"
    assert set(f[5].details["changed"][0].keys()) == {"key", "detail", "scope"}
    assert f[5].details["deleted"] == [] and f[5].details["new"] == []
    assert f[8].state == "REVIEW" and f[8].details["fatbin"] == {
        "skipped": "cuobjdump not available"
    }
    assert [x.id for x in checks.run_checks(c, r)] == list(range(1, 9))


def test_spcx_missing_in_one_cell_fails(tmp_path):
    c, r = sides(
        tmp_path,
        drop={"cu13/cp311/aarch64:nixl_cuXX.libs/ucx/libuct_ib_mlx5_gdp.so.0.0.0"},
    )
    f = checks.check_spcx(c, r)
    assert f.state == "FAIL"
    assert f.details["missing"] == [
        {"module": "libuct_ib_mlx5_gdp.so.0.0.0", "scope": "cu13: cp311 x aarch64"}
    ]


def test_infinia_bundled_red_lib_fails(tmp_path):
    c, r = sides(tmp_path, vendored={"libred_client-0a0a0a0a.so.2": b"x"})
    f = checks.check_infinia(c, r)
    assert f.state == "FAIL" and f.details["bundled_red_libs"]


def test_infinia_red_lib_anywhere_in_the_wheel_fails(tmp_path):
    c, r = sides(tmp_path, extra={"*:nixl_cuXX/libred_async.so": b"x"})
    f = checks.check_infinia(c, r)
    assert f.state == "FAIL" and f.details["bundled_red_libs"]


def test_infinia_pass_records_needed_and_runpath(tmp_path):
    c, r = sides(tmp_path)
    f = checks.check_infinia(c, r)
    assert (
        f.state == "PASS"
        and "libred_client.so.2" in f.details["needed"]
        and "/opt/ddn/red" in f.details["runpath"]
    )


def test_spcx_ref_from_manifest(tmp_path):
    c, r = sides(tmp_path)
    (c.dir / "build-manifest.json").write_text(
        json.dumps(
            {
                "spcx": {"enabled": True, "ref": "v0.3.x", "commit": "a" * 40},
                "ucx": {"release": "1.23.0"},
            }
        )
    )
    c = checks.load_side(c.dir)
    f = checks.check_spcx_ref(c, r)
    assert (
        f.state == "PASS"
        and f.details["candidate"]["commit"] == "a" * 40
        and f.details["changed"] is None
    )


def test_spcx_ref_disabled_while_bundled_fails(tmp_path):
    c, r = sides(tmp_path)
    (c.dir / "build-manifest.json").write_text(
        json.dumps({"spcx": {"enabled": False, "ref": None, "commit": None}})
    )
    assert checks.check_spcx_ref(checks.load_side(c.dir), r).state == "FAIL"


def test_ucx_disagreement_fails(tmp_path):
    c, r = sides(
        tmp_path,
        vendored_by_cell={
            "cu12/cp310/x86_64": {"libucp-deadbeef.so.0.0.0": b"\0ucp\0" + b"1.22.0\0"}
        },
    )
    f = checks.check_ucx(c, r)
    assert f.state == "FAIL" and "1.22.0" in f.finding and "1.23.0" in f.finding


def test_ucx_manifest_mismatch_fails_and_readme_wins(tmp_path):
    block = "\n## Build info\n\n```text\n# Library version: 1.23.0\n# Git branch 'v1.23.x', revision e6f37ab\n```\n"
    c, r = sides(tmp_path, readme_block=block)
    (c.dir / "build-manifest.json").write_text(
        json.dumps({"ucx": {"release": "1.24.0", "revision": "abc"}})
    )
    f = checks.check_ucx(checks.load_side(c.dir), r)
    assert (
        f.state == "FAIL"
        and f.details["revision"] == "e6f37ab"
        and f.details["release_by_source"]["readme"] == "1.23.0"
    )


def test_ucx_skipped_when_unreadable(tmp_path):
    c, r = sides(tmp_path, ucx_release="x")
    assert checks.check_ucx(c, r).state == "SKIPPED"


def test_ucx_skipped_when_a_subset_is_unreadable(tmp_path):
    c, r = sides(
        tmp_path,
        vendored_by_cell={
            "cu12/cp310/x86_64": {"libucp-deadbeef.so.0.0.0": b"\0junk\0"}
        },
    )
    f = checks.check_ucx(c, r)
    assert f.state == "SKIPPED" and "1.23.0" in f.finding and "7 of 8" in f.finding
    assert f.details["unreadable"] == "cu12: cp310 x x86_64"


def test_malformed_default_manifest_is_named_in_check_3(tmp_path):
    c, r = sides(tmp_path)
    (c.dir / "build-manifest.json").write_text("{not json")
    c = checks.load_side(c.dir)
    assert c.manifest is None and "not valid JSON" in (c.manifest_error or "")
    f = checks.check_spcx_ref(c, r)
    assert f.state == "SKIPPED" and "build-manifest.json" in f.finding
    with pytest.raises(ValueError, match="not valid JSON"):
        checks.load_side(c.dir, c.dir / "build-manifest.json")
    with pytest.raises(FileNotFoundError):
        checks.load_side(c.dir, c.dir / "nope.json")


def test_files_new_cu13_only_folds(tmp_path):
    c, r = sides(tmp_path, extra={"cu13:*:nixl_ep_cu13/nixl_ep_cpp_torch214.so": b"ep"})
    f = checks.check_files(c, r)
    assert f.state == "REVIEW"
    assert {
        "key": "nixl_ep_cu13/nixl_ep_cpp_torch214.so",
        "detail": "nixl_ep_cu13/nixl_ep_cpp_torch214.so",
        "scope": "cu13: all",
    } in f.details["new"]
    assert f.details["deleted"] == [] and f.details["counts"]["new"] == 1


def test_files_rehashed_and_mesonpy_and_asymmetry(tmp_path):
    c = checks.load_side(
        make_set(
            tmp_path / "c",
            "1.6.1",
            pys=("cp310", "cp311", "cp312"),
            extra={"*:.nixl_cuXX.mesonpy.libs/libnixl.so": b"changed"},
            vendored={"libucs-bbbbbbbb.so.0.0.0": b"ucs2"},
            drop={"*:nixl_cuXX.libs/libucs-aaaaaaaa.so.0.0.0"},
        )
    )
    r = checks.load_side(make_set(tmp_path / "r", "1.6.0"))
    f = checks.check_files(c, r)
    row = next(m for m in f.details["mesonpy"] if m["module"] == "libnixl.so")
    assert row["changed"] == 8 and row["identical"] == 0
    assert any(
        x["key"] == "nixl_cuXX.libs/libucs-HASH.so.0.0.0".replace("cuXX", "cu12")
        for x in f.details["rehashed"]
    )
    assert f.details["only_in_candidate"] == [
        "cu12/cp312/x86_64",
        "cu12/cp312/aarch64",
        "cu13/cp312/x86_64",
        "cu13/cp312/aarch64",
    ]
    assert f.details["only_in_reference"] == []
    assert any(v["field"] == "Version" for v in f.details["versions"])


def test_files_identical_sets_pass(tmp_path):
    c = checks.load_side(make_set(tmp_path / "c", "1.6.0"))
    r = checks.load_side(make_set(tmp_path / "r", "1.6.0"))
    assert checks.check_files(c, r).state == "PASS"


def test_nixl_plugins_inconsistent_within_variant_fails_and_lists_added(tmp_path):
    c, r = sides(
        tmp_path, extra={"cu12/cp310/x86_64:nixl_cuXX.libs/nixl/libplugin_NEW.so": b"p"}
    )
    f = checks.check_nixl_plugins(c, r)
    assert f.state == "FAIL"
    assert f.details["added"] == [
        {"module": "libplugin_NEW.so", "scope": "cu12: cp310 x x86_64"}
    ]


def test_nixl_plugins_pass_with_cu13_only_module(tmp_path):
    c, r = sides(tmp_path)
    f = checks.check_nixl_plugins(c, r)
    assert f.state == "PASS" and f.details["per_variant_count"] == {
        "cu12": 12,
        "cu13": 13,
    }
    assert f.details["added"] == [] and f.details["removed"] == []


def test_ucx_plugins_pass_counts_and_removed_vs_reference(tmp_path):
    c, r = sides(tmp_path, drop={"*:nixl_cuXX.libs/ucx/libucx_perftest_mad.so.0.0.0"})
    f = checks.check_ucx_plugins(c, r)
    assert f.state == "PASS" and f.details["per_variant_count"] == {
        "cu12": 10,
        "cu13": 10,
    }
    assert f.details["removed"] == [
        {"module": "libucx_perftest_mad.so.0.0.0", "scope": "all"}
    ]


def test_sizes_tables(tmp_path):
    c, r = sides(tmp_path, extra={"cu13:*:nixl_ep_cu13/big.so": b"\0" * 1_500_000})
    f = checks.check_sizes(c, r, None)
    assert (
        f.state == "REVIEW"
        and f.details["totals"]["candidate"] > f.details["totals"]["reference"]
    )
    assert any(
        b["key"] == "nixl_ep_cu13/big.so" and b["scope"] == "cu13: all"
        for b in f.details["big_deltas"]
    )
    assert len(f.details["per_cell"]) == 9 and {"cu12", "cu13"} <= {
        row["cuda"] for row in f.details["per_variant"]
    }
