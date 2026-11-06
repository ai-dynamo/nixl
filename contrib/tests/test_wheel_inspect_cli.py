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

from wheel_inspect import checks, cli, report
from wheel_inspect_fixtures import make_set


def test_inspect_cli_writes_three_reports_and_table(tmp_path, capsys):
    make_set(tmp_path / "c", "1.6.1")
    make_set(tmp_path / "r", "1.6.0")
    rc = cli.main(
        [
            "inspect",
            "--candidate",
            str(tmp_path / "c"),
            "--reference",
            str(tmp_path / "r"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    for ext in ("json", "md", "html"):
        assert (tmp_path / "out" / f"wheel-inspection.{ext}").is_file()
    doc = json.loads((tmp_path / "out" / "wheel-inspection.json").read_text())
    assert doc["schema_version"] == 1 and [c["id"] for c in doc["checks"]] == list(
        range(1, 9)
    )
    assert set(doc["summary"]) == {"PASS", "FAIL", "REVIEW", "SKIPPED"}
    assert doc["candidate"]["label"] == "1.6.1" and doc["candidate"]["wheels"] == 9
    assert (
        doc["candidate"]["manifest"] is False and len(doc["candidate"]["sha256"]) == 9
    )
    out = capsys.readouterr().out
    assert "SPCX plugin bundling" in out and "| 8 |" in out


def test_inspect_cli_short_set_exits_2(tmp_path, capsys):
    d = make_set(tmp_path / "c", "1.6.1")
    next(d.glob("nixl_cu13-*aarch64.whl")).unlink()
    make_set(tmp_path / "r", "1.6.0")
    rc = cli.main(
        [
            "inspect",
            "--candidate",
            str(d),
            "--reference",
            str(tmp_path / "r"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "not a full product" in capsys.readouterr().err
    assert not (tmp_path / "out" / "wheel-inspection.json").exists()


def test_inspect_cli_corrupt_wheel_exits_2(tmp_path, capsys):
    d = make_set(tmp_path / "c", "1.6.1")
    bad = next(d.glob("nixl_cu12-*x86_64.whl"))
    bad.write_bytes(b"not a zip")
    make_set(tmp_path / "r", "1.6.0")
    rc = cli.main(
        [
            "inspect",
            "--candidate",
            str(d),
            "--reference",
            str(tmp_path / "r"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2 and bad.name in capsys.readouterr().err


def test_inspect_cli_unwritable_out_dir_exits_2(tmp_path, capsys):
    make_set(tmp_path / "c", "1.6.1")
    make_set(tmp_path / "r", "1.6.0")
    (tmp_path / "file").write_text("x")
    rc = cli.main(
        [
            "inspect",
            "--candidate",
            str(tmp_path / "c"),
            "--reference",
            str(tmp_path / "r"),
            "--out-dir",
            str(tmp_path / "file" / "out"),
        ]
    )
    assert rc == 2 and "wheel_inspect:" in capsys.readouterr().err


def test_inspect_cli_explicit_missing_manifest_exits_2(tmp_path, capsys):
    make_set(tmp_path / "c", "1.6.1")
    make_set(tmp_path / "r", "1.6.0")
    rc = cli.main(
        [
            "inspect",
            "--candidate",
            str(tmp_path / "c"),
            "--reference",
            str(tmp_path / "r"),
            "--out-dir",
            str(tmp_path / "out"),
            "--candidate-manifest",
            str(tmp_path / "nope.json"),
        ]
    )
    assert rc == 2 and "nope.json" in capsys.readouterr().err


def test_inspect_cli_missing_dir_exits_2(tmp_path, capsys):
    make_set(tmp_path / "r", "1.6.0")
    rc = cli.main(
        [
            "inspect",
            "--candidate",
            str(tmp_path / "nope"),
            "--reference",
            str(tmp_path / "r"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2 and "nope" in capsys.readouterr().err


def test_markdown_and_html_render_every_check(tmp_path):
    c = checks.load_side(make_set(tmp_path / "c", "1.6.1"))
    r = checks.load_side(make_set(tmp_path / "r", "1.6.0"))
    doc = report.build_document(c, r, checks.run_checks(c, r), "test")
    md, html_text = report.render_markdown(doc), report.render_html(doc)
    for name in checks.CHECK_NAMES.values():
        assert name in md and name in html_text
    assert "<script" not in html_text and "1.6.1" in html_text and "1.6.0" in md
    assert report.summary_line(doc).startswith("PASS ")


def test_html_escapes_untrusted_text():
    doc = {
        "schema_version": 1,
        "generated_at": "x",
        "tool": {"version": "t"},
        "candidate": {
            "label": "<b>",
            "dir": "",
            "wheels": 0,
            "manifest": False,
            "version": "",
            "sha256": {},
        },
        "reference": {
            "label": "r",
            "dir": "",
            "wheels": 0,
            "manifest": False,
            "version": "",
            "sha256": {},
        },
        "summary": {"PASS": 0, "FAIL": 0, "REVIEW": 0, "SKIPPED": 0},
        "checks": [
            {
                "id": 1,
                "name": "n",
                "state": "PASS",
                "finding": "<img src=x onerror=1>",
                "details": {},
            }
        ],
    }
    html_text = report.render_html(doc)
    assert (
        "<img" not in html_text and "&lt;img" in html_text and "&lt;b&gt;" in html_text
    )
