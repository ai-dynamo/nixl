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

"""The report document (JSON) and its Markdown and HTML renderings."""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from wheel_inspect.checks import STATES, Finding, Side

SCHEMA_VERSION = 1
REPORT_BASENAME = "wheel-inspection"

Block = tuple[Any, ...]  # ("h1"|"h2"|"p", text) or ("table", headers, rows)


def tool_version() -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return (
        proc.stdout.strip()
        if proc.returncode == 0 and proc.stdout.strip()
        else "unknown"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _side_doc(side: Side) -> dict[str, Any]:
    return {
        "label": side.label,
        "dir": str(side.dir),
        "version": side.wheelset.version,
        "wheels": len(side.wheelset.wheels),
        "manifest": side.manifest is not None,
        "sha256": {
            wf.name: _sha256_file(wf.path)
            for wf in sorted(side.wheelset.wheels.values(), key=lambda w: w.name)
        },
    }


def build_document(
    candidate: Side, reference: Side, findings: list[Finding], tool_version_: str
) -> dict[str, Any]:
    summary = {state: 0 for state in STATES}
    for f in findings:
        summary[f.state] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {"name": "wheel_inspect", "version": tool_version_},
        "candidate": _side_doc(candidate),
        "reference": _side_doc(reference),
        "summary": summary,
        "checks": [
            {
                "id": f.id,
                "name": f.name,
                "state": f.state,
                "finding": f.finding,
                "details": f.details,
            }
            for f in findings
        ],
    }


def summary_line(doc: dict[str, Any]) -> str:
    return " ".join(f"{state} {doc['summary'].get(state, 0)}" for state in STATES)


# --- blocks ------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:+.1f}" if value else "0.0"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value) if value else "-"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_cell(v)}" for k, v in value.items()) if value else "-"
    return str(value)


def _detail_blocks(details: dict[str, Any]) -> list[Block]:
    blocks: list[Block] = []
    for key, value in details.items():
        title = key.replace("_", " ")
        if isinstance(value, list):
            if not value:
                blocks.append(("p", f"{title}: none"))
            elif all(isinstance(v, dict) for v in value):
                headers: list[str] = []
                for row in value:
                    for k in row:
                        if k not in headers:
                            headers.append(k)
                blocks.append(("p", f"{title} ({len(value)})"))
                blocks.append(
                    (
                        "table",
                        headers,
                        [[_cell(row.get(h)) for h in headers] for row in value],
                    )
                )
            else:
                blocks.append(("p", f"{title}: {_cell(value)}"))
        elif isinstance(value, dict):
            if value and all(isinstance(v, dict) for v in value.values()):
                headers = ["key"]
                for row in value.values():
                    for k in row:
                        if k not in headers:
                            headers.append(k)
                blocks.append(("p", title))
                blocks.append(
                    (
                        "table",
                        headers,
                        [
                            [name] + [_cell(row.get(h)) for h in headers[1:]]
                            for name, row in value.items()
                        ],
                    )
                )
            elif value:
                blocks.append(
                    (
                        "table",
                        [title, "value"],
                        [[k, _cell(v)] for k, v in value.items()],
                    )
                )
            else:
                blocks.append(("p", f"{title}: none"))
        else:
            blocks.append(("p", f"{title}: {_cell(value)}"))
    return blocks


def _document_blocks(doc: dict[str, Any]) -> list[Block]:
    cand, ref = doc["candidate"], doc["reference"]
    blocks: list[Block] = [
        ("h1", f"NIXL wheel inspection: {cand['label']} vs {ref['label']}"),
        (
            "p",
            f"Generated {doc['generated_at']} by {doc['tool'].get('name', 'wheel_inspect')} {doc['tool'].get('version', '?')}. {summary_line(doc)}.",
        ),
        (
            "table",
            ["side", "label", "version", "wheels", "manifest", "source"],
            [
                [
                    "candidate",
                    cand["label"],
                    cand["version"],
                    str(cand["wheels"]),
                    _cell(cand["manifest"]),
                    cand["dir"],
                ],
                [
                    "reference",
                    ref["label"],
                    ref["version"],
                    str(ref["wheels"]),
                    _cell(ref["manifest"]),
                    ref["dir"],
                ],
            ],
        ),
        ("h2", "Results"),
        (
            "table",
            ["#", "Check", "State", "Finding"],
            [
                [str(c["id"]), c["name"], c["state"], c["finding"]]
                for c in doc["checks"]
            ],
        ),
    ]
    for c in doc["checks"]:
        blocks.append(("h2", f"{c['id']}. {c['name']} — {c['state']}"))
        blocks.append(("p", c["finding"]))
        blocks.extend(_detail_blocks(c["details"]))
    for name, side in (("candidate", cand), ("reference", ref)):
        blocks.append(("h2", f"Wheel checksums: {name} {side['label']}"))
        blocks.append(
            ("table", ["wheel", "sha256"], [[k, v] for k, v in side["sha256"].items()])
        )
    return blocks


def render_markdown(doc: dict[str, Any]) -> str:
    out: list[str] = []
    for block in _document_blocks(doc):
        kind = block[0]
        if kind == "h1":
            out.append(f"# {block[1]}\n")
        elif kind == "h2":
            out.append(f"## {block[1]}\n")
        elif kind == "p":
            out.append(f"{block[1]}\n")
        else:
            _, headers, rows = block
            esc = [h.replace("|", "\\|") for h in headers]
            out.append("| " + " | ".join(esc) + " |")
            out.append("|" + "|".join(" --- " for _ in headers) + "|")
            for row in rows:
                out.append(
                    "| "
                    + " | ".join(
                        str(v).replace("|", "\\|").replace("\n", " ") for v in row
                    )
                    + " |"
                )
            out.append("")
    return "\n".join(out)


_CSS = (
    "body{font-family:sans-serif;font-size:14px;max-width:1200px;margin:16px auto}"
    "table{border-collapse:collapse;margin:8px 0}"
    "td,th{border:1px solid #bbb;padding:3px 8px;text-align:left;vertical-align:top;font-size:13px}"
    "th{background:#eee}"
    ".PASS{background:#d4edda}.FAIL{background:#f8d7da;font-weight:bold}.REVIEW{background:#fff3cd}.SKIPPED{background:#e2e3e5}"
)


def render_html(doc: dict[str, Any]) -> str:
    out: list[str] = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        f"<title>{html.escape(doc['candidate']['label'])} vs {html.escape(doc['reference']['label'])}</title>",
        f"<style>{_CSS}</style></head><body>",
    ]
    for block in _document_blocks(doc):
        kind = block[0]
        if kind in ("h1", "h2"):
            out.append(f"<{kind}>{html.escape(block[1])}</{kind}>")
        elif kind == "p":
            out.append(f"<p>{html.escape(block[1])}</p>")
        else:
            _, headers, rows = block
            out.append(
                "<table><tr>"
                + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
                + "</tr>"
            )
            for row in rows:
                cells = []
                for h, v in zip(headers, row):
                    text = html.escape(str(v))
                    css = f' class="{text}"' if h == "State" and text in STATES else ""
                    cells.append(f"<td{css}>{text}</td>")
                out.append("<tr>" + "".join(cells) + "</tr>")
            out.append("</table>")
    out.append("</body></html>")
    return "\n".join(out)


def write_reports(doc: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{REPORT_BASENAME}.json").write_text(json.dumps(doc, indent=2) + "\n")
    (out_dir / f"{REPORT_BASENAME}.md").write_text(render_markdown(doc))
    (out_dir / f"{REPORT_BASENAME}.html").write_text(render_html(doc))


def results_table(doc: dict[str, Any]) -> str:
    rows = [
        f"| {c['id']} | {c['name']} | {c['state']} | {c['finding']} |"
        for c in doc["checks"]
    ]
    return "\n".join(
        ["| # | Check | State | Finding |", "| --- | --- | --- | --- |", *rows]
    )


def load_document(path: Path) -> Optional[dict[str, Any]]:
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None
