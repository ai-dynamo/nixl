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

"""The eight inspection checks, each producing a Finding with scope-folded details."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from wheel_inspect.buildinfo import WheelFacts, wheel_facts
from wheel_inspect.elf import parse_elf
from wheel_inspect.inventory import Member, diff_pair, inventory
from wheel_inspect.wheelset import (
    ARCH_ORDER,
    CUDA_ORDER,
    Cell,
    WheelSet,
    load_wheelset,
    py_sort_key,
    sort_cells,
)

STATES = ("PASS", "FAIL", "REVIEW", "SKIPPED")
CHECK_NAMES = {
    1: "SPCX plugin bundling",
    2: "DDN INFINIA bundling",
    3: "SPCX built from the expected ref",
    4: "UCX version and commit",
    5: "Files added / removed / changed",
    6: "NIXL plugins bundled",
    7: "UCX plugins bundled",
    8: "Compressed wheel size",
}
SPCX_MODULES = ("libuct_ib_mlx5_ext.so.0.0.0", "libuct_ib_mlx5_gdp.so.0.0.0")
INFINIA_PLUGIN = "libplugin_INFINIA.so"
BIG_DELTA_BYTES = 1_000_000

_UCX_MODULE = re.compile(r"^nixl_cu\d+\.libs/ucx/([^/]+\.so\.0\.0\.0)$")
_NIXL_PLUGIN = re.compile(r"^nixl_cu\d+\.libs/nixl/([^/]+\.so)$")
_RED_LIB = re.compile(r"^nixl_cu\d+\.libs/libred_[^/]*$")
_MESONPY = re.compile(r"^\.nixl_cu\d+\.mesonpy\.libs/((?:plugins/)?[^/]+)$")
_EP_MODULE = re.compile(r"^nixl_ep_cu\d+/.+\.so$")
_SM_ARCH = re.compile(r"sm_\d+[af]?")


@dataclass
class Finding:
    id: int
    name: str
    state: str
    finding: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Side:
    label: str
    dir: Path
    wheelset: WheelSet
    inventories: dict[Cell, dict[str, Member]]
    facts: dict[Cell, WheelFacts]
    manifest: Optional[dict[str, Any]]

    @property
    def binary_cells(self) -> list[Cell]:
        return sort_cells(self.wheelset.binary)

    def member_bytes(self, cell: Cell, path: str) -> bytes:
        with zipfile.ZipFile(self.wheelset.wheels[cell].path) as zf:
            return zf.read(path)


def load_side(
    directory: Path, manifest: Optional[Path] = None, label: Optional[str] = None
) -> Side:
    directory = Path(directory)
    wheelset = load_wheelset(directory)
    inventories = {cell: inventory(wf.path) for cell, wf in wheelset.wheels.items()}
    facts = {
        cell: wheel_facts(wf.path, wf.package, wf.version)
        for cell, wf in wheelset.wheels.items()
    }
    manifest_path = Path(manifest) if manifest else directory / "build-manifest.json"
    manifest_doc: Optional[dict[str, Any]] = None
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text())
            manifest_doc = loaded if isinstance(loaded, dict) else None
        except ValueError:
            manifest_doc = None
    return Side(
        label or wheelset.version, directory, wheelset, inventories, facts, manifest_doc
    )


# --- scope folding -------------------------------------------------------------


def _order(order: tuple[str, ...], value: str) -> tuple[int, str]:
    return (order.index(value) if value in order else len(order), value)


def fold_scope(cells: Iterable[Cell], universe: Iterable[Cell]) -> str:
    """Compact rendering of the cells a finding applies to, the way the 1.4.1 audit did it."""
    chosen = set(cells)
    everything = [c for c in universe if not c.is_meta]
    has_meta = Cell.META in chosen
    chosen.discard(Cell.META)
    if not chosen and not has_meta:
        return "none"
    if not chosen:
        return "meta"
    if chosen >= set(everything):
        return "all"
    by_cuda: dict[str, set[tuple[str, str]]] = defaultdict(set)
    full: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for c in chosen:
        by_cuda[c.cuda].add((c.py, c.arch))
    for c in everything:
        full[c.cuda].add((c.py, c.arch))
    parts = []
    for cuda in sorted(by_cuda, key=lambda v: _order(CUDA_ORDER, v)):
        group = by_cuda[cuda]
        if group == full[cuda]:
            parts.append(f"{cuda}: all")
            continue
        pys = sorted({p for p, _ in group}, key=py_sort_key)
        arches = sorted({a for _, a in group}, key=lambda v: _order(ARCH_ORDER, v))
        if len(pys) * len(arches) == len(group):
            parts.append(f"{cuda}: {'/'.join(pys)} x {'/'.join(arches)}")
        else:
            ordered = sorted(
                group, key=lambda pa: (py_sort_key(pa[0]), _order(ARCH_ORDER, pa[1]))
            )
            parts.append(f"{cuda}: " + ", ".join(f"{p}-{a}" for p, a in ordered))
    text = "; ".join(parts)
    return f"{text}; meta" if has_meta else text


# --- helpers -------------------------------------------------------------------


def _modules(inv: dict[str, Member], pattern: re.Pattern[str]) -> dict[str, Member]:
    out = {}
    for key, member in inv.items():
        match = pattern.match(key)
        if match:
            out[match.group(1)] = member
    return out


def _module_sets(side: Side, pattern: re.Pattern[str]) -> dict[Cell, dict[str, Member]]:
    return {
        cell: _modules(side.inventories[cell], pattern) for cell in side.binary_cells
    }


def _mib(n: float) -> float:
    return n / (1 << 20)


def _finding(id_: int, state: str, text: str, details: dict[str, Any]) -> Finding:
    return Finding(id_, CHECK_NAMES[id_], state, text, details)


def _presence(
    sets: dict[Cell, dict[str, Member]], universe: list[Cell], modules: Iterable[str]
) -> dict[str, str]:
    return {
        m: fold_scope([c for c in universe if m in sets[c]], universe) for m in modules
    }


# --- checks --------------------------------------------------------------------


def check_spcx(cand: Side, ref: Side) -> Finding:
    universe = cand.binary_cells
    sets = _module_sets(cand, _UCX_MODULE)
    ref_sets = _module_sets(ref, _UCX_MODULE)
    missing = []
    for module in SPCX_MODULES:
        absent = [c for c in universe if module not in sets[c]]
        if absent:
            missing.append({"module": module, "scope": fold_scope(absent, universe)})
    details = {
        "present": _presence(sets, universe, SPCX_MODULES),
        "missing": missing,
        "reference": _presence(ref_sets, ref.binary_cells, SPCX_MODULES),
    }
    if missing:
        text = "; ".join(f"{m['module']} missing in {m['scope']}" for m in missing)
        return _finding(1, "FAIL", text, details)
    return _finding(
        1,
        "PASS",
        f"{' and '.join(SPCX_MODULES)} in all {len(universe)} binary wheels",
        details,
    )


def check_infinia(cand: Side, ref: Side) -> Finding:
    universe = cand.binary_cells
    sets = _module_sets(cand, _NIXL_PLUGIN)
    missing = [c for c in universe if INFINIA_PLUGIN not in sets[c]]
    unreadable, no_ddn = [], []
    needed: set[str] = set()
    runpaths: set[str] = set()
    for cell in universe:
        member = sets[cell].get(INFINIA_PLUGIN)
        if member is None:
            continue
        info = parse_elf(cand.member_bytes(cell, member.path))
        if info is None:
            unreadable.append(cell)
            continue
        needed.update(info.needed)
        if info.runpath or info.rpath:
            runpaths.add(info.runpath or info.rpath or "")
        if not any(n.startswith("libred_client.so") for n in info.needed) or not any(
            n.startswith("libred_async.so") for n in info.needed
        ):
            no_ddn.append(cell)
    bundled: dict[str, set[Cell]] = defaultdict(set)
    for cell in universe:
        for key in cand.inventories[cell]:
            if _RED_LIB.match(key):
                bundled[key].add(cell)
    ref_sets = _module_sets(ref, _NIXL_PLUGIN)
    details = {
        "present": fold_scope([c for c in universe if c not in missing], universe),
        "missing": fold_scope(missing, universe) if missing else None,
        "unreadable": fold_scope(unreadable, universe) if unreadable else None,
        "not_linking_ddn": fold_scope(no_ddn, universe) if no_ddn else None,
        "needed": sorted(needed),
        "runpath": "; ".join(sorted(runpaths)) or None,
        "bundled_red_libs": [
            {"key": k, "scope": fold_scope(cells, universe)}
            for k, cells in sorted(bundled.items())
        ],
        "reference": {
            "present": fold_scope(
                [c for c in ref.binary_cells if INFINIA_PLUGIN in ref_sets[c]],
                ref.binary_cells,
            )
        },
    }
    problems = []
    if missing:
        problems.append(f"{INFINIA_PLUGIN} missing in {details['missing']}")
    if unreadable:
        problems.append(
            f"{INFINIA_PLUGIN} not a readable ELF in {details['unreadable']}"
        )
    if no_ddn:
        problems.append(
            f"{INFINIA_PLUGIN} does not link libred_client/libred_async in {details['not_linking_ddn']}"
        )
    if bundled:
        problems.append("DDN runtime libraries bundled: " + ", ".join(sorted(bundled)))
    if problems:
        return _finding(2, "FAIL", "; ".join(problems), details)
    return _finding(
        2,
        "PASS",
        f"{INFINIA_PLUGIN} in all {len(universe)} binary wheels; DDN runtime libraries left external",
        details,
    )


def _spcx_block(side: Side) -> Optional[dict[str, Any]]:
    if not side.manifest:
        return None
    block = side.manifest.get("spcx") or {}
    return {
        "enabled": bool(block.get("enabled")),
        "ref": block.get("ref"),
        "commit": block.get("commit"),
    }


def check_spcx_ref(cand: Side, ref: Side) -> Finding:
    sets = _module_sets(cand, _UCX_MODULE)
    bundled = all(all(m in sets[c] for m in SPCX_MODULES) for c in cand.binary_cells)
    cblock, rblock = _spcx_block(cand), _spcx_block(ref)
    details: dict[str, Any] = {
        "candidate": cblock,
        "reference": rblock,
        "changed": None,
        "bundled": bundled,
    }
    if cblock is None:
        return _finding(
            3,
            "SKIPPED",
            "no build manifest for the candidate; SPCX ref and commit not recorded",
            details,
        )
    if rblock is not None:
        details["changed"] = (cblock["ref"], cblock["commit"]) != (
            rblock["ref"],
            rblock["commit"],
        )
    if bundled and (not cblock["enabled"] or not cblock["commit"]):
        return _finding(
            3,
            "FAIL",
            f"SPCX modules are bundled but the manifest records enabled={cblock['enabled']}, commit={cblock['commit']}",
            details,
        )
    if cblock["enabled"] and cblock["ref"] and cblock["commit"]:
        text = f"SPCX ref {cblock['ref']}, commit {cblock['commit']}"
        if rblock is not None:
            text += (
                " (unchanged since reference)"
                if not details["changed"]
                else f" (reference: {rblock['ref']} @ {rblock['commit']})"
            )
        return _finding(3, "PASS", text, details)
    if not bundled and not cblock["enabled"]:
        return _finding(3, "PASS", "SPCX not bundled, and the manifest agrees", details)
    return _finding(3, "FAIL", f"manifest spcx block is incomplete: {cblock}", details)


def _ucx_summary(side: Side) -> dict[str, Any]:
    per_wheel = []
    readme_values: set[str] = set()
    scanned_values: set[str] = set()
    revision = branch = None
    disagree = []
    for cell in side.binary_cells:
        facts = side.facts[cell]
        readme = facts.build_info.ucx_release if facts.build_info else None
        scanned = facts.ucx_release_scanned
        per_wheel.append({"cell": cell.label, "readme": readme, "scanned": scanned})
        if readme:
            readme_values.add(readme)
        if scanned:
            scanned_values.add(scanned)
        if readme and scanned and readme != scanned:
            disagree.append(cell)
        if facts.build_info and not revision:
            revision, branch = (
                facts.build_info.ucx_revision,
                facts.build_info.ucx_branch,
            )
    manifest_ucx = (side.manifest or {}).get("ucx") or {}
    return {
        "per_wheel": per_wheel,
        "readme": readme_values,
        "scanned": scanned_values,
        "values": readme_values | scanned_values,
        "disagree": disagree,
        "revision": revision or manifest_ucx.get("revision"),
        "branch": branch or manifest_ucx.get("branch"),
        "manifest_release": manifest_ucx.get("release"),
        "declared_commit": manifest_ucx.get("declared_commit"),
        "declared_ref": manifest_ucx.get("declared_ref"),
    }


def check_ucx(cand: Side, ref: Side) -> Finding:
    c, r = _ucx_summary(cand), _ucx_summary(ref)

    def one(values: set[str]) -> Optional[str]:
        return next(iter(values)) if len(values) == 1 else None

    details = {
        "release_by_source": {
            "readme": one(c["readme"]),
            "scanned": one(c["scanned"]),
            "manifest": c["manifest_release"],
        },
        "per_wheel": c["per_wheel"],
        "revision": c["revision"],
        "branch": c["branch"],
        "declared_commit": c["declared_commit"],
        "declared_ref": c["declared_ref"],
        "reference": {"release": one(r["values"]), "revision": r["revision"]},
    }
    universe = cand.binary_cells
    if not c["values"]:
        return _finding(
            4,
            "SKIPPED",
            "UCX release string not readable from any wheel (no README block, no libucp string)",
            details,
        )
    if c["disagree"]:
        return _finding(
            4,
            "FAIL",
            f"README and libucp disagree on the UCX release in {fold_scope(c['disagree'], universe)}",
            details,
        )
    if len(c["values"]) > 1:
        return _finding(
            4,
            "FAIL",
            f"UCX release differs across wheels: {', '.join(sorted(c['values']))}",
            details,
        )
    release = one(c["values"])
    if c["manifest_release"] and c["manifest_release"] != release:
        return _finding(
            4,
            "FAIL",
            f"wheels carry UCX {release} but the manifest says {c['manifest_release']}",
            details,
        )
    text = f"UCX {release} in all {len(universe)} binary wheels"
    if c["revision"]:
        text += f", revision {c['revision']}" + (
            f" ({c['branch']})" if c["branch"] else ""
        )
    else:
        text += "; commit not recorded"
    ref_release = one(r["values"])
    if ref_release and ref_release != release:
        text += f"; reference {ref.label} had {ref_release}"
    return _finding(4, "PASS", text, details)


def _version_rows(cand: Side, ref: Side, shared: list[Cell]) -> list[dict[str, Any]]:
    rows = []

    def add(
        field_name: str,
        getter: Callable[[WheelFacts], Any],
        cells: list[Cell],
        member: str,
    ) -> None:
        pairs: dict[tuple[str, str], set[Cell]] = defaultdict(set)
        for cell in cells:
            rv, cv = getter(ref.facts[cell]), getter(cand.facts[cell])
            pairs[(str(rv), str(cv))].add(cell)
        for (rv, cv), cells_ in sorted(pairs.items()):
            rows.append(
                {
                    "member": member,
                    "field": field_name,
                    "reference": rv,
                    "candidate": cv,
                    "changed": rv != cv,
                    "scope": fold_scope(cells_, shared),
                }
            )

    add("Version", lambda f: f.metadata_version, shared, "dist-info/METADATA")
    add("Generator", lambda f: f.wheel_generator, shared, "dist-info/WHEEL")
    add("tools", lambda f: f.sbom_tools, shared, "sboms/auditwheel.cdx.json")
    add("UCX release", lambda f: f.ucx_release_scanned, shared, "libucp")
    if Cell.META in cand.wheelset.wheels and Cell.META in ref.wheelset.wheels:
        add(
            "Version",
            lambda f: f.metadata_version,
            [Cell.META],
            "meta dist-info/METADATA",
        )
        add(
            "Requires-Dist",
            lambda f: f.requires_dist,
            [Cell.META],
            "meta dist-info/METADATA",
        )
        add(
            "Generator",
            lambda f: f.wheel_generator,
            [Cell.META],
            "meta dist-info/WHEEL",
        )
    comp_changes: dict[tuple[str, str, str], set[Cell]] = defaultdict(set)
    for cell in shared:
        rc, cc = ref.facts[cell].sbom_components, cand.facts[cell].sbom_components
        for name in sorted(set(rc) | set(cc)):
            if rc.get(name) != cc.get(name):
                comp_changes[(name, str(rc.get(name)), str(cc.get(name)))].add(cell)
    for (name, rv, cv), cells_ in sorted(comp_changes.items()):
        rows.append(
            {
                "member": "sboms/auditwheel.cdx.json",
                "field": f"component {name}",
                "reference": rv,
                "candidate": cv,
                "changed": True,
                "scope": fold_scope(cells_, shared),
            }
        )
    return rows


def check_files(cand: Side, ref: Side) -> Finding:
    shared, only_c, only_r = cand.wheelset.pairs_with(ref.wheelset)
    pairs = list(shared)
    if Cell.META in cand.wheelset.wheels and Cell.META in ref.wheelset.wheels:
        pairs.append(Cell.META)
    found: dict[tuple[str, str, str], set[Cell]] = defaultdict(set)
    mesonpy: dict[str, dict[str, int]] = defaultdict(
        lambda: {"identical": 0, "changed": 0, "cells": 0}
    )
    for cell in pairs:
        diff = diff_pair(ref.inventories[cell], cand.inventories[cell])
        for key, m in diff.new.items():
            found[("new", key, m.path)].add(cell)
        for key, m in diff.deleted.items():
            found[("deleted", key, m.path)].add(cell)
        for key, (old, new) in diff.rehashed.items():
            found[
                (
                    "rehashed",
                    key,
                    f"{old.path.rsplit('/', 1)[-1]} -> {new.path.rsplit('/', 1)[-1]}",
                )
            ].add(cell)
        for key in diff.changed:
            found[("changed", key, "")].add(cell)
        if cell.is_meta:
            continue
        for key, member in cand.inventories[cell].items():
            match = _MESONPY.match(key)
            if not match:
                continue
            row = mesonpy[match.group(1)]
            row["cells"] += 1
            other = ref.inventories[cell].get(key)
            if other is not None and other.digest == member.digest:
                row["identical"] += 1
            else:
                row["changed"] += 1

    lists: dict[str, list[dict[str, str]]] = {
        "new": [],
        "deleted": [],
        "rehashed": [],
        "changed": [],
    }
    for (cls, key, detail), cells in sorted(found.items()):
        lists[cls].append(
            {"key": key, "detail": detail, "scope": fold_scope(cells, shared)}
        )
    counts = {cls: len(rows) for cls, rows in lists.items()}
    details: dict[str, Any] = {
        **lists,
        "counts": counts,
        "mesonpy": [{"module": m, **row} for m, row in sorted(mesonpy.items())],
        "versions": _version_rows(cand, ref, shared),
        "only_in_candidate": [c.label for c in only_c],
        "only_in_reference": [c.label for c in only_r],
        "shared_cells": len(shared),
    }
    text = (
        f"{counts['new']} new, {counts['deleted']} deleted, {counts['rehashed']} rehashed, "
        f"{counts['changed']} changed paths across {len(shared)} shared cells + meta"
    )
    if only_c or only_r:
        text += f"; cells only in candidate: {len(only_c)}, only in reference: {len(only_r)}"
    state = (
        "PASS" if not any(counts.values()) and not only_c and not only_r else "REVIEW"
    )
    if state == "PASS":
        text = f"identical member sets and contents across {len(shared)} shared cells + meta"
    return _finding(5, state, text, details)


def _plugin_check(
    id_: int, cand: Side, ref: Side, pattern: re.Pattern[str], what: str
) -> Finding:
    universe = cand.binary_cells
    sets = _module_sets(cand, pattern)
    ref_sets = _module_sets(ref, pattern)
    by_variant: dict[str, list[Cell]] = defaultdict(list)
    for cell in universe:
        by_variant[cell.cuda].append(cell)
    inconsistent = []
    per_variant_count = {}
    for cuda, cells in sorted(
        by_variant.items(), key=lambda kv: _order(CUDA_ORDER, kv[0])
    ):
        union = set().union(*(set(sets[c]) for c in cells))
        per_variant_count[cuda] = len(union)
        for module in sorted(union):
            absent = [c for c in cells if module not in sets[c]]
            if absent:
                inconsistent.append(
                    {
                        "cuda": cuda,
                        "module": module,
                        "missing_in": fold_scope(absent, universe),
                    }
                )
    cand_modules = sorted(
        set().union(*(set(s) for s in sets.values())) if sets else set()
    )
    ref_modules = (
        set().union(*(set(s) for s in ref_sets.values())) if ref_sets else set()
    )
    added = [
        {
            "module": m,
            "scope": fold_scope([c for c in universe if m in sets[c]], universe),
        }
        for m in cand_modules
        if m not in ref_modules
    ]
    removed = [
        {
            "module": m,
            "scope": fold_scope(
                [c for c in ref.binary_cells if m in ref_sets[c]], ref.binary_cells
            ),
        }
        for m in sorted(ref_modules)
        if m not in set(cand_modules)
    ]
    details = {
        "present": _presence(sets, universe, cand_modules),
        "per_variant_count": per_variant_count,
        "inconsistent": inconsistent,
        "added": added,
        "removed": removed,
        "reference_per_variant_count": {
            cuda: len(set().union(*(set(ref_sets[c]) for c in cells)))
            for cuda, cells in sorted(
                defaultdict(
                    list,
                    {
                        c.cuda: [x for x in ref.binary_cells if x.cuda == c.cuda]
                        for c in ref.binary_cells
                    },
                ).items()
            )
        },
    }
    counts = ", ".join(f"{cuda}: {n}" for cuda, n in per_variant_count.items())
    delta = f"; {len(added)} added, {len(removed)} removed since {ref.label}"
    if inconsistent:
        odd = "; ".join(
            f"{i['module']} missing in {i['missing_in']}" for i in inconsistent
        )
        return _finding(
            id_,
            "FAIL",
            f"{what} set differs within a CUDA variant: {odd}{delta}",
            details,
        )
    return _finding(
        id_,
        "PASS",
        f"{counts} {what} modules, identical across all wheels of each CUDA variant{delta}",
        details,
    )


def check_nixl_plugins(cand: Side, ref: Side) -> Finding:
    return _plugin_check(6, cand, ref, _NIXL_PLUGIN, "NIXL plugin")


def check_ucx_plugins(cand: Side, ref: Side) -> Finding:
    return _plugin_check(7, cand, ref, _UCX_MODULE, "UCX")


def _size_row(
    label: str, reference: int, candidate: int, **extra: Any
) -> dict[str, Any]:
    delta = candidate - reference
    pct = (100.0 * delta / reference) if reference else None
    return {
        "cell": label,
        "reference": reference,
        "candidate": candidate,
        "delta": delta,
        "pct": pct,
        **extra,
    }


def _fatbin_arches(cuobjdump: str, data: bytes, name: str) -> Optional[list[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / re.sub(r"[^A-Za-z0-9._-]", "_", name)
        path.write_bytes(data)
        try:
            proc = subprocess.run(
                [cuobjdump, "--list-elf", str(path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
    if proc.returncode != 0:
        return None
    return sorted(set(_SM_ARCH.findall(proc.stdout)))


def check_sizes(cand: Side, ref: Side, cuobjdump: Optional[str] = None) -> Finding:
    shared, _, _ = cand.wheelset.pairs_with(ref.wheelset)
    pairs = list(shared) + (
        [Cell.META]
        if Cell.META in cand.wheelset.wheels and Cell.META in ref.wheelset.wheels
        else []
    )
    per_cell = []
    groups: dict[tuple[str, str], list[Cell]] = defaultdict(list)
    for cell in pairs:
        rsize, csize = ref.wheelset.wheels[cell].size, cand.wheelset.wheels[cell].size
        per_cell.append(
            _size_row(
                cell.label, rsize, csize, cuda=cell.cuda, py=cell.py, arch=cell.arch
            )
        )
        groups[(cell.cuda, cell.arch)].append(cell)
    per_variant = []
    for (cuda, arch), cells in sorted(
        groups.items(),
        key=lambda kv: (_order(CUDA_ORDER, kv[0][0]), _order(ARCH_ORDER, kv[0][1])),
    ):
        rsum = sum(ref.wheelset.wheels[c].size for c in cells)
        csum = sum(cand.wheelset.wheels[c].size for c in cells)
        per_variant.append(
            _size_row(
                f"{cuda}/{arch}", rsum, csum, cuda=cuda, arch=arch, cells=len(cells)
            )
        )
    totals = _size_row(
        "all",
        sum(ref.wheelset.wheels[c].size for c in pairs),
        sum(cand.wheelset.wheels[c].size for c in pairs),
        cells=len(pairs),
    )
    uncompressed = []
    big: dict[str, dict[str, Any]] = {}
    for cell in pairs:
        rinv, cinv = ref.inventories[cell], cand.inventories[cell]
        uncompressed.append(
            _size_row(
                cell.label,
                sum(m.size for m in rinv.values()),
                sum(m.size for m in cinv.values()),
            )
        )
        for key in set(rinv) | set(cinv):
            rs = rinv[key].size if key in rinv else 0
            cs = cinv[key].size if key in cinv else 0
            if abs(cs - rs) >= BIG_DELTA_BYTES:
                entry = big.setdefault(
                    key,
                    {
                        "key": key,
                        "cells": set(),
                        "reference": set(),
                        "candidate": set(),
                    },
                )
                entry["cells"].add(cell)
                entry["reference"].add(rs)
                entry["candidate"].add(cs)
    big_deltas: list[dict[str, Any]] = [
        {
            "key": k,
            "scope": fold_scope(e["cells"], shared),
            "reference": sorted(e["reference"]),
            "candidate": sorted(e["candidate"]),
        }
        for k, e in sorted(big.items())
    ]
    tool = cuobjdump or shutil.which("cuobjdump")
    fatbin: Any
    if tool is None:
        fatbin = {"skipped": "cuobjdump not available"}
    else:
        fatbin = []
        for entry in big_deltas:
            key = str(entry["key"])
            if not _EP_MODULE.match(key):
                continue
            for cell in sorted(big[key]["cells"]):
                rm, cm = ref.inventories[cell].get(entry["key"]), cand.inventories[
                    cell
                ].get(entry["key"])
                fatbin.append(
                    {
                        "module": key,
                        "cell": cell.label,
                        "ref_arches": (
                            _fatbin_arches(
                                tool, ref.member_bytes(cell, rm.path), rm.path
                            )
                            if rm
                            else None
                        ),
                        "cand_arches": (
                            _fatbin_arches(
                                tool, cand.member_bytes(cell, cm.path), cm.path
                            )
                            if cm
                            else None
                        ),
                    }
                )
    largest = max(per_cell, key=lambda r: abs(r["delta"])) if per_cell else None
    text = (
        f"release total {_mib(totals['reference']):.1f} -> {_mib(totals['candidate']):.1f} MiB "
        f"({_mib(totals['delta']):+.1f} MiB, {totals['pct']:+.1f}%)"
        if totals["pct"] is not None
        else "no comparable wheels"
    )
    if largest:
        text += f"; largest change {largest['cell']} {_mib(largest['delta']):+.1f} MiB"
    details = {
        "per_cell": per_cell,
        "per_variant": per_variant,
        "totals": totals,
        "uncompressed_per_cell": uncompressed,
        "big_deltas": big_deltas,
        "fatbin": fatbin,
    }
    return _finding(8, "REVIEW", text, details)


def run_checks(cand: Side, ref: Side, cuobjdump: Optional[str] = None) -> list[Finding]:
    return [
        check_spcx(cand, ref),
        check_infinia(cand, ref),
        check_spcx_ref(cand, ref),
        check_ucx(cand, ref),
        check_files(cand, ref),
        check_nixl_plugins(cand, ref),
        check_ucx_plugins(cand, ref),
        check_sizes(cand, ref, cuobjdump),
    ]
