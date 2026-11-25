"""What each side was built from, as its build manifest records it, for the report."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from wheel_inspect import manifest as manifest_mod
from wheel_inspect.checks import Side, fold_scope
from wheel_inspect.wheelset import Cell

# Always differ between two builds, so never flagged as changed.
_BUILD_ITEMS = ("Built by", "Published to", "Manifest")


def _join(*parts: Any, sep: str = ", ") -> Optional[str]:
    return sep.join(str(p) for p in parts if p) or None


def _facts(doc: dict[str, Any]) -> dict[str, Optional[str]]:
    nixl, ucx, spcx, infinia, jenkins = (
        manifest_mod.block(doc, name)
        for name in ("nixl", "ucx", "spcx", "infinia", "jenkins")
    )
    facts: dict[str, Optional[str]] = {
        "NIXL commit": nixl.get("commit"),
        "NIXL ref built": nixl.get("ref"),
        "UCX repo": ucx.get("repo"),
        "UCX declared": _join(
            ucx.get("declared_ref"), ucx.get("declared_commit"), sep=" @ "
        ),
        "UCX in the image": _join(
            ucx.get("release"), ucx.get("branch"), ucx.get("revision")
        ),
        "SPCX plugin": (
            _join(spcx.get("ref"), spcx.get("commit"), sep=" @ ")
            if spcx.get("enabled")
            else "not built"
        ),
        "Infinia libs": (
            _join(infinia.get("image"), infinia.get("digest"), sep=" @ ")
            if infinia.get("enabled")
            else "not built"
        ),
    }
    for variant, entry in manifest_mod.base_images(doc).items():
        facts[f"Base image {variant}"] = _join(
            _join(entry.get("image"), entry.get("tag"), sep=":"),
            f"CUDA {entry['cuda_version']}" if entry.get("cuda_version") else None,
        )
        facts[f"Torch requested {variant}"] = _join(
            *(entry.get("torch_requested") or [])
        )
    facts["Built by"] = _join(
        _join(jenkins.get("job"), jenkins.get("build"), sep=" #"),
        f"ci_refspec {jenkins['ci_refspec']}" if jenkins.get("ci_refspec") else None,
    )
    facts["Published to"] = jenkins.get("folder")
    facts["Manifest"] = _join(
        f"schema {doc.get('schema_version')}", f"generated {doc.get('generated_at')}"
    )
    return facts


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(p) if p.isdigit() else 0 for p in version.split("."))


def _torch_rows(side: Side) -> Optional[list[dict[str, str]]]:
    """Torch versions built into, or skipped for, the binary wheels, folded by cell."""
    if not side.manifest:
        return None
    cells = {wf.name: cell for cell, wf in side.wheelset.binary.items()}
    groups: dict[tuple[str, str, str], set[Cell]] = defaultdict(set)
    for name, entry in manifest_mod.wheel_entries(side.manifest).items():
        torch = entry.get("torch")
        if name not in cells or not isinstance(torch, dict):
            continue
        for built in torch.get("built") or []:
            detail = _join(built.get("channel"), built.get("resolved"), sep=" ") or ""
            groups[(str(built.get("version")), "built", detail)].add(cells[name])
        for skipped in torch.get("skipped") or []:
            groups[
                (
                    str(skipped.get("version")),
                    "skipped",
                    str(skipped.get("reason") or ""),
                )
            ].add(cells[name])
    universe = side.binary_cells
    return [
        {
            "torch": version,
            "outcome": outcome,
            "detail": detail,
            "scope": fold_scope(found, universe),
        }
        for (version, outcome, detail), found in sorted(
            groups.items(), key=lambda kv: (_version_key(kv[0][0]), kv[0][1:])
        )
    ]


def _toolchain(doc: Optional[dict[str, Any]]) -> dict[str, Optional[str]]:
    return {
        tool: _join(*values) if isinstance(values, list) else _join(values)
        for tool, values in manifest_mod.block(doc or {}, "toolchain").items()
    }


def _rows(
    key: str, cand: dict[str, Optional[str]], ref: dict[str, Optional[str]], both: bool
) -> list[dict[str, Any]]:
    rows = []
    for name in list(cand) + [n for n in ref if n not in cand]:
        c, r = cand.get(name), ref.get(name)
        rows.append(
            {
                key: name,
                "candidate": c,
                "reference": r,
                "changed": (c != r) if both and name not in _BUILD_ITEMS else None,
            }
        )
    return rows


def build(cand: Side, ref: Side) -> Optional[dict[str, Any]]:
    """The report's provenance block; None when neither side has a usable manifest."""
    if not cand.manifest and not ref.manifest:
        return None
    both = bool(cand.manifest and ref.manifest)
    return {
        "facts": _rows(
            "item",
            _facts(cand.manifest) if cand.manifest else {},
            _facts(ref.manifest) if ref.manifest else {},
            both,
        ),
        "toolchain": _rows(
            "tool", _toolchain(cand.manifest), _toolchain(ref.manifest), both
        ),
        "torch": {"candidate": _torch_rows(cand), "reference": _torch_rows(ref)},
    }
