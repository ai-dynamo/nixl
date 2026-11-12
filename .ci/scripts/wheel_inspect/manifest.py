"""build-manifest.json, the provenance record the nightly wheel build publishes next to a wheel set.

Written by ``.ci/scripts/merge_build_manifest.py``. Schema 1 keys ``base_image`` by CUDA major;
schema 2 (one CUDA line) carries a single flat ``base_image``. Wheels are matched by file name,
so nothing here depends on the per-wheel axis fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "build-manifest.json"
SCHEMAS = (1, 2)


class ManifestError(Exception):
    pass


def load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ManifestError(f"not valid JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise ManifestError("not a JSON object")
    if doc.get("schema_version") not in SCHEMAS:
        raise ManifestError(
            f"schema_version {doc.get('schema_version')!r} is not supported (known: {', '.join(map(str, SCHEMAS))})"
        )
    if doc.get("complete") is not True:
        errors = doc.get("errors") or ["no reason recorded"]
        raise ManifestError("marked incomplete by the build: " + "; ".join(errors))
    if not wheel_entries(doc):
        raise ManifestError("lists no wheels")
    return doc


def block(doc: dict[str, Any], name: str) -> dict[str, Any]:
    value = doc.get(name)
    return value if isinstance(value, dict) else {}


def wheel_entries(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """File name -> manifest entry, binary and meta wheels alike."""
    wheels = doc.get("wheels")
    if not isinstance(wheels, list):
        return {}
    return {str(w["name"]): w for w in wheels if isinstance(w, dict) and w.get("name")}


def mismatches(doc: dict[str, Any], sha256: dict[str, str]) -> list[str]:
    """Why this manifest does not describe the wheel files at hand; empty when it does."""
    entries = wheel_entries(doc)
    problems = []
    unlisted = sorted(set(sha256) - set(entries))
    absent = sorted(set(entries) - set(sha256))
    differ = sorted(
        name
        for name in set(sha256) & set(entries)
        if entries[name].get("sha256") != sha256[name]
    )
    if differ:
        problems.append(
            f"sha256 differs for {len(differ)} wheel(s): {', '.join(differ)}"
        )
    if unlisted:
        problems.append(f"wheels not in the manifest: {', '.join(unlisted)}")
    if absent:
        problems.append(f"manifest wheels not in the set: {', '.join(absent)}")
    return problems


def base_images(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """CUDA variant ('cu13') -> its base_image block, for either schema."""
    raw = block(doc, "base_image")
    if doc.get("schema_version") == 1:
        return {
            f"cu{major}": entry
            for major, entry in sorted(raw.items())
            if isinstance(entry, dict) and any(entry.values())
        }
    variants = sorted(
        {
            name.split("-", 1)[0][len("nixl_") :]
            for name in wheel_entries(doc)
            if name.startswith("nixl_cu")
        }
    )
    return {variants[0]: raw} if len(variants) == 1 and raw else {}
