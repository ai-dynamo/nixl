#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract selected Doxygen XML members as reusable Fern Markdown snippets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
import xml.etree.ElementTree as ET


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def normalized_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def load_members(xml_dir: Path) -> dict[tuple[str, str, str], list[ET.Element]]:
    members: dict[tuple[str, str, str], list[ET.Element]] = {}
    for xml_path in sorted(xml_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        for compound in root.findall("./compounddef"):
            compound_name = normalized_text(compound.find("compoundname"))
            for member in compound.findall(".//memberdef"):
                kind = member.get("kind")
                name = normalized_text(member.find("name"))
                if not compound_name or not kind or not name:
                    continue
                key = (compound_name, kind, name)
                members.setdefault(key, []).append(member)
    return members


def code_text(element: ET.Element) -> str:
    parts = [element.text or ""]
    for child in element:
        parts.append(" " if child.tag == "sp" else code_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def render_programlisting(member: ET.Element) -> str:
    listing = member.find("./detaileddescription//programlisting")
    if listing is None:
        fail(f"Doxygen member has no example code: {normalized_text(member.find('name'))}")

    lines = []
    for code_line in listing.findall("codeline"):
        lines.append(code_text(code_line).rstrip())
    code = textwrap.dedent("\n".join(lines)).strip("\n")
    if not code:
        fail(f"Doxygen member has an empty example: {normalized_text(member.find('name'))}")
    return code


def render_enum(member: ET.Element) -> str:
    details = member.find("./detaileddescription/para")
    description = " ".join((details.text or "").split()) if details is not None else ""
    if not description:
        description = normalized_text(member.find("briefdescription"))
    if not description:
        fail(f"Doxygen enum has no description: {normalized_text(member.find('name'))}")

    rows = []
    for enum_value in member.findall("enumvalue"):
        name = normalized_text(enum_value.find("name"))
        value_description = normalized_text(enum_value.find("briefdescription"))
        if not value_description:
            value_description = normalized_text(enum_value.find("detaileddescription"))
        if not name or not value_description:
            fail(f"Doxygen enum value is undocumented in {normalized_text(member.find('name'))}")
        rows.append(f"| `{name}` | {value_description} |")

    if not rows:
        fail(f"Doxygen enum has no values: {normalized_text(member.find('name'))}")

    code = render_programlisting(member)
    return (
        f"{description}\n\n"
        "| Value | Description |\n"
        "|-------|-------------|\n"
        + "\n".join(rows)
        + f"\n\n```cpp\n{code}\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    snippets = manifest.get("snippets")
    if not isinstance(snippets, list) or not snippets:
        fail("snippet manifest contains no snippets")

    members = load_members(args.xml_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs: set[str] = set()

    renderers = {"enum": render_enum}
    for entry in snippets:
        compound = entry.get("compound")
        kind = entry.get("kind")
        member_name = entry.get("member")
        output = entry.get("output")
        if (
            not isinstance(compound, str)
            or kind not in renderers
            or not isinstance(member_name, str)
            or not isinstance(output, str)
        ):
            fail("invalid snippet manifest entry")

        output_path = Path(output)
        if output_path.name != output or output_path.suffix != ".mdx":
            fail(f"snippet output must be a plain .mdx filename: {output}")
        if output in expected_outputs:
            fail(f"duplicate snippet output: {output}")
        expected_outputs.add(output)

        matching_members = members.get((compound, kind, member_name), [])
        if not matching_members:
            fail(f"Doxygen member was not found: {compound}: {kind} {member_name}")
        if len(matching_members) != 1:
            fail(f"Doxygen member is ambiguous: {compound}: {kind} {member_name}")
        content = renderers[kind](matching_members[0])
        (args.output_dir / output).write_text(content, encoding="utf-8")

    actual_outputs = {path.name for path in args.output_dir.glob("*.mdx")}
    if actual_outputs != expected_outputs:
        fail("snippet output directory contains unexpected files")


if __name__ == "__main__":
    main()
