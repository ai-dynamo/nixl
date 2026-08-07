#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly RUST_TOOLCHAIN_DEFAULT="nightly-2025-09-04"
readonly CARGO_DOC_MD_VERSION="0.11.0"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
manifest="${repo_root}/src/bindings/rust/Cargo.toml"
output_parent="${repo_root}/fern/docs/generated"
output_dir="${output_parent}/rust"
tool_cache="${NIXL_DOC_TOOLS_CACHE:-/tmp/nixl-doc-tools}"
rust_toolchain="${NIXL_RUSTDOC_TOOLCHAIN:-${RUST_TOOLCHAIN_DEFAULT}}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nixl-rust-docs.XXXXXX")"
stage_parent=""

cleanup() {
    rm -rf -- "${work_dir}"
    if [[ -n "${stage_parent}" && -d "${stage_parent}" ]]; then
        rm -rf -- "${stage_parent}"
    fi
}
trap cleanup EXIT

fail() {
    echo "error: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command '$1' was not found"
}

require_command git
require_command cargo
require_command rustup
require_command clang
require_command g++
require_command mktemp
require_command python3

[[ -s "${manifest}" ]] || fail "Rust crate manifest was not found: ${manifest}"

if ! rustup toolchain list | grep -q "^${rust_toolchain}-"; then
    echo "Installing Rust toolchain ${rust_toolchain}..."
    rustup toolchain install "${rust_toolchain}" --profile minimal
fi

if [[ -n "${CARGO_DOC_MD_BIN:-}" ]]; then
    cargo_doc_md_bin="${CARGO_DOC_MD_BIN}"
elif command -v cargo-doc-md >/dev/null 2>&1; then
    cargo_doc_md_bin="$(command -v cargo-doc-md)"
else
    cargo_doc_md_root="${tool_cache}/cargo-doc-md-${CARGO_DOC_MD_VERSION}"
    cargo_doc_md_bin="${cargo_doc_md_root}/bin/cargo-doc-md"
    if [[ ! -x "${cargo_doc_md_bin}" ]]; then
        echo "Installing cargo-doc-md v${CARGO_DOC_MD_VERSION}..."
        cargo "+${rust_toolchain}" install cargo-doc-md \
            --version "${CARGO_DOC_MD_VERSION}" \
            --locked \
            --root "${cargo_doc_md_root}"
    fi
fi

[[ -x "${cargo_doc_md_bin}" ]] || fail "cargo-doc-md is not executable: ${cargo_doc_md_bin}"

cargo_target="${work_dir}/target"
converted_output="${work_dir}/markdown"
organized_output="${work_dir}/organized"

echo "Generating rustdoc JSON for nixl-sys..."
(
    cd "${repo_root}"
    CARGO_TARGET_DIR="${cargo_target}" cargo "+${rust_toolchain}" rustdoc \
        --package nixl-sys \
        --features stub-api \
        --lib \
        -- \
        --output-format=json \
        -Z unstable-options
)

rustdoc_json="${cargo_target}/doc/nixl_sys.json"
[[ -s "${rustdoc_json}" ]] || fail "rustdoc did not generate the expected JSON file"

echo "Converting rustdoc JSON to module-oriented Markdown..."
"${cargo_doc_md_bin}" \
    --json "${rustdoc_json}" \
    --output "${converted_output}"

generated_crate="${converted_output}/nixl_sys"
[[ -d "${generated_crate}" ]] || fail "cargo-doc-md did not generate the nixl_sys documentation"

echo "Applying Fern compatibility fixes..."
python3 - "${generated_crate}" "${organized_output}" <<'PYTHON'
from pathlib import Path
import re
import sys

source_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
route_root = "/nixl/api-reference-generated/rust"
tick = chr(96)


def make_mdx_compatible(content: str) -> str:
    output = []
    in_fence = False
    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith(tick * 3):
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence:
            output.append(line)
            continue

        segments = line.split(tick)
        for index in range(0, len(segments), 2):
            marker = "__NIXL_LINE_BREAK__"
            segment = segments[index].replace("<br>", marker).replace("<br />", marker)
            segment = segment.replace("<", "&lt;").replace(">", "&gt;")
            segments[index] = segment.replace(marker, "<br />")
        output.append(tick.join(segments))
    return "".join(output)


source_pages = [
    path for path in sorted(source_dir.rglob("*.md"))
    if path.relative_to(source_dir) != Path("bindings.md")
]
destinations = {}
for source_path in source_pages:
    relative = source_path.relative_to(source_dir)
    fern_relative = Path(*(part.replace("_", "-") for part in relative.parts))
    module_dir = source_dir / relative.with_suffix("")
    destination = fern_relative.with_suffix("") / "index.md" if module_dir.is_dir() else fern_relative
    destinations[relative.as_posix()] = destination

link_pattern = re.compile(r"\]\(([^)#]+\.md)(#[^)]*)?\)")
for source_path in source_pages:
    relative = source_path.relative_to(source_dir)
    destination = destinations[relative.as_posix()]
    content = source_path.read_text(encoding="utf-8")

    if relative == Path("index.md"):
        content = re.sub(
            r"\n### \[`bindings`\]\(bindings\.md\)\n\n\*[^\n]+\*\n",
            "\n",
            content,
        )
        title = "Rust API Reference"
    else:
        heading = re.search(r"(?m)^# Module: (.+)$", content)
        if heading is None:
            raise RuntimeError(f"module heading is missing from {source_path}")
        title = heading.group(1)
        content = content[:heading.start()] + content[heading.end():]

    def replace_link(match: re.Match[str]) -> str:
        target = (relative.parent / match.group(1)).as_posix()
        target_destination = destinations.get(target)
        if target_destination is None:
            raise RuntimeError(f"unresolved generated link in {source_path}: {target}")
        route_parts = target_destination.with_suffix("").parts
        if route_parts[-1] == "index":
            route_parts = route_parts[:-1]
        route_parts = tuple(part.replace("_", "-") for part in route_parts)
        route = "/".join((route_root.rstrip("/"), *route_parts))
        return f"]({route}{match.group(2) or ''})"

    content = link_pattern.sub(replace_link, content)
    content = re.sub(r"(?m)^## nixl_sys(?:::[^\n:]+)*::([^:\n]+)$", r"## \1", content)
    body = make_mdx_compatible(content.strip())
    destination_path = output_dir / destination
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        f"---\ntitle: {title}\n---\n\n{body}\n",
        encoding="utf-8",
    )
PYTHON

grep -Fq "## Agent" "${organized_output}/agent.md" ||
    fail "generated Agent API is missing"
grep -Fq "fn new(name: &str)" "${organized_output}/agent.md" ||
    fail "generated Agent methods are missing"
grep -Fq "## MemType" "${organized_output}/descriptors/index.md" ||
    fail "generated MemType API is missing"
grep -Fq "## XferRequest" "${organized_output}/xfer.md" ||
    fail "generated XferRequest API is missing"
if [[ -e "${organized_output}/bindings.md" ]]; then
    fail "private raw FFI bindings leaked into the generated documentation"
fi

mkdir -p -- "${output_parent}"
stage_parent="$(mktemp -d "${output_parent}/.rust.generated.XXXXXX")"
stage_dir="${stage_parent}/rust"
mv -- "${organized_output}" "${stage_dir}"

expected_output_dir="${repo_root}/fern/docs/generated/rust"
[[ "${output_dir}" == "${expected_output_dir}" ]] ||
    fail "refusing to replace unexpected output directory: ${output_dir}"

rm -rf -- "${output_dir}"
mv -- "${stage_dir}" "${output_dir}"
rmdir -- "${stage_parent}"
stage_parent=""

echo "Generated Rust API Markdown in ${output_dir}"
