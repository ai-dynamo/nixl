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
stage_dir=""

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

echo "Converting rustdoc JSON to Markdown..."
"${cargo_doc_md_bin}" \
    --json "${rustdoc_json}" \
    --output "${converted_output}"

generated_crate="${converted_output}/nixl_sys"
[[ -d "${generated_crate}" ]] || fail "cargo-doc-md did not generate the nixl_sys documentation"

# Fern cannot resolve a local link when a Markdown file and directory share the
# same basename. Preserve the module hierarchy while disambiguating this page.
mv -- "${generated_crate}/descriptors.md" "${generated_crate}/descriptor-types.md"
echo "Making generated Markdown MDX-compatible..."
python3 - "${generated_crate}" <<'PYTHON'
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
index = output_dir / "index.md"
content = index.read_text(encoding="utf-8")
content = content.replace("](descriptors.md)", "](descriptor-types.md)")
index.write_text(content, encoding="utf-8")

for path in sorted(output_dir.rglob("*.md")):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output = []
    in_fence = False

    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            output.append(line)
            continue

        if in_fence:
            output.append(line)
            continue

        # MDX treats raw Rust generic syntax as JSX. Escape angle brackets in
        # prose while preserving fenced and inline code spans.
        segments = line.split("`")
        for segment_index in range(0, len(segments), 2):
            segment = segments[segment_index]
            line_break = "__NIXL_CARGO_DOC_MD_LINE_BREAK__"
            segment = segment.replace("<br>", line_break)
            segment = segment.replace("<br />", line_break)
            segment = segment.replace("<", "&lt;").replace(">", "&gt;")
            segments[segment_index] = segment.replace(line_break, "<br />")

        output.append("`".join(segments))

    path.write_text("".join(output), encoding="utf-8")
PYTHON

expected_pages=(
    index.md
    nixl_sys.md
    agent.md
    descriptor-types.md
    descriptors/query.md
    descriptors/reg.md
    descriptors/sync_manager.md
    descriptors/xfer.md
    descriptors/xfer_dlist_handle.md
    notify.md
    utils/params.md
    utils/string_list.md
    xfer.md
)

for page in "${expected_pages[@]}"; do
    [[ -s "${generated_crate}/${page}" ]] ||
        fail "expected generated page is missing or empty: ${page}"
done

grep -Fq "nixl_sys::agent::Agent" "${generated_crate}/agent.md" ||
    fail "generated Agent API is missing"
grep -Fq "nixl_sys::agent::AgentConfig" "${generated_crate}/agent.md" ||
    fail "generated AgentConfig API is missing"
grep -Fq "nixl_sys::descriptors::MemType" "${generated_crate}/descriptor-types.md" ||
    fail "generated descriptor API is missing"
grep -Fq "nixl_sys::xfer::XferRequest" "${generated_crate}/xfer.md" ||
    fail "generated transfer API is missing"

mkdir -p -- "${output_parent}"
stage_parent="$(mktemp -d "${output_parent}/.rust.generated.XXXXXX")"
stage_dir="${stage_parent}/rust"
mv -- "${generated_crate}" "${stage_dir}"

expected_output_dir="${repo_root}/fern/docs/generated/rust"
[[ "${output_dir}" == "${expected_output_dir}" ]] ||
    fail "refusing to replace unexpected output directory: ${output_dir}"

rm -rf -- "${output_dir}"
mv -- "${stage_dir}" "${output_dir}"
rmdir -- "${stage_parent}"
stage_parent=""
stage_dir=""

echo "Generated Rust API Markdown in ${output_dir}"
