#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly HANDSDOWN_VERSION="2.1.0"
readonly SOURCE_REPOSITORY="https://github.com/ai-dynamo/nixl"
readonly SOURCE_BRANCH="main"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
source_file="${repo_root}/src/api/python/_api.py"
export_file="${repo_root}/src/api/python/__init__.py"
output_parent="${repo_root}/fern/docs/generated"
output_dir="${output_parent}/python"
tool_cache="${NIXL_DOC_TOOLS_CACHE:-/tmp/nixl-doc-tools}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nixl-python-docs.XXXXXX")"
stage_dir=""

cleanup() {
    rm -rf -- "${work_dir}"
    if [[ -n "${stage_dir}" && -d "${stage_dir}" ]]; then
        rm -rf -- "${stage_dir}"
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
require_command mktemp
require_command python3

[[ -s "${source_file}" ]] || fail "Python API implementation was not found: ${source_file}"
[[ -s "${export_file}" ]] || fail "Python API export manifest was not found: ${export_file}"

# _api.py implements the documented public API. __init__.py is the export
# manifest; scanning it too would duplicate aliases of the same definitions.
for public_symbol in nixl_agent nixl_agent_config nixl_prepped_dlist_handle nixl_thread_sync_t nixl_xfer_handle; do
    grep -q "${public_symbol}" "${export_file}" ||
        fail "public symbol is missing from src/api/python/__init__.py: ${public_symbol}"
done

if [[ -n "${HANDSDOWN_BIN:-}" ]]; then
    handsdown_bin="${HANDSDOWN_BIN}"
elif command -v handsdown >/dev/null 2>&1; then
    handsdown_bin="$(command -v handsdown)"
else
    handsdown_venv="${tool_cache}/handsdown-${HANDSDOWN_VERSION}"
    handsdown_bin="${handsdown_venv}/bin/handsdown"

    if [[ ! -x "${handsdown_bin}" ]]; then
        echo "Installing handsdown v${HANDSDOWN_VERSION}..."
        python3 -m venv "${handsdown_venv}"
        "${handsdown_venv}/bin/python" -m pip install --quiet --disable-pip-version-check \
            "handsdown==${HANDSDOWN_VERSION}"
    fi
fi

[[ -x "${handsdown_bin}" ]] || fail "handsdown is not executable: ${handsdown_bin}"

mkdir -p -- "${output_parent}"
handsdown_output="${work_dir}/handsdown"
mkdir -p -- "${handsdown_output}"

echo "Generating Python API Markdown with handsdown..."
(
    cd "${repo_root}"
    "${handsdown_bin}" \
        --input-path "${repo_root}" \
        --files src/api/python/_api.py \
        --output-path "${handsdown_output}" \
        --name "NIXL Python API" \
        --theme material \
        --external "${SOURCE_REPOSITORY}" \
        --branch "${SOURCE_BRANCH}" \
        --quiet
)

generated_page="${handsdown_output}/src/api/python/_api.md"
[[ -s "${generated_page}" ]] || fail "handsdown did not generate the expected Python API page"

stage_dir="$(mktemp -d "${output_parent}/.python.generated.XXXXXX")"
mv -- "${generated_page}" "${stage_dir}/python-api.md"

# Handsdown emits a MkDocs breadcrumb whose discarded index pages are not part
# of the Fern output. Remove that line and give the module a user-facing title.
python3 - "${stage_dir}/python-api.md" <<'PYTHON'
from pathlib import Path
import sys

page = Path(sys.argv[1])
lines = page.read_text(encoding="utf-8").splitlines(keepends=True)
if lines and lines[0].strip() == "# Api":
    lines[0] = "# Python API Reference\n"
lines = [line for line in lines if "NIXL Python API Index" not in line]
page.write_text("".join(lines), encoding="utf-8")
PYTHON

# Guard against the malformed Doxygen-style annotations this generator replaces.
grep -Fq "agent_name: str" "${stage_dir}/python-api.md" ||
    fail "generated page is missing the nixl_agent constructor"
grep -Fq "backends: list[str]" "${stage_dir}/python-api.md" ||
    fail "generated page is missing Python generic type annotations"
if grep -Fq "self self" "${stage_dir}/python-api.md"; then
    fail "generated page contains a malformed self parameter"
fi

expected_output_dir="${repo_root}/fern/docs/generated/python"
[[ "${output_dir}" == "${expected_output_dir}" ]] ||
    fail "refusing to replace unexpected output directory: ${output_dir}"

rm -rf -- "${output_dir}"
mv -- "${stage_dir}" "${output_dir}"
stage_dir=""

echo "Generated Python API Markdown in ${output_dir}/python-api.md"
