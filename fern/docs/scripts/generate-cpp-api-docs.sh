#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly DOXYBOOK2_VERSION="1.5.0"
readonly DOXYBOOK2_ARCHIVE_SHA256="3fb90354b7ab3e8139a5606221865ff6aa0c53f2805e56088dcbd8185ebb5b41"
readonly DOXYBOOK2_URL="https://github.com/matusnovak/doxybook2/releases/download/v${DOXYBOOK2_VERSION}/doxybook2-linux-amd64-v${DOXYBOOK2_VERSION}.zip"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
output_parent="${repo_root}/fern/docs/generated"
output_dir="${output_parent}/cpp"
tool_cache="${NIXL_DOC_TOOLS_CACHE:-/tmp/nixl-doc-tools}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nixl-cpp-docs.XXXXXX")"
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
require_command doxygen
require_command mktemp
require_command python3

if [[ -n "${DOXYBOOK2_BIN:-}" ]]; then
    doxybook2_bin="${DOXYBOOK2_BIN}"
elif command -v doxybook2 >/dev/null 2>&1; then
    doxybook2_bin="$(command -v doxybook2)"
else
    require_command curl
    require_command sha256sum
    require_command unzip

    doxybook2_dir="${tool_cache}/doxybook2-${DOXYBOOK2_VERSION}"
    doxybook2_bin="${doxybook2_dir}/bin/doxybook2"
    archive="${doxybook2_dir}/doxybook2.zip"

    if [[ ! -x "${doxybook2_bin}" ]]; then
        mkdir -p -- "${doxybook2_dir}"
        echo "Downloading doxybook2 v${DOXYBOOK2_VERSION}..."
        curl --fail --location --silent --show-error "${DOXYBOOK2_URL}" --output "${archive}"
        echo "${DOXYBOOK2_ARCHIVE_SHA256}  ${archive}" | sha256sum --check --status ||
            fail "doxybook2 archive checksum verification failed"
        unzip -q -o "${archive}" -d "${doxybook2_dir}"
        chmod +x "${doxybook2_bin}"
    fi
fi

[[ -x "${doxybook2_bin}" ]] || fail "doxybook2 is not executable: ${doxybook2_bin}"

doxyfile="${work_dir}/Doxyfile"
doxygen_output="${work_dir}/doxygen"

cat >"${doxyfile}" <<EOF
PROJECT_NAME = "NIXL C++ API"
OUTPUT_DIRECTORY = "${doxygen_output}"
STRIP_FROM_PATH = "${repo_root}/"

INPUT = "${repo_root}/src/api/cpp/nixl.h" \
        "${repo_root}/src/api/cpp/nixl_types.h" \
        "${repo_root}/src/api/cpp/nixl_params.h" \
        "${repo_root}/src/api/cpp/nixl_descriptors.h"
FILE_PATTERNS = *.h
RECURSIVE = NO

EXTRACT_ALL = YES
EXTRACT_PRIVATE = NO
EXTRACT_STATIC = NO
HIDE_UNDOC_MEMBERS = NO
HIDE_UNDOC_CLASSES = NO

GENERATE_HTML = NO
GENERATE_LATEX = NO
GENERATE_XML = YES
XML_OUTPUT = xml

ENABLE_PREPROCESSING = YES
MACRO_EXPANSION = YES
QUIET = YES
WARN_IF_UNDOCUMENTED = NO
EOF

echo "Generating Doxygen XML..."
doxygen "${doxyfile}"

mkdir -p -- "${output_parent}"
stage_dir="$(mktemp -d "${output_parent}/.cpp.generated.XXXXXX")"

echo "Converting Doxygen XML to Markdown..."
"${doxybook2_bin}" \
    --quiet \
    --input "${doxygen_output}/xml" \
    --output "${stage_dir}" \
    --config-data '{"useFolders":false,"foldersToGenerate":["classes","files","namespaces"]}'

echo "Making generated Markdown MDX-compatible..."
python3 - "${stage_dir}" <<'PY'
from pathlib import Path
import re
import sys

output_dir = Path(sys.argv[1])

for path in sorted(output_dir.glob("*.md")):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output = []
    in_fence = False
    heading_ids = set()

    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            output.append(line)
            continue

        if in_fence:
            output.append(line)
            continue

        # Give Fern the same explicit heading IDs used by Doxybook2 links.
        # Only the first overload receives the shared function anchor.
        heading = re.match(r"^(#{2,6})\s+(.+?)(\r?\n)?$", line)
        if heading is not None:
            title = heading.group(2)
            heading_id = re.sub(r"\s+", "-", title.strip().lower()).replace("_", "-")
            if re.fullmatch(r"[a-z0-9-]+", heading_id) and heading_id not in heading_ids:
                ending = heading.group(3) or ""
                line = f"{heading.group(1)} {title} [#{heading_id}]{ending}"
                heading_ids.add(heading_id)

        # Doxybook2 uses the operator spelling in link fragments, but Markdown
        # heading IDs omit the angle bracket.
        line = line.replace("#function-operator<", "#function-operator")

        # MDX treats raw C++ template syntax as JSX. Escape angle brackets in
        # prose and tables while preserving fenced and inline code spans.
        segments = line.split("`")
        for index in range(0, len(segments), 2):
            segment = segments[index]
            line_break = "__NIXL_DOXYBOOK_LINE_BREAK__"
            segment = segment.replace("<br>", line_break)
            segment = segment.replace("<", "&lt;").replace(">", "&gt;")
            segment = segment.replace(r"\\&gt;", "&gt;")
            segments[index] = segment.replace(line_break, "<br />")

        output.append("`".join(segments))

    path.write_text("".join(output), encoding="utf-8")
PY

expected_pages=(
    classnixlAgent.md
    structnixlAgentConfig.md
    nixl__types_8h.md
    classnixlBasicDesc.md
    classnixlBlobDesc.md
    structnixlRemoteDesc.md
    classnixlDescList.md
)

for page in "${expected_pages[@]}"; do
    [[ -s "${stage_dir}/${page}" ]] || fail "expected generated page is missing or empty: ${page}"
done

expected_output_dir="${repo_root}/fern/docs/generated/cpp"
[[ "${output_dir}" == "${expected_output_dir}" ]] ||
    fail "refusing to replace unexpected output directory: ${output_dir}"

rm -rf -- "${output_dir}"
mv -- "${stage_dir}" "${output_dir}"
stage_dir=""

echo "Generated C++ API Markdown in ${output_dir}"
