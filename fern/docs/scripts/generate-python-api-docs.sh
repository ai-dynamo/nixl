#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly HANDSDOWN_VERSION="2.1.0"
readonly SOURCE_REPOSITORY="https://github.com/ai-dynamo/nixl"
readonly SOURCE_BRANCH="main"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
source_dir="${repo_root}/src/api/python"
export_file="${source_dir}/__init__.py"
ep_source_dir="${repo_root}/examples/device/ep/nixl_ep"
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

[[ -d "${source_dir}" ]] || fail "Python API source directory was not found: ${source_dir}"
[[ -s "${export_file}" ]] || fail "Python API export manifest was not found: ${export_file}"
[[ -d "${ep_source_dir}" ]] || fail "Expert Parallel API source directory was not found: ${ep_source_dir}"

# Verify the public export surface before documenting every module in the package.
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
python_output="${handsdown_output}/python"
ep_output="${handsdown_output}/expert-parallel"
mkdir -p -- "${python_output}" "${ep_output}"

generate_package_docs() {
    local package_dir="$1"
    local package_output="$2"
    local package_name="$3"
    local source_code_path="$4"

    "${handsdown_bin}" \
        --input-path "${package_dir}" \
        --output-path "${package_output}" \
        --name "${package_name}" \
        --theme material \
        --external "${SOURCE_REPOSITORY}" \
        --source-code-path "${source_code_path}" \
        --branch "${SOURCE_BRANCH}" \
        --quiet
}

echo "Generating Python API Markdown with handsdown..."
generate_package_docs "${source_dir}" "${python_output}" "NIXL Python API" "src/api/python"
generate_package_docs \
    "${ep_source_dir}" \
    "${ep_output}" \
    "NIXL Expert Parallel API" \
    "examples/device/ep/nixl_ep"

stage_dir="$(mktemp -d "${output_parent}/.python.generated.XXXXXX")"
python_stage_dir="${stage_dir}/Files/src/api/python"
ep_stage_dir="${stage_dir}/Files/examples/device/ep/nixl_ep"
mkdir -p -- "${python_stage_dir}" "${ep_stage_dir}"
cp -a -- "${python_output}/." "${python_stage_dir}/"
cp -a -- "${ep_output}/." "${ep_stage_dir}/"

# Preserve Handsdown's module pages under their source paths, then derive
# Doxybook-style class and module indices without duplicating API content.
python3 - "${stage_dir}" "${repo_root}" <<'PYTHON'
import ast
from pathlib import Path
import re
import shutil
import sys

output_dir = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
files_dir = output_dir / "Files"

for path in sorted(files_dir.rglob("README.md")):
    path.rename(path.with_name("index.md"))
for path in sorted(files_dir.rglob("_api.md")):
    path.rename(path.with_name("api.md"))

titles = {
    "Files/src/api/python/index.md": "Python Bindings",
    "Files/src/api/python/api.md": "Python API",
    "Files/src/api/python/logging.md": "Logging",
    "Files/examples/device/ep/nixl_ep/index.md": "Expert Parallel",
    "Files/examples/device/ep/nixl_ep/buffer.md": "Buffer",
    "Files/examples/device/ep/nixl_ep/utils.md": "Utilities",
}
index_labels = ("NIXL Python API Index", "NIXL Expert Parallel API Index")

for path in sorted(files_dir.rglob("*.md")):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    relative_path = path.relative_to(output_dir).as_posix()
    title = titles.get(relative_path, path.stem.replace("_", " ").title())

    if lines and lines[0].startswith("# "):
        del lines[0]
    lines = [line for line in lines if not any(label in line for label in index_labels)]

    content = "".join(lines)
    content = content.replace("(./_api.md", "(./api.md")
    if path.name == "index.md":
        content = re.sub(r"(\./[^)]+\.md)#[^)]+", r"\1", content)
        content = content.replace("[Api](./api.md)", "[Python API](./api.md)")
        content = content.replace("[Utils](./utils.md)", "[Utilities](./utils.md)")
    path.write_text(f"---\ntitle: {title}\n---\n{content}", encoding="utf-8")


def write_page(path: Path, title: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {title}\n---\n\n{content.rstrip()}\n", encoding="utf-8")


folder_pages = {
    "Files/index.md": ("Files", "Python modules arranged by their source paths."),
    "Files/src/index.md": ("Source", "Generated documentation for Python modules under `src/`."),
    "Files/src/api/index.md": ("API", "Generated documentation for public API modules."),
    "Files/examples/index.md": ("Examples", "Generated documentation for Python example packages."),
    "Files/examples/device/index.md": ("Device", "Generated documentation for device examples."),
    "Files/examples/device/ep/index.md": (
        "EP",
        "Generated documentation for Expert Parallel packages.",
    ),
}
for relative_path, (title, content) in folder_pages.items():
    write_page(output_dir / relative_path, title, content)

packages = (
    {
        "source": repo_root / "src/api/python",
        "output": output_dir / "Files/src/api/python",
        "module": "nixl",
    },
    {
        "source": repo_root / "examples/device/ep/nixl_ep",
        "output": output_dir / "Files/examples/device/ep/nixl_ep",
        "module": "nixl_ep",
    },
)

modules = []
classes = []
for package in packages:
    source_root = package["source"]
    output_root = package["output"]
    module_root = package["module"]

    for source_path in sorted(source_root.rglob("*.py")):
        relative_source = source_path.relative_to(source_root)
        if relative_source.name == "__init__.py":
            relative_doc = relative_source.parent / "index.md"
            module_parts = relative_source.parent.parts
        else:
            relative_doc = relative_source.with_suffix(".md")
            if relative_doc.name == "_api.md":
                relative_doc = relative_doc.with_name("api.md")
            module_parts = relative_source.with_suffix("").parts

        doc_path = output_root / relative_doc
        if not doc_path.is_file():
            continue

        module_name = ".".join((module_root, *module_parts))
        doc_link = doc_path.relative_to(output_dir).as_posix()
        modules.append((module_name, doc_link))

        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            classes.append((node.name, module_name, f"{doc_link}#{node.name.lower()}"))

module_lines = [f"- [`{name}`](./{link})" for name, link in sorted(set(modules))]
class_lines = [
    f"- [`{name}`](./{link}) — `{module_name}`"
    for name, module_name, link in sorted(classes, key=lambda item: (item[0].lower(), item[1]))
]

write_page(
    output_dir / "index.md",
    "Python API Index",
    "Browse the generated Python documentation by class, module, or source file.\n\n"
    "- [Classes Index](./index_classes.md)\n"
    "- [Modules Index](./index_modules.md)\n"
    "- [Files](./Files/index.md)",
)
write_page(
    output_dir / "index_classes.md",
    "Classes Index",
    "\n".join(class_lines) or "No public classes were found.",
)
write_page(
    output_dir / "index_modules.md",
    "Modules Index",
    "\n".join(module_lines) or "No documented modules were found.",
)

# Fern does not resolve Handsdown's relative .md links within an auto-generated
# folder. Normalize page paths and replace those links with absolute Fern routes.
route_root = "/nixl/api-reference-generated/python"
source_pages = sorted(output_dir.rglob("*.md"))
destinations = {}
for source_path in source_pages:
    relative_path = source_path.relative_to(output_dir)
    destinations[relative_path.as_posix()] = Path(
        *(part.lower().replace("_", "-") for part in relative_path.parts)
    )

normalized_dir = output_dir.with_name(f"{output_dir.name}-fern")
link_pattern = re.compile(r"\]\(([^)#]+\.md)(#[^)]*)?\)")
for source_path in source_pages:
    relative_path = source_path.relative_to(output_dir)
    content = source_path.read_text(encoding="utf-8")

    def replace_link(match: re.Match[str]) -> str:
        target_path = (source_path.parent / match.group(1)).resolve()
        try:
            target_relative = target_path.relative_to(output_dir.resolve()).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"generated link escapes the Python documentation: {source_path}: {match.group(1)}"
            ) from error
        target_destination = destinations.get(target_relative)
        if target_destination is None:
            raise RuntimeError(f"unresolved generated link in {source_path}: {match.group(1)}")

        route_parts = target_destination.with_suffix("").parts
        if route_parts[-1] == "index":
            route_parts = route_parts[:-1]
        route = "/".join((route_root.rstrip("/"), *route_parts))
        return f"]({route}{match.group(2) or ''})"

    destination_path = normalized_dir / destinations[relative_path.as_posix()]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(link_pattern.sub(replace_link, content), encoding="utf-8")

shutil.rmtree(output_dir)
normalized_dir.rename(output_dir)
PYTHON

# Guard against malformed annotations and ensure every generated API group is indexed.
python_api_page="${stage_dir}/files/src/api/python/api.md"
python_logging_page="${stage_dir}/files/src/api/python/logging.md"
ep_buffer_page="${stage_dir}/files/examples/device/ep/nixl-ep/buffer.md"
ep_utils_page="${stage_dir}/files/examples/device/ep/nixl-ep/utils.md"

grep -Fq "agent_name: str" "${python_api_page}" ||
    fail "generated page is missing the nixl_agent constructor"
grep -Fq "backends: list[str]" "${python_api_page}" ||
    fail "generated page is missing Python generic type annotations"
grep -Fq "get_logger" "${python_logging_page}" ||
    fail "generated page is missing the logging API"
grep -Fq "### Buffer().dispatch" "${ep_buffer_page}" ||
    fail "generated page is missing the Expert Parallel Buffer API"
grep -Fq "## EventOverlap" "${ep_utils_page}" ||
    fail "generated page is missing the Expert Parallel utilities API"
grep -Fq "nixl_thread_sync_t" "${stage_dir}/index-classes.md" ||
    fail "classes index is missing the Python bindings API"
grep -Fq "Buffer" "${stage_dir}/index-classes.md" ||
    fail "classes index is missing the Expert Parallel API"
grep -Fq "nixl_ep.buffer" "${stage_dir}/index-modules.md" ||
    fail "modules index is missing the Expert Parallel package"
if grep -Fq "self self" "${python_api_page}"; then
    fail "generated page contains a malformed self parameter"
fi

expected_output_dir="${repo_root}/fern/docs/generated/python"
[[ "${output_dir}" == "${expected_output_dir}" ]] ||
    fail "refusing to replace unexpected output directory: ${output_dir}"

rm -rf -- "${output_dir}"
mv -- "${stage_dir}" "${output_dir}"
stage_dir=""

echo "Generated Python API Markdown in ${output_dir}"
