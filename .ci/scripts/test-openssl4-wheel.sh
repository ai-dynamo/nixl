#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run inside the OpenSSL 4 wheel image, which supplies the external TLS libs.
set -euo pipefail

python_version=${1:?Usage: test-openssl4-wheel.sh PYTHON_VERSION OPENSSL_VERSION}
openssl_version=${2:?Usage: test-openssl4-wheel.sh PYTHON_VERSION OPENSSL_VERSION}
python_tag="cp${python_version//./}"
python="/opt/python/${python_tag}-${python_tag}/bin/python"

actual_openssl=$(/usr/local/openssl/bin/openssl version)
echo "$actual_openssl"
[[ "$actual_openssl" == "OpenSSL ${openssl_version} "* ]]

shopt -s nullglob
wheels=(/workspace/nixl/dist/nixl_cu13-*-${python_tag}-${python_tag}-*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
    echo "ERROR: expected one CUDA 13 wheel for ${python_version}, found ${#wheels[@]}" >&2
    exit 1
fi

smoke_dir=$(mktemp -d /tmp/nixl-openssl4-smoke.XXXXXX)
trap 'rm -rf "$smoke_dir"' EXIT
uv venv --python "$python" "$smoke_dir/venv"
# The core API needs numpy; torch is only used by EP/tensor operations.
uv pip install --python "$smoke_dir/venv/bin/python" --no-deps numpy "${wheels[0]}"
cd "$smoke_dir"
unset PYTHONPATH
"$smoke_dir/venv/bin/python" - <<'PY'
import importlib.metadata
import re
import subprocess

dist = importlib.metadata.distribution("nixl-cu13")
expected = {"libssl.so.4", "libcrypto.so.4"}
found = set()
for entry in dist.files or []:
    path = dist.locate_file(entry)
    if ".so" not in path.name:
        continue
    if path.name.startswith(("libssl", "libcrypto")):
        raise RuntimeError(f"OpenSSL must remain external to the wheel: {entry}")
    dynamic = subprocess.check_output(["readelf", "-d", str(path)], text=True)
    for line in dynamic.splitlines():
        if "(NEEDED)" not in line:
            continue
        for library in re.findall(r"\[(lib(?:ssl|crypto)[^\]]*)\]", line):
            if library not in expected:
                raise RuntimeError(f"Unexpected OpenSSL dependency in {entry}: {library}")
            found.add(library)
if found != expected:
    raise RuntimeError(f"Expected dependencies on {expected}, found {found}")

# Import the CUDA backend directly: CUDA 13 builds do not emit the meta wheel.
from nixl_cu13._api import nixl_agent  # noqa: E402

print(f"OpenSSL 4 wheel smoke test passed: {nixl_agent.__name__}; {sorted(found)}")
PY
