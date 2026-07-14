#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Manually publish nixl wheels (and optionally crates) to Artifactory from an
# already-completed GitHub Actions build run — mirrors the ci.yml upload-x86-wheels /
# upload-arm-wheels / upload-crates jobs, so you don't have to re-run a 2h build just to
# publish artifacts a PR run already produced.
#
# Requires: gh (authenticated), docker.
# Env:
#   ARTIFACTORY_URL          - JFrog base URL (same value as the CI secret)
#   ARTIFACTORY_PYPI_TOKEN   - access token for the pypi repo (wheels)
#   ARTIFACTORY_CARGO_TOKEN  - access token for the cargo repo (crates; only if --crates)
# Usage:
#   ARTIFACTORY_URL=... ARTIFACTORY_PYPI_TOKEN=... \
#     ./contrib/upload-release-artifacts.sh <RUN_ID> [--crates <ECR_IMAGE>]
set -euo pipefail

RUN_ID="${1:?usage: upload-release-artifacts.sh <RUN_ID> [--crates <ECR_IMAGE>]}"
REPO="${REPO:-ai-dynamo/nixl}"
: "${ARTIFACTORY_URL:?set ARTIFACTORY_URL}"
: "${ARTIFACTORY_PYPI_TOKEN:?set ARTIFACTORY_PYPI_TOKEN}"
JF_IMAGE="releases-docker.jfrog.io/jfrog/jfrog-cli-v2-jf"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

upload_arch () {  # $1 = arch label, $2 = artifact-name glob pattern
  local arch="$1" pattern="$2" dir="$WORK/$1"
  mkdir -p "$dir"
  echo "== [$arch] downloading artifacts matching '$pattern' from run $RUN_ID =="
  gh run download "$RUN_ID" --repo "$REPO" --pattern "$pattern" --dir "$dir"
  # Flatten the per-artifact subdirs into one dir of *.whl (mirrors merge-multiple).
  find "$dir" -name '*.whl' -exec mv -t "$dir" {} + 2>/dev/null || true
  local n; n=$(ls "$dir"/*.whl 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "ERROR: no wheels found for $arch"; return 1; }
  local ver; ver=$(basename "$(ls "$dir"/nixl*.whl | head -n1)" | cut -d'-' -f2)
  echo "== [$arch] uploading $n wheels (version $ver) to Artifactory =="
  local cn="upload_nixl_${arch}_${RUN_ID}"
  docker rm -f "$cn" >/dev/null 2>&1 || true
  # Pass the token via env (not the command line) and don't use `set -x`, so the secret
  # is never echoed into logs.
  docker create --name "$cn" -w /workspace -e CI=true -e JFROG_CLI_LOG_LEVEL=INFO \
    -e ART_TOKEN="$ARTIFACTORY_PYPI_TOKEN" -e ART_URL="$ARTIFACTORY_URL" "$JF_IMAGE" bash -c "
      TARGET_PROPS=\"CI_PIPELINE_ID=${RUN_ID};component_name=nixl;os=linux;arch=${arch};version=${ver}\" &&
      jf rt upload '*.whl' 'sw-dynamo-nixl-pypi-local/release/${ver}/${RUN_ID}/${arch}/' \
        --target-props=\"\$TARGET_PROPS\" \
        --access-token \"\$ART_TOKEN\" --url \"\$ART_URL\" \
        --flat --fail-no-op=true --detailed-summary"
  docker cp "$dir/." "$cn:/workspace/"
  docker start -a "$cn"
  docker rm -f "$cn" >/dev/null 2>&1 || true
}

# Wheels — same artifact patterns the CI upload jobs use.
upload_arch x86_64  'dist-build-nixl-manylinux*'
upload_arch aarch64 'dist-build-nixl-arm-manylinux*'

# Crates (optional) — published from the built runtime image via cargo, exactly like the
# upload-crates job. Pass --crates <ECR_IMAGE> (e.g. <ECR>/nixl-ci:build-nixl-<sha>-<run>).
if [ "${2:-}" = "--crates" ]; then
  : "${ARTIFACTORY_CARGO_TOKEN:?set ARTIFACTORY_CARGO_TOKEN for --crates}"
  IMAGE="${3:?--crates requires the built ECR image name}"
  echo "== publishing crates from $IMAGE =="
  docker run --rm -e ARTIFACTORY_CARGO_TOKEN="$ARTIFACTORY_CARGO_TOKEN" -e CI_PIPELINE_ID="$RUN_ID" \
    "$IMAGE" /bin/bash -c "set -x &&
      sed -i -E 's/^(version = \"([^\"]+)\")/version = \"\2-rc.${RUN_ID}\"/' Cargo.toml &&
      cargo check --manifest-path src/bindings/rust/Cargo.toml &&
      cargo publish --manifest-path src/bindings/rust/Cargo.toml \
        --token 'Bearer $ARTIFACTORY_CARGO_TOKEN' \
        --index 'sparse+${ARTIFACTORY_URL}/api/cargo/sw-dynamo-nixl-cargo-local/index/' \
        --no-verify --allow-dirty"
fi
echo "== done =="
