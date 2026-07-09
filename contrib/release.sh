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

# NIXL release pipeline (separate from the Jenkins build+test CI).
#
# Jenkins (.ci/jenkins/lib/build-wheel-matrix.yaml) builds + tests the release wheels and
# uploads them to Artifactory staging (sw-dynamo-nixl-pypi-local/release/<version>/<build>/<arch>/).
# This script GRABS that already-published set and drives the release steps:
#   1. verify the wheels are present in Artifactory,
#   2. trigger the nixl-ci GitLab pipeline (nSpect register + wheel scan), and
#   3. ship to prod (TODO — target not defined yet; gated behind --ship).
#
# Nothing is rebuilt here. See also the skills: artifactory-publish, gitlab-nspect-trigger,
# nspect-wheel-upload. Source of truth for the GitLab side: nixl-ci .gitlab-ci.yml.
set -euo pipefail

usage() {
  cat >&2 <<EOF
usage: release.sh --version <X.Y.Z[-rcN]> --build-id <jenkins_build> [--ship] [--dry-run]

  --version    wheel/program version, e.g. 1.4.0 or 1.4.0-rc0 (the <version> path segment)
  --build-id   the id used as the Artifactory path segment when Jenkins uploaded the wheels
               (the build-wheel job uses \$BUILD_NUMBER); also passed to nixl-ci as GITHUB_RUN_ID
  --ship       run the ship-to-prod step (currently a stub; target TBD)
  --dry-run    pass DRY_RUN=true to the nixl-ci trigger

Env / creds:
  ARTIFACTORY_URL           JFrog base (default https://artifactory.nvidia.com/artifactory)
  ARTIFACTORY_PYPI_TOKEN    token to list the wheels (default: ~/.art-token)
  GITLAB_TRIGGER_URL        nixl-ci project trigger API (.../api/v4/projects/<id>/trigger/pipeline)
  GITLAB_TRIGGER_TOKEN      the pipeline trigger token
  NSPECT_ID                 program id (default NSPECT-WO64-8O3P)
  NIXL_CI_REF               branch of nixl-ci to run (default main)
EOF
  exit 1
}

VERSION="" BUILD_ID="" SHIP=0 DRY_RUN=false
while [ $# -gt 0 ]; do
  case "$1" in
    --version)  VERSION="${2:?}"; shift 2;;
    --build-id) BUILD_ID="${2:?}"; shift 2;;
    --ship)     SHIP=1; shift;;
    --dry-run)  DRY_RUN=true; shift;;
    -h|--help)  usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done
[ -n "$VERSION" ] && [ -n "$BUILD_ID" ] || usage

: "${ARTIFACTORY_URL:=https://artifactory.nvidia.com/artifactory}"
: "${ARTIFACTORY_PYPI_TOKEN:=$( [ -f "$HOME/.art-token" ] && tr -d '\n' < "$HOME/.art-token" || true )}"
: "${NSPECT_ID:=NSPECT-WO64-8O3P}"
: "${NIXL_CI_REF:=main}"
PYPI_REPO="sw-dynamo-nixl-pypi-local"
# base version + rc number (WHEEL_VERSION for nixl-ci is the base, e.g. 1.4.0)
BASE_VERSION="${VERSION%%-rc*}"
RC_NUMBER="${VERSION##*-rc}"; [ "$RC_NUMBER" = "$VERSION" ] && RC_NUMBER=""

# --- 1) verify the wheels are in Artifactory ---------------------------------
echo "== verifying wheels under release/${BASE_VERSION}/${BUILD_ID}/ =="
[ -n "$ARTIFACTORY_PYPI_TOKEN" ] || { echo "ERROR: set ARTIFACTORY_PYPI_TOKEN (or ~/.art-token)"; exit 1; }
total=0
for arch in x86_64 aarch64; do
  n=$(curl -fsS -H "Authorization: Bearer $ARTIFACTORY_PYPI_TOKEN" \
        "${ARTIFACTORY_URL}/api/storage/${PYPI_REPO}/release/${BASE_VERSION}/${BUILD_ID}/${arch}/" \
        | grep -oE '"uri"[^,]+\.whl' | wc -l || echo 0)
  echo "  ${arch}: ${n} wheels"
  total=$((total + n))
done
[ "$total" -gt 0 ] || { echo "ERROR: no wheels found — did the Jenkins build-wheel job publish them?"; exit 1; }
echo "  ${total} wheels present."

# --- 2) trigger the nixl-ci GitLab pipeline (nSpect register + scan) ----------
if [ -n "${GITLAB_TRIGGER_URL:-}" ] && [ -n "${GITLAB_TRIGGER_TOKEN:-}" ]; then
  echo "== triggering nixl-ci nSpect + scan pipeline (ref=${NIXL_CI_REF}) =="
  TOKEN_FILE=$(mktemp); trap 'rm -f "$TOKEN_FILE"' EXIT; chmod 600 "$TOKEN_FILE"
  printf '%s' "$GITLAB_TRIGGER_TOKEN" > "$TOKEN_FILE"
  RESP=$(curl -fsSL --request POST \
    --form "token=<${TOKEN_FILE}" \
    --form "ref=${NIXL_CI_REF}" \
    --form "variables[PIPELINE_TYPE]=rc" \
    --form "variables[DRY_RUN]=${DRY_RUN}" \
    --form "variables[NSPECT_ID]=${NSPECT_ID}" \
    --form "variables[NSPECT_RELEASE_VERSION]=${BASE_VERSION}" \
    --form "variables[NSPECT_REGISTERED]=false" \
    --form "variables[WHEEL_VERSION]=${BASE_VERSION}" \
    --form "variables[RC_TAG]=v${VERSION}" \
    --form "variables[RC_NUMBER]=${RC_NUMBER}" \
    --form "variables[GITHUB_RUN_ID]=${BUILD_ID}" \
    --form "variables[ENABLE_WHEEL_SCAN]=true" \
    "$GITLAB_TRIGGER_URL")
  echo "  pipeline: $(printf '%s' "$RESP" | (command -v jq >/dev/null && jq -r '.web_url // .id' || cat))"
else
  echo "== nixl-ci trigger SKIPPED (set GITLAB_TRIGGER_URL + GITLAB_TRIGGER_TOKEN) =="
  echo "   would pass: PIPELINE_TYPE=rc WHEEL_VERSION=${BASE_VERSION} GITHUB_RUN_ID=${BUILD_ID} NSPECT_ID=${NSPECT_ID} ENABLE_WHEEL_SCAN=true"
fi

# --- 3) ship to prod (TODO) --------------------------------------------------
ship_to_prod() {
  echo "== ship-to-prod: NOT IMPLEMENTED =="
  # TODO: promote release/${BASE_VERSION}/${BUILD_ID}/ from the sw-dynamo-nixl-pypi-local
  # staging repo to the production target once decided (Artifactory prod repo / NGC / PyPI).
  echo "   target undecided — see the release plan; nothing shipped."
  return 0
}
[ "$SHIP" -eq 1 ] && ship_to_prod || echo "== ship-to-prod skipped (pass --ship) =="

echo "== done: ${BASE_VERSION} (build ${BUILD_ID}) =="
