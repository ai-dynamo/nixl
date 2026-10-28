#!/bin/bash -eE
# Publish step of build-wheel-nightly-matrix.yaml: extracts the cell's
# wheels from the built image, uploads them with their properties and
# deposits this cell's build-manifest fragment on the PVC. Env: as
# build-wheel.sh, plus PUBLISH_DIR, WHEEL_REPO_URL/WHEEL_REPO_API,
# MANIFEST_STAGING_ROOT and ARTIFACTORY_USER/ARTIFACTORY_TOKEN.

set -x
# Extract the wheels from the built image, then upload.
rm -rf dist && mkdir -p dist
cid=$(docker create "${WHEEL_IMAGE_TAG}")
docker cp "${cid}:/workspace/nixl/dist/." ./dist/
docker rm -f "${cid}" || true
ls -l dist/

# Nightly runs leave PUBLISH_DIR empty and publish to verification/;
# manual release runs can pass release/<ver>. Allowlist it - the URL is
# PUT with the service token, and a path like ../<other-repo> would
# normalize into a different Artifactory repo.
PUBLISH_DIR="${PUBLISH_DIR:-verification}"
[[ "${PUBLISH_DIR}" =~ ^(verification|release/[A-Za-z0-9][A-Za-z0-9._-]*)$ ]] \
  || { echo "ERROR: invalid PUBLISH_DIR '${PUBLISH_DIR}'"; exit 1; }
. ./wheel_build_ids.env
. ./cell.env
CELL_SPCX_REF="${UCX_SPCX_PLUGIN_REF}"

# Outside the build_options.env guard below: the release folder no longer
# carries the sha, so NIXL_SHA must be set even for a ref that predates
# --build-options-file.
PROPERTIES="NIXL_SHA=${NIXL_SHA}"
PROPERTIES+=";NIXL_VERSION=${NIXL_VERSION}"
PROPERTIES+=";JOB_NAME=${JOB_NAME}"
PROPERTIES+=";BUILD_NUMBER=${BUILD_NUMBER}"

# Provenance recorded by Build Wheel (contrib/build-container.sh
# --build-options-file). Only set properties for values that exist,
# so an older built ref without this flag just skips them.
if [ -f build_options.env ]; then
  . ./build_options.env
  # The plugin job's provenance beats the built ref's own default.
  [ -z "${CELL_SPCX_REF}" ] || UCX_SPCX_PLUGIN_REF="${CELL_SPCX_REF}"
  [ -n "${UCX_REF:-}" ] && PROPERTIES+=";UCX_REF=${UCX_REF}"
  [ -n "${UCX_SHA:-}" ] && PROPERTIES+=";UCX_SHA=${UCX_SHA}"
  [ -n "${CUDA_VERSION:-}" ] && PROPERTIES+=";CUDA_VERSION=${CUDA_VERSION}"
  [ "${BUILD_UCX_SPCX_PLUGIN:-}" = "true" ] && [ -n "${UCX_SPCX_PLUGIN_REF:-}" ] \
    && PROPERTIES+=";UCX_SPCX_PLUGIN_REF=${UCX_SPCX_PLUGIN_REF}"
  [ "${BUILD_INFINIA:-}" = "true" ] && [ -n "${INFINIA_LIBS_IMAGE:-}" ] \
    && PROPERTIES+=";INFINIA_LIBS_IMAGE=${INFINIA_LIBS_IMAGE}"
  # A trailing `[ ... ] && ...` that does not fire would otherwise leave
  # the whole if-block with a non-zero status.
  true
fi
# From the plugin job's provenance file (cell.env), not build_options.env.
if [ -n "${UCX_SPCX_PLUGIN_COMMIT:-}" ]; then
  PROPERTIES+=";UCX_SPCX_PLUGIN_COMMIT=${UCX_SPCX_PLUGIN_COMMIT}"
fi

# Build-info sidecar for the manifest, measured inside the wheel image
# (ucx_info, git, the auditwheel env) plus the per-venv pip dumps the
# built ref's build-wheel.sh left in build-info/. Never fatal.
rm -rf build-info && mkdir -p build-info
BINARY_WHL="$(cd dist && ls nixl_cu*.whl 2>/dev/null | head -1)"
CU_TAG="cu${CUDA_VERSION:-${cuda}}"; CU_TAG="${CU_TAG//./}"
if [ -n "${BINARY_WHL}" ]; then
  docker run --rm -v "${PWD}:/work" "${WHEEL_IMAGE_TAG}" \
    python3 /work/.ci/scripts/extract_build_info.py \
      --wheel "/work/dist/${BINARY_WHL}" \
      --out "/work/build-info/${BINARY_WHL}.build-info.json" \
      --python-version "${python_version}" \
      --platform "manylinux_${manylinux}_${arch}" \
      --cu-tag "${CU_TAG}" \
      --torch-requested "${TORCH_VERSIONS_REQUESTED:-}" \
      --pip-dumps /workspace/nixl/build-info \
      --source-dir /workspace/nixl \
    || echo "WARNING: build-info sidecar not written for ${BINARY_WHL}"
fi

# verification/ keeps its g<nixl-sha>.ucx<ucx-sha> schema (consumed by the
# build-llm-container verification pipeline) and is shared between builds;
# release/<ver> gets one folder per build.
if [ "${PUBLISH_DIR}" = "verification" ]; then
  [ -n "${UCX_SHA:-}" ] || { echo "ERROR: verification publish needs UCX_SHA from build_options.env"; exit 1; }
  FOLDER="g${NIXL_SHA}.ucx${UCX_SHA}"
else
  FOLDER="${BUILD_NUMBER}"
fi
DEST="${WHEEL_REPO_URL}/${PUBLISH_DIR}/${FOLDER}"
PROPS_DEST="${WHEEL_REPO_API}/${PUBLISH_DIR}/${FOLDER}"

# curl-PUT (no python on the runner for twine); Artifactory indexes it as a
# pip repo. rc fails the step on any upload error; set +x hides the token.
set +x
rc=0
for whl in dist/nixl*.whl; do
  curl -fsSL -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" \
    -T "${whl}" "${DEST}/$(basename "${whl}")" || rc=1
  curl -fsSL -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" -X PUT \
    "${PROPS_DEST}/$(basename "${whl}")?properties=${PROPERTIES}" || rc=1
done
# Deposit this cell's fragment for pipeline_stop's manifest. Best effort:
# a missing fragment makes the manifest step mark the build UNSTABLE, it
# must not fail a published wheel.
if [ "$rc" -eq 0 ]; then
  WHEELS="$(cd dist && ls nixl*.whl | tr '\n' ' ')"
  printf 'PUBLISH_FOLDER=%s\nWHEELS="%s"\n' "${PUBLISH_DIR}/${FOLDER}" "${WHEELS% }" >> cell.env
  CELL_DIR="${MANIFEST_STAGING_ROOT}/${BUILD_NUMBER}/${CELL_KEY}"
  { mkdir -p "${CELL_DIR}" && cp cell.env wheel_build_ids.env "${CELL_DIR}/" \
    && { [ ! -f build_options.env ] || cp build_options.env "${CELL_DIR}/"; } \
    && { ! ls build-info/*.build-info.json >/dev/null 2>&1 || cp build-info/*.build-info.json "${CELL_DIR}/"; }; } \
    || echo "WARNING: manifest fragment not deposited under ${CELL_DIR}"
fi
exit $rc
