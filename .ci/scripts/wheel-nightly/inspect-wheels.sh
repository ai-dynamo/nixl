#!/bin/bash -eE
# pipeline_stop of build-wheel-publish-matrix.yaml, release runs only:
# compares the wheel set this build published against INSPECT_REFERENCE with
# .ci/scripts/wheel_inspect, run in a plain Python image, which also uploads
# the report next to the wheels. Exit 0 means a report was produced
# (wheel-inspection/wheel-inspection.{json,md,html}); wheel-inspection/status.txt
# holds the result counts, or the reason there is no report; exit 124 is the
# 30-minute cap. The downloaded wheels are removed on exit either way.
# Env: WHEEL_REPO_URL, PUBLISH_DIR, BUILD_NUMBER, INSPECT_REFERENCE,
# CI_PYTHON_IMAGE, ARTIFACTORY_USER/ARTIFACTORY_TOKEN.

[[ "${PUBLISH_DIR:-}" =~ ^release/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || { echo "ERROR: not a release run (PUBLISH_DIR='${PUBLISH_DIR:-}')"; exit 1; }
CANDIDATE_URL="${WHEEL_REPO_URL:?}/${PUBLISH_DIR}/${BUILD_NUMBER:?}"

OUT_DIR=wheel-inspection
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
trap 'rm -rf "${OUT_DIR}/candidate" "${OUT_DIR}/reference"' EXIT

# Same podman storage reset as publish-manifest.sh, which may have stopped before it.
rm -f /etc/containers/storage.conf /usr/share/containers/storage.conf
podman system reset -f >/dev/null 2>&1 || true

timeout 30m podman run --rm -v "${PWD}:/work" -w /work \
  -e ARTIFACTORY_USER -e ARTIFACTORY_TOKEN \
  "${CI_PYTHON_IMAGE:?}" \
  python3 .ci/scripts/wheel_inspect release-run \
    --candidate "artifactory:${CANDIDATE_URL}" --reference "${INSPECT_REFERENCE:?}" \
    --out-dir "${OUT_DIR}" --status-file "${OUT_DIR}/status.txt"
