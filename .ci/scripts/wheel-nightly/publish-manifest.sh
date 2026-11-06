#!/bin/bash -eE
# pipeline_stop of build-wheel-nightly-matrix.yaml: merges this build's
# manifest fragments from the PVC into build-manifest.json and uploads it
# next to the wheels (the merge itself is .ci/scripts/merge_build_manifest.py, run
# in a plain Python image). Exit 0 means published; otherwise the reason is
# in manifest-status.txt and build-manifest.json, when present, shows what
# was assembled. The build's staging dir is removed on exit either way and
# stale siblings are pruned; --cleanup-only does only that.
# Env: MANIFEST_STAGING_ROOT, BUILD_NUMBER, CI_PYTHON_IMAGE, and for the
# merge EXPECTED_WHEEL_CELLS, WHEEL_REPO_URL/WHEEL_REPO_API/WHEEL_REPO_UI,
# JOB_NAME, BUILD_URL, NIXL_VERSION, ci_refspec, PUBLISH_DIR,
# ARTIFACTORY_USER/ARTIFACTORY_TOKEN.

STAGING_ROOT="${MANIFEST_STAGING_ROOT:?}"
STAGING_DIR="${STAGING_ROOT}/${BUILD_NUMBER:?}"

cleanup() {
  rm -rf "${STAGING_DIR}"
  if [ -d "${STAGING_ROOT}" ]; then
    find "${STAGING_ROOT}" -mindepth 1 -maxdepth 1 -type d -mmin +720 -exec rm -rf {} +
  fi
}
if [ "${1:-}" = "--cleanup-only" ]; then
  cleanup
  exit 0
fi
trap cleanup EXIT

# Copied once into the workspace: the PVC is writable by other jobs, so
# nothing under it is opened in place or put on a command line.
rm -rf manifest-fragments build-manifest.json manifest-status.txt
mkdir -p manifest-fragments
if [ -d "${STAGING_DIR}" ]; then
  cp -a "${STAGING_DIR}/." manifest-fragments/
fi
echo "Fragments: $(ls -1 manifest-fragments | wc -l) entries"

# Fresh pod: the same podman storage reset as the cells' Prepare step.
rm -f /etc/containers/storage.conf /usr/share/containers/storage.conf
podman system reset -f >/dev/null 2>&1 || true

podman run --rm -v "${PWD}:/work" -w /work \
  -e EXPECTED_WHEEL_CELLS -e WHEEL_REPO_URL -e WHEEL_REPO_API -e WHEEL_REPO_UI \
  -e JOB_NAME -e BUILD_NUMBER -e BUILD_URL -e NIXL_VERSION -e ci_refspec -e PUBLISH_DIR \
  -e ARTIFACTORY_USER -e ARTIFACTORY_TOKEN \
  "${CI_PYTHON_IMAGE:?}" \
  python3 .ci/scripts/merge_build_manifest.py \
    --fragments manifest-fragments --out build-manifest.json --status-file manifest-status.txt
