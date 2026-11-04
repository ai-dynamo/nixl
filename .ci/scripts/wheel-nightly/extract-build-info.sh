#!/bin/bash -eE
# Extract Build Info step of build-wheel-nightly-matrix.yaml: pulls the
# cell's wheels out of the built image into dist/ and writes the wheel's
# build-info sidecar into build-info/ for Publish to upload alongside the
# manifest fragment. Env: as build-wheel.sh.

set -x
rm -rf dist && mkdir -p dist
cid=$(docker create "${WHEEL_IMAGE_TAG}")
docker cp "${cid}:/workspace/nixl/dist/." ./dist/
docker rm -f "${cid}" || true
ls -l dist/
ls dist/nixl-*-py3-none-any.whl >/dev/null 2>&1 \
  || { echo "ERROR: meta wheel missing from dist/ - the source ref's Dockerfile.manylinux must emit it unconditionally"; exit 1; }

. ./cell.env
# The toolkit the image compiled against; Publish and the manifest read it
# from cell.env.
CUDA_VERSION="$(docker run --rm "${WHEEL_IMAGE_TAG}" nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)"
[ -n "${CUDA_VERSION}" ] || { echo "ERROR: cannot read the CUDA version from ${WHEEL_IMAGE_TAG}"; exit 1; }
echo "CUDA_VERSION=${CUDA_VERSION}" >> cell.env

# Measured inside the wheel image (ucx_info, git, the auditwheel env) plus
# the per-venv pip dumps the built ref's build-wheel.sh left in
# build-info/. Never fatal: a missing sidecar is reported by the manifest.
rm -rf build-info && mkdir -p build-info
BINARY_WHL="$(cd dist && ls nixl_cu*.whl 2>/dev/null | head -1)"
CU_TAG="cu${CUDA_VERSION//./}"
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
