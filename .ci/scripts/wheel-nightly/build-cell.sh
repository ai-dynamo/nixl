#!/bin/bash -eE
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Build Wheel step of build-wheel-nightly-matrix.yaml: builds one matrix
# cell's wheel image from nixl-src. Env: the matrix axes (manylinux, arch,
# python_version, cuda) and env block, registry_host/registry_path,
# BUILD_NUMBER, NIXL_VERSION, plus /tmp/ghconfig and wheel_build_ids.env
# from the Prepare and Checkout Source steps.

# Build the wheel image from the checked-out nixl-src. --tag pins the
# image so Publish extracts dist/. No gitlab credentials: the plugin is
# not cloned here, nixl-ci-compile-ucx-plugin compiled it.
set -x
# Infinia is always bundled in the nightly; the built source ref's
# build-container.sh pins the image. The spcx plugin is not built here -
# it was compiled against the same wheel_base and comes off the PVC.
# The two CUDA majors have different torch coverage: 2.11 is not available
# on the cu13 index, and 2.14 is not available on the cu12 one.
# Never default the CUDA line: an unset axis falling through to cu13 made
# both cells build cu13 under identical filenames, so each overwrote the
# other and the folder silently held half a set.
case "${cuda}" in
  12)
    BASE_IMAGE_FLAGS="--base-image ${CU12_BASE_IMAGE} --base-image-tag ${CU12_BASE_TAG} --cuda-version ${CU12_CUDA_VERSION}"
    TORCH_VERSIONS="2.11,2.12,2.13"
    ;;
  13)
    BASE_IMAGE_FLAGS=""
    TORCH_VERSIONS="2.12,2.13,2.14,2.15,2.16"
    ;;
  *)
    echo "ERROR: cuda is '${cuda:-<unset>}', expected 12 or 13"
    exit 1
    ;;
esac
export DOCKER_BUILD_EXTRA_ARGS="--secret id=ghconfig,src=/tmp/ghconfig"
WHEEL_BASE_IMAGE="${registry_host}${registry_path}/${arch}/${WHEEL_BASE_IMAGE_NAME}-cu${cuda}:${WHEEL_BASE_TAG}"
CELL_KEY="${manylinux}_${arch}_py${python_version}_cu${cuda}"

# Ref + commit of the staged plugin module, written by
# nixl-ci-compile-ucx-plugin next to the per-arch dirs. Another job
# wrote it, so it is parsed key by key and validated, never sourced. A
# malformed file is a plugin-job bug; an absent one is an older plugin
# job, and the manifest records the plugin as unknown.
SPCX_PROV="${SPCX_STAGING_ROOT}/${BUILD_NUMBER}/cu${cuda}/spcx-provenance.env"
UCX_SPCX_PLUGIN_REF=""; UCX_SPCX_PLUGIN_COMMIT=""; SPCX_FLAGS=""
if [ -f "${SPCX_PROV}" ]; then
  UCX_SPCX_PLUGIN_REF="$(grep -m1 '^UCX_SPCX_PLUGIN_REF=' "${SPCX_PROV}" | cut -d= -f2-)"
  UCX_SPCX_PLUGIN_COMMIT="$(grep -m1 '^UCX_SPCX_PLUGIN_COMMIT=' "${SPCX_PROV}" | cut -d= -f2-)"
  [[ "${UCX_SPCX_PLUGIN_REF}" =~ ^[A-Za-z0-9._/-]+$ ]] \
    || { echo "ERROR: malformed UCX_SPCX_PLUGIN_REF in ${SPCX_PROV}"; exit 1; }
  [[ "${UCX_SPCX_PLUGIN_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "ERROR: malformed UCX_SPCX_PLUGIN_COMMIT in ${SPCX_PROV}"; exit 1; }
  SPCX_FLAGS="--ucx-spcx-plugin-ref ${UCX_SPCX_PLUGIN_REF}"
  # Puts the "SPCX Plugin:" line back into the wheel README, which
  # build-container.sh only sets on its clone path. Refs that predate
  # the README block merely warn about an unconsumed build-arg.
  DOCKER_BUILD_EXTRA_ARGS+=" --build-arg UCX_SPCX_PLUGIN_VERSION=${UCX_SPCX_PLUGIN_COMMIT}"
fi

# Cell facts for the fragment Publish deposits, written before the
# build so a failed cell leaves nothing behind.
if [ -x nixl-src/.ci/scripts/wheel_build_info.py ]; then
  BUILD_INFO_SUPPORTED=true
else
  BUILD_INFO_SUPPORTED=false
fi
{
  echo "CELL_KEY=${CELL_KEY}"
  echo "CUDA=${cuda}"
  echo "ARCH=${arch}"
  echo "PYTHON_VERSION=${python_version}"
  echo "MANYLINUX=${manylinux}"
  echo "TORCH_VERSIONS_REQUESTED=${TORCH_VERSIONS}"
  echo "WHEEL_BASE_IMAGE=${WHEEL_BASE_IMAGE}"
  echo "UCX_SPCX_PLUGIN_REF=${UCX_SPCX_PLUGIN_REF}"
  echo "UCX_SPCX_PLUGIN_COMMIT=${UCX_SPCX_PLUGIN_COMMIT}"
  echo "BUILD_INFO_SUPPORTED=${BUILD_INFO_SUPPORTED}"
} > cell.env

cd nixl-src
./contrib/build-container.sh \
  ${BASE_IMAGE_FLAGS} \
  --wheel-base "manylinux_${manylinux}" \
  --python-versions "${python_version}" \
  --arch ${arch} \
  --torch-versions "${TORCH_VERSIONS}" \
  --tag "${WHEEL_IMAGE_TAG}" \
  --dockerfile contrib/Dockerfile.manylinux \
  --wheel-base-image "${WHEEL_BASE_IMAGE}" \
  --ucx-spcx-plugin-install "${SPCX_STAGING_ROOT}/${BUILD_NUMBER}/cu${cuda}/${arch}" \
  ${SPCX_FLAGS} \
  --build-infinia \
  --build-options-file ../build_options.env
