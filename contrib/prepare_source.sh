#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

RXDM_URI=""
DXS_URI=""
NCCL_TCPXO_URI="https://github.com/google/nccl-plugin-gpudirect-tcpxo.git"
NCCL_TCPXO_DIRNAME="nccl-plugin-gpudirect-tcpxo"
NCCL_TCPXO_REF="v1.0.18"
GETOPTS="hr:d:nb:f:s:o:"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BUILD_WORKDIR=""
USE_NCCL_TCPXO_RXDM_DXS="true"
FASTRAK_RXDM_DIR="${SCRIPT_DIR}/../src/plugins/gpudirect_tcpxo/fastrak-rxdm"
BUILD_SCRIPT_DIR="${SCRIPT_DIR}/../contrib"
OUTPUT_DIR="${BUILD_SCRIPT_DIR}/build_temp"

cleanup() {
  if [[ -n "${BUILD_WORKDIR}" ]]; then
    rm -rf "${BUILD_WORKDIR}"
  fi
}
trap cleanup EXIT

if [[ -f "${SCRIPT_DIR}/../.env" ]]; then
  source "${SCRIPT_DIR}/../.env"
fi

usage() {
  printf 'Usage: %s [-h] [-n | <-r URI> <-d URI>] [-b REF] [-f DIR] [-s DIR] [-o DIR]\n\n' "$0"
  echo "Where:"
  echo "  -h        : Display this help message."
  echo "  -r        : URI to pull RxDM from."
  echo "  -d        : URI to pull DXS from."
  echo "  -n        : Use RxDM and DXS from NCCL TCPXO Repo ${NCCL_TCPXO_URI} (default: enabled)"
  echo "  -b        : Commit ref for NCCL TCPXO repository. (default: ${NCCL_TCPXO_REF})"
  echo "  -f        : Directory containing fastrak-rxdm patch/bazel files. (default: ${FASTRAK_RXDM_DIR})"
  echo "  -s        : Directory containing build scripts (build-rxdm.sh). (default: ${BUILD_SCRIPT_DIR})"
  echo "  -o        : Output directory for DXS/RxDM sources. (default: ${OUTPUT_DIR})"
}

prepare_use_nccl_tcpxo() {
  BUILD_WORKDIR=$(mktemp -d)

  git clone "${NCCL_TCPXO_URI}" "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}"
  git -C "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}" checkout --detach "${NCCL_TCPXO_REF}"

  rm -rf "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"

  # Copy RxDM
  cp -r "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}/buffer_mgmt_daemon" "${OUTPUT_DIR}"
  cp -r "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}/dmabuf_bridge" "${OUTPUT_DIR}"
  cp -r "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}/cuda_helpers" "${OUTPUT_DIR}"
  cp -r "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}/patches" "${OUTPUT_DIR}"
  cp -r "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}/webrtc" "${OUTPUT_DIR}"

  # Bazel Settings
  cp "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}/.bazelrc" "${OUTPUT_DIR}/.bazelrc"
  cp "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}/.bazelversion" "${OUTPUT_DIR}/.bazelversion"
  cp "${FASTRAK_RXDM_DIR}/nccl_tcpxo_rxdm_module.bazel" "${OUTPUT_DIR}/MODULE.bazel"

  # Copy DXS
  cp -r "${BUILD_WORKDIR}/${NCCL_TCPXO_DIRNAME}/dxs" "${OUTPUT_DIR}"
}

prepare_use_rxdm_dxs_uri() {
  rm -rf "${OUTPUT_DIR}"
  mkdir -p "$(dirname "${OUTPUT_DIR}")"

  if [[ "${RXDM_URI}" == file://* ]]; then
    LOCAL_PATH="${RXDM_URI#file://}"
    mkdir -p "${OUTPUT_DIR}"
    rsync -av --exclude='.git' "${LOCAL_PATH}/" "${OUTPUT_DIR}/"
  else
    git clone "${RXDM_URI}" "${OUTPUT_DIR}"
  fi

  pushd "${OUTPUT_DIR}" >/dev/null
  ./scripts/01_pull_dxs.sh -u "$DXS_URI" -c "bc094ee5acf2969e1a15e580eb62511cb734b97c"
  popd >/dev/null
}

while getopts ${GETOPTS} opt; do
  case "${opt}" in
  h)
    usage
    exit 0
    ;;
  r)
    RXDM_URI="${OPTARG}"
    USE_NCCL_TCPXO_RXDM_DXS="false"
    ;;
  d)
    DXS_URI="${OPTARG}"
    USE_NCCL_TCPXO_RXDM_DXS="false"
    ;;
  n)
    USE_NCCL_TCPXO_RXDM_DXS="true"
    ;;
  b)
    NCCL_TCPXO_REF="${OPTARG}"
    ;;
  f)
    FASTRAK_RXDM_DIR="${OPTARG}"
    ;;
  s)
    BUILD_SCRIPT_DIR="${OPTARG}"
    ;;
  o)
    OUTPUT_DIR="${OPTARG}"
    ;;
  *)
    usage
    exit 1
    ;;
  esac
done
shift "$((OPTIND - 1))"

FASTRAK_RXDM_DIR="$(readlink -m "${FASTRAK_RXDM_DIR}")"
BUILD_SCRIPT_DIR="$(readlink -m "${BUILD_SCRIPT_DIR}")"
OUTPUT_DIR="$(readlink -m "${OUTPUT_DIR}")"

if [[ "${USE_NCCL_TCPXO_RXDM_DXS}" == "true" ]]; then
  prepare_use_nccl_tcpxo
else
  if [[ -z "${RXDM_URI}" || -z "${DXS_URI}" ]]; then
    echo "Error: Both -r <RXDM_URI> and -d <DXS_URI> must be provided." >&2
    usage
    exit 1
  fi
  prepare_use_rxdm_dxs_uri
fi

# Copy build script and apply patches
cp "${BUILD_SCRIPT_DIR}/build-rxdm.sh" "${OUTPUT_DIR}/build-rxdm.sh"
chmod +x "${OUTPUT_DIR}/build-rxdm.sh"
patch -d "${OUTPUT_DIR}" -p1 <"${FASTRAK_RXDM_DIR}/rxdm.patch"
