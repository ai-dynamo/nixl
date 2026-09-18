#!/bin/bash
# Copyright 2026 Google LLC
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
set -ex

PREPARE="false"
CLEAN="false"
RXDM_URI=""
DXS_URI=""
NCCL_TCPXO_URI="https://github.com/google/nccl-plugin-gpudirect-tcpxo.git"
NCCL_TCPXO_DIRNAME="nccl-plugin-gpudirect-tcpxo"
GETOPTS="hpcr:d:n"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BUILD_WORKDIR="${SCRIPT_DIR}/build_temp"
USE_NCCL_TCPXO_RXDM_DXS="true"

if [[ -f "${SCRIPT_DIR}/../.env" ]]; then
  source "${SCRIPT_DIR}/../.env"
fi

usage() {
  echo "Usage: $0 [-h] [-p] [-c] \n"
  echo "Where:"
  echo "  -h        : Display this help message."
  echo "  -p        : Prepare sources for build."
  echo "  -c        : Clean build sources."
  echo "  -r        : URI to pull RxDM from ."
  echo "  -d        : URI to pull DXS from."
  echo "  -n        : Use RxDM and DXS from NCCL TCPXO Repo ${NCCL_TCPXO_URI} (default: enabled)"
}

HASH_FILE="${SCRIPT_DIR}/../subprojects/packagefiles/fastrak-rxdm/.fastrak-rxdm-dxs.hash"
TARBALL_DEST="${SCRIPT_DIR}/../subprojects/packagefiles/fastrak-rxdm/fastrak-rxdm.tar.gz"

prepare_use_nccl_tcpxo() {
   # Check if source already exists. If so, bail.
  if [[ -d "${BUILD_WORKDIR}" ]]; then
    exit 0
  fi
  # Create a temporary directory to clone into.
  mkdir -p "$BUILD_WORKDIR"
  cd "$BUILD_WORKDIR"

  mkdir -p fastrak-rxdm

  git clone ${NCCL_TCPXO_URI} ${NCCL_TCPXO_DIRNAME}

  # Copy RxDM
  cp -r ${NCCL_TCPXO_DIRNAME}/buffer_mgmt_daemon fastrak-rxdm
  cp -r ${NCCL_TCPXO_DIRNAME}/dmabuf_bridge fastrak-rxdm
  cp -r ${NCCL_TCPXO_DIRNAME}/cuda_helpers fastrak-rxdm
  cp -r ${NCCL_TCPXO_DIRNAME}/patches fastrak-rxdm
  cp -r ${NCCL_TCPXO_DIRNAME}/webrtc fastrak-rxdm

  # Bazel Settings
  cp ${NCCL_TCPXO_DIRNAME}/.bazelrc fastrak-rxdm/.bazelrc
  cp ${NCCL_TCPXO_DIRNAME}/.bazelversion fastrak-rxdm/.bazelversion
  cp "${SCRIPT_DIR}/../subprojects/packagefiles/fastrak-rxdm/nccl_tcpxo_rxdm_module.bazel" fastrak-rxdm/MODULE.bazel

  # Copy DXS
  cp -r ${NCCL_TCPXO_DIRNAME}/dxs fastrak-rxdm

  # Genrate tarball and move to destination.
  tar -zcf fastrak-rxdm.tar.gz fastrak-rxdm --warning=no-file-changed
  mv fastrak-rxdm.tar.gz "${TARBALL_DEST}"

  cd "$SCRIPT_DIR"
}

prepare_use_rxdm_dxs_uri() {
  # If URIs are not local files, try to get git hashes for caching
  if [[ "${RXDM_URI}" != file://* ]] && [[ "${DXS_URI}" != file://* ]]; then
    RXDM_HASH=$(git ls-remote "${RXDM_URI}" HEAD 2>/dev/null | awk '{print $1}')
    DXS_HASH=$(git ls-remote "${DXS_URI}" HEAD 2>/dev/null | awk '{print $1}')
    CURRENT_HASHES="${RXDM_HASH} ${DXS_HASH}"

    if [[ -n "${RXDM_HASH}" ]] && [[ -f "${HASH_FILE}" ]] && [[ -f "${TARBALL_DEST}" ]]; then
      SAVED_HASHES=$(cat "${HASH_FILE}")
      if [[ "${CURRENT_HASHES}" == "${SAVED_HASHES}" ]]; then
        echo "Using cached fastrak-rxdm.tar.gz (hashes match: ${CURRENT_HASHES})"
        exit 0
      fi
    fi
  fi

  # Check if source already exists. If so, bail.
  if [[ -d "${BUILD_WORKDIR}" ]]; then
    exit 0
  fi
  # Create a temporary directory to clone into.
  mkdir -p "$BUILD_WORKDIR"
  cd "$BUILD_WORKDIR"

  if [[ "${RXDM_URI}" == file://* ]]; then
    LOCAL_PATH="${RXDM_URI#file://}"
    rsync -av --exclude='.git' "${LOCAL_PATH}/" fastrak-rxdm/
  else
    git clone "${RXDM_URI}" fastrak-rxdm
  fi

  cd fastrak-rxdm
  ./scripts/01_pull_dxs.sh -u "$DXS_URI" -c "bc094ee5acf2969e1a15e580eb62511cb734b97c"
  cd ../
  tar -zcf fastrak-rxdm.tar.gz fastrak-rxdm --warning=no-file-changed
  mv fastrak-rxdm.tar.gz "${TARBALL_DEST}"

  # Save the new hashes if applicable
  if [[ "${RXDM_URI}" != file://* ]] && [[ "${DXS_URI}" != file://* ]] && [[ -n "${CURRENT_HASHES:-}" ]]; then
    echo "${CURRENT_HASHES}" >"${HASH_FILE}"
  fi

  cd "$SCRIPT_DIR"
}

clean() {
  cd "${SCRIPT_DIR}"
  # We keep the tarball and hash file for caching across builds,
  # but we still clean up the temporary workspace directory.
  rm -rf "${BUILD_WORKDIR}"
}

while getopts ${GETOPTS} opt; do
  case "${opt}" in
  h)
    usage
    exit 0
    ;;
  p)
    PREPARE="true"
    ;;
  c)
    CLEAN="true"
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
  *)
    usage
    exit 1
    ;;
  esac
done
shift "$((OPTIND - 1))"

if [[ "${PREPARE}" == "true" && "${CLEAN}" == "true" ]]; then
  echo "Cannot set prepare and clean together."
  exit 1
elif [[ "${PREPARE}" == "false" && "${CLEAN}" == "false" ]]; then
  usage
  exit 0
elif [[ "${PREPARE}" == "true" ]]; then
  if [[ "${USE_NCCL_TCPXO_RXDM_DXS}" == "true" ]]; then
    prepare_use_nccl_tcpxo
  else
    prepare_use_rxdm_dxs_uri
  fi
elif [[ "${CLEAN}" == "true" ]]; then
  clean
fi
