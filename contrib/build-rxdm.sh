#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

SKIP_CLEAN=0
INSTALL_DIR="/usr/local/lib"

while getopts "sd:" opt; do
  case ${opt} in
  s)
    SKIP_CLEAN=1
    ;;
  d)
    INSTALL_DIR="${OPTARG}"
    ;;
  \?)
    echo "Usage: $0 [-s] [-d <install_dir>]"
    exit 1
    ;;
  esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cd "${SCRIPT_DIR}"

if [ "${SKIP_CLEAN}" -eq 0 ]; then
  bazel clean
fi

bazel build --define webrtc=system //buffer_mgmt_daemon:rxdm_dxs
mkdir -p "${INSTALL_DIR}"
install -m 755 bazel-bin/buffer_mgmt_daemon/librxdm_dxs.so "${INSTALL_DIR}/"
echo "${INSTALL_DIR}" >/etc/ld.so.conf.d/rxdm_dxs.conf
ldconfig
