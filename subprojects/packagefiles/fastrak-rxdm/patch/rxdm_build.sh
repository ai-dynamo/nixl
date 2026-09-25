#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

SKIP_CLEAN=0

while getopts "s" opt; do
  case ${opt} in
    s )
      SKIP_CLEAN=1
      ;;
    \? )
      echo "Usage: $0 [-s]"
      exit 1
      ;;
  esac
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "${SCRIPT_DIR}"

if [ "${SKIP_CLEAN}" -eq 0 ]; then
  bazel clean
fi

bazel build --define webrtc=system //buffer_mgmt_daemon:rxdm_dxs
