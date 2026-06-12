#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Best-effort NVTX capture for CI artifacts. Profiles the tracing gtest under
# Nsight Systems and writes a .nsys-rep. If profiling is unavailable (e.g. perf
# permissions are denied inside a container), the test is skipped (exit 77)
# rather than failing -- the same tests also run in the normal gtest job, which
# is what gates correctness.
#
# Usage: run_nsys_tracing.sh <nsys> <gtest_exe> <out_dir> [extra gtest args...]
set -u

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <nsys> <gtest_exe> <out_dir> [gtest args...]" >&2
    exit 2
fi

NSYS="$1"
shift
GTEST="$1"
shift
OUT_DIR="$1"
shift

mkdir -p "${OUT_DIR}/artifacts"
OUT="${OUT_DIR}/artifacts/nixl_nvtx"

"${NSYS}" profile --trace=nvtx,osrt --force-overwrite true --output "${OUT}" \
    "${GTEST}" "$@" --gtest_filter='*Tracing*'
rc=$?

if [ "${rc}" -ne 0 ]; then
    echo "tracing_nsys: nsys profiling unavailable or failed (rc=${rc}); skipping"
    exit 77
fi

if [ -f "${OUT}.nsys-rep" ]; then
    echo "tracing_nsys: wrote ${OUT}.nsys-rep"
fi
exit 0
