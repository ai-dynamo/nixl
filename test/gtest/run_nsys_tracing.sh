#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Best-effort NVTX capture for CI artifacts. Profiles the tracing gtest under
# Nsight Systems and writes a .nsys-rep. Only a genuine "profiling unavailable"
# condition (e.g. perf permissions denied inside a container) is reported as a
# skip (exit 77); any other nsys/gtest failure -- or a successful run that did
# not actually produce the .nsys-rep -- is propagated as a real failure so this
# capture test still catches regressions on the profiled path.
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
LOG="${OUT}.log"

"${NSYS}" profile --trace=nvtx,osrt --force-overwrite true --output "${OUT}" \
    "${GTEST}" "$@" --gtest_filter='*Tracing*' >"${LOG}" 2>&1
rc=$?

if [ "${rc}" -ne 0 ]; then
    # Distinguish "profiling not permitted here" (a legitimate skip) from a real
    # nsys/gtest failure, which must fail the test.
    if grep -qiE 'denied|not permitted|permission|unsupported|insufficient|capabilit' "${LOG}"; then
        echo "tracing_nsys: nsys profiling unavailable (rc=${rc}); skipping"
        sed 's/^/  nsys: /' "${LOG}" >&2
        exit 77
    fi
    echo "tracing_nsys: nsys/gtest run failed (rc=${rc})" >&2
    cat "${LOG}" >&2
    exit "${rc}"
fi

if [ ! -f "${OUT}.nsys-rep" ]; then
    echo "tracing_nsys: profiling reported success but ${OUT}.nsys-rep was not created" >&2
    cat "${LOG}" >&2
    exit 1
fi

# Sanity-check expected NVTX ranges and the new metadata-exchange span.
NVTX_RANGES="$(nsys stats --force-export=true --report nvtx_sum --format csv "${OUT}.nsys-rep" 2>/dev/null \
    | grep 'nixl::' | sed 's/.*,//' | sort -u)"
for expected in nixl::loadRemoteMD nixl::registerMem nixl::postXferReq.write; do
    if ! echo "${NVTX_RANGES}" | grep -q "${expected}"; then
        echo "tracing_nsys: expected NVTX range '${expected}' missing from capture" >&2
        echo "${NVTX_RANGES}" | sed 's/^/  seen: /' >&2
        exit 1
    fi
done

echo "tracing_nsys: wrote ${OUT}.nsys-rep"
exit 0
