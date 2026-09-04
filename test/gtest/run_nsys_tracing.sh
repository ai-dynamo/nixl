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
for expected in nixl::loadRemoteMD nixl::registerMem nixl::postXferReq.write nixl::postXferReq.read; do
    if ! echo "${NVTX_RANGES}" | grep -Fq -- "${expected}"; then
        echo "tracing_nsys: expected NVTX range '${expected}' missing from capture" >&2
        echo "${NVTX_RANGES}" | sed 's/^/  seen: /' >&2
        exit 1
    fi
done

SQLITE="${OUT}.sqlite"
if ! "${NSYS}" export --type sqlite --force-overwrite true --output "${SQLITE}" \
    "${OUT}.nsys-rep" >/dev/null; then
    echo "tracing_nsys: failed to export ${OUT}.nsys-rep to SQLite" >&2
    exit 1
fi
python3 - "${SQLITE}" <<'PY' || exit 1
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1])
rows = connection.execute(
    """
    SELECT COALESCE(events.text, strings.value), events.uint64Value
    FROM NVTX_EVENTS AS events
    LEFT JOIN StringIds AS strings ON events.textId = strings.id
    WHERE COALESCE(events.text, strings.value) IN
          ('nixl::postXferReq.write', 'nixl::postXferReq.read', 'nixl::xfer.complete')
    """
).fetchall()
post_ids = [payload for name, payload in rows if name.startswith("nixl::postXferReq.")]
complete_ids = [payload for name, payload in rows if name == "nixl::xfer.complete"]
if not post_ids or not complete_ids:
    raise SystemExit("tracing_nsys: missing correlated post or completion events")
if any(payload is None for payload in post_ids + complete_ids):
    raise SystemExit("tracing_nsys: correlation payload is not unsigned 64-bit")
if not set(complete_ids).issubset(set(post_ids)):
    raise SystemExit("tracing_nsys: completion correlation payload was not posted")
if max(post_ids.count(payload) for payload in set(post_ids)) < 2:
    raise SystemExit("tracing_nsys: no request correlation payload was reused")
PY

echo "tracing_nsys: wrote ${OUT}.nsys-rep"
exit 0
