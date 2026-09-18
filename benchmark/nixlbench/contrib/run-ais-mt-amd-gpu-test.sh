#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run nixlbench with the AIS_MT backend on AMD VRAM against files under
# FILEPATH, wrapped in hipFile's ais-stats so HIPFILE_STATS_LEVEL output can
# confirm whether the fastpath was used.
#
# Meant to run inside a ROCm container built from .ci/dockerfiles/Dockerfile.rocm
# with an NVMe-backed directory bind-mounted in. See docs/rocm-ci.md for the
# docker run invocation.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run-ais-mt-amd-gpu-test.sh [--skip-ais-stats] [-h]

Runs nixlbench --backend AIS_MT under ais-stats. See docs/rocm-ci.md for the
docker run invocation.

  --skip-ais-stats  Run nixlbench directly, without the ais-stats wrapper.

Environment:
  FILEPATH             Required. Existing, writable directory; nixlbench creates
                       nixlbench_ais_mt_test_file_* under it (see nixl_worker.cpp).
  NIXL_PREFIX          Default: /opt/nixl
  ROCM_PATH            Default: /opt/rocm
  AIS_STATS_BIN        Default: ${ROCM_PATH}/bin/ais-stats
  HIPFILE_STATS_LEVEL  Default: 1 (0=off, 1=basic, 2=detailed)
  NIXLBENCH_EXTRA      Extra nixlbench args, word-split (e.g. '--num_iter 500')

Example:
  FILEPATH=/data/ais-mt-run ./run-ais-mt-amd-gpu-test.sh
EOF
}

SKIP_AIS_STATS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-ais-stats) SKIP_AIS_STATS=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

NIXL_PREFIX="${NIXL_PREFIX:-/opt/nixl}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
AIS_STATS_BIN="${AIS_STATS_BIN:-${ROCM_PATH}/bin/ais-stats}"
FILEPATH="${FILEPATH:-}"
HIPFILE_STATS_LEVEL="${HIPFILE_STATS_LEVEL:-1}"

if [[ -z "$FILEPATH" ]]; then
    echo "ERROR: set FILEPATH to a writable directory inside the container." >&2
    exit 1
fi
if [[ ! -d "$FILEPATH" ]]; then
    echo "ERROR: FILEPATH=${FILEPATH} is not a directory. AIS_MT creates files under it." >&2
    exit 1
fi

# The plugins live under an arch-specific libdir, so NIXL cannot find them
# without this; ROCM_PATH/lib supplies libhipfile at runtime.
arch="$(uname -m)"
case "$arch" in
    aarch64) libarch="aarch64-linux-gnu" ;;
    x86_64)  libarch="x86_64-linux-gnu" ;;
    *)       libarch="${arch}-linux-gnu" ;;
esac
export ROCM_PATH HIPFILE_STATS_LEVEL
export LD_LIBRARY_PATH="${NIXL_PREFIX}/lib:${NIXL_PREFIX}/lib/${libarch}:${NIXL_PREFIX}/lib/${libarch}/plugins:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export NIXL_PLUGIN_DIR="${NIXL_PREFIX}/lib/${libarch}/plugins"

# docker --user keeps the image's ENV HOME=/home/svc-nixl. NIXL then resolves
# $HOME/.nixl.cfg (see configuration.cpp) and weakly_canonical(2) returns
# EACCES for UIDs that cannot search that home directory.
if [[ -n "${HOME:-}" ]] && [[ ! -x "$HOME" ]]; then
    export HOME=/tmp
fi

command -v "${NIXL_PREFIX}/bin/nixlbench" >/dev/null 2>&1 || {
    echo "ERROR: missing ${NIXL_PREFIX}/bin/nixlbench." >&2
    exit 1
}
if [[ "$SKIP_AIS_STATS" != 1 ]]; then
    command -v "${AIS_STATS_BIN}" >/dev/null 2>&1 || {
        echo "ERROR: ais-stats not found at ${AIS_STATS_BIN} (set AIS_STATS_BIN, or pass --skip-ais-stats)." >&2
        exit 1
    }
fi

# --gds_mt_num_threads is nixlbench's generic thread-count flag; it applies to
# AIS_MT too despite the GDS-specific name.
nb_cmd=(
    "${NIXL_PREFIX}/bin/nixlbench"
    --backend AIS_MT
    --filepath "$FILEPATH"
    --initiator_seg_type VRAM
    --target_seg_type VRAM
    --gds_mt_num_threads 4
    --total_buffer_size $((64 * 1024 * 1024))
    --start_block_size 65536
    --max_block_size 65536
    --start_batch_size 1
    --max_batch_size 1
    --num_iter 200
    --warmup_iter 40
    --op_type WRITE
    --check_consistency
)
if [[ -n "${NIXLBENCH_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206  # deliberate word-split of caller-supplied args
    nb_cmd+=(${NIXLBENCH_EXTRA})
fi

report="$(mktemp -t ais-mt-ais-stats.XXXXXX.log)"
trap 'rm -f "$report"' EXIT

echo "HIPFILE_STATS_LEVEL=${HIPFILE_STATS_LEVEL}"
echo "Running: ${nb_cmd[*]}"
if [[ "$SKIP_AIS_STATS" = 1 ]]; then
    "${nb_cmd[@]}" 2>&1 | tee "$report"
else
    "${AIS_STATS_BIN}" "${nb_cmd[@]}" 2>&1 | tee "$report"
fi

# hipFile reports the fastpath in its stats output; a fallback means the I/O
# did not take the AIS path, which is usually the point of running this.
if grep -Eiq 'fastpath|fast[[:space:]]*path' "$report"; then
    echo "OK: hipFile stats report fastpath I/O."
elif grep -Eiq 'fallback' "$report"; then
    echo "FAIL: hipFile stats report fallback I/O, not fastpath." >&2
    # Keep the log: it is the only record of why hipFile fell back.
    trap - EXIT
    echo "Log retained: $report" >&2
    exit 1
else
    echo "WARN: no fastpath/fallback token in output; try HIPFILE_STATS_LEVEL=2." >&2
fi
