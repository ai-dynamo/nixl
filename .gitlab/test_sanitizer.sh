#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Runs the test suites against a sanitizer-instrumented build (built with
# -Dsanitizer=address,undefined or -Dsanitizer=thread). Two stages:
#   1. the mock-based gtest suite via `meson test --suite sanitizer`
#      (options + suppression lists baked into the meson test environment, see
#      test/gtest/meson.build and test/gtest/sanitizer/);
#   2. the single-threaded real-backend binary serdes_test from the install
#      dir, with the *SAN_OPTIONS set here.
# Multi-process tests (e.g. nixl_test) are intentionally skipped: AddressSanitizer
# cannot track allocations across process boundaries. Any sanitizer finding in
# NIXL code aborts the run (halt_on_error=1) and fails the job.

# shellcheck disable=SC1091
. "$(dirname "$0")/../.ci/scripts/common.sh"

set -e
set -x
set -o pipefail

# Parse commandline arguments: first is the install dir, optional second is a
# label (the sanitizer variant, e.g. asan_ubsan/tsan) used only to name the
# archived log artifact.
INSTALL_DIR=$1
SAN_LABEL="${2:-${sanitizer:-sanitizer}}"
NIXL_BUILD_DIR=${NIXL_BUILD_DIR:-nixl_build}
WORKSPACE_DIR="$(pwd)"

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir> [sanitizer_label]"
    exit 1
fi

if [ ! -d "$NIXL_BUILD_DIR" ]; then
    echo "Build directory '$NIXL_BUILD_DIR' not found; run .gitlab/build.sh with a -Dsanitizer=... build first"
    exit 1
fi

ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"

# Archive the meson logs (full per-test output, incl. crash logs that the console
# summary truncates) as a per-task tarball. Named with arch + sanitizer so the
# artifacts from the parallel matrix tasks do not collide. Done in an EXIT trap,
# with absolute paths, so it runs even when a stage crashes and after later
# stages cd into the install dir. See archiveArtifacts in test-sanitizer-matrix.yaml.
case "${NIXL_BUILD_DIR}" in
    /*) NIXL_BUILD_ABS="${NIXL_BUILD_DIR}" ;;
    *)  NIXL_BUILD_ABS="${WORKSPACE_DIR}/${NIXL_BUILD_DIR}" ;;
esac
collect_sanitizer_logs() {
    tar -czf "${WORKSPACE_DIR}/sanitizer-logs-${ARCH}-${SAN_LABEL}.tar.gz" \
        -C "${NIXL_BUILD_ABS}" meson-logs 2>/dev/null || true
}
trap collect_sanitizer_logs EXIT

# Run every stage to completion even if earlier ones fail, collect the failures,
# and report them at the end (the script still exits non-zero if any failed).
SANITIZER_FAILURES=""
run_stage() {
    local name="$1"
    shift
    # Tee each stage to a per-stage log under meson-logs so the archived tarball
    # captures sanitizer output from the install-dir binaries (serdes_test,
    # ucx_backend_test) too -- otherwise that output only reaches the Jenkins
    # console and is lost from the artifact. Relies on `set -o pipefail` so the
    # `if` still sees the stage's own exit status, not tee's.
    local logf
    logf="${NIXL_BUILD_ABS}/meson-logs/sanitizer-stage-$(printf '%s' "$name" | tr ' /' '__').log"
    mkdir -p "${NIXL_BUILD_ABS}/meson-logs"
    echo "==== Running: ${name} ===="
    if "$@" 2>&1 | tee "${logf}"; then
        echo "==== PASSED: ${name} ===="
    else
        local rc=${PIPESTATUS[0]}
        echo "==== FAILED: ${name} (rc=${rc}) ===="
        SANITIZER_FAILURES="${SANITIZER_FAILURES} ${name}"
    fi
}

# The instrumented test binaries link transitive deps (gRPC/upb, etc.) that live
# in the install dir, not the build dir. meson test prepends the build-dir lib
# paths and appends this, so the instrumented build-dir libraries still win while
# third-party deps resolve. Mirrors test_cpp.sh.
export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH

# etcd backs metadata-exchange tests in the gtest suite.
start_etcd_server "/nixl/sanitizer_ci"

# Suppression dir resolved to an absolute path before any cd.
SAN_SUPP_DIR=$(cd "$(dirname "$0")/.." && pwd)/test/gtest/sanitizer

# Instrumented runs are much slower, and the gtest binary runs the full suite
# (incl. multi-second etcd tests) as a single meson test entry, so allow a large
# per-test timeout multiplier (30s base x 20 = 600s). meson itself runs every
# test in the suite and reports all failures, so this stage already completes the
# whole suite before returning.
run_stage "meson sanitizer suite" \
    meson test -C "${NIXL_BUILD_DIR}" --suite sanitizer --print-errorlogs --timeout-multiplier 20

# Real-backend binaries from test_cpp.sh. All *SAN_OPTIONS are exported; only
# those matching how the binary was built take effect, so the same script works
# for both the asan_ubsan and tsan builds. Multi-process tests (nixl_test
# target/initiator, the telemetry agent+reader pair) are excluded per the
# request -- ASan cannot track allocations across process boundaries.
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:halt_on_error=1"
export LSAN_OPTIONS="suppressions=${SAN_SUPP_DIR}/lsan.supp"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:suppressions=${SAN_SUPP_DIR}/ubsan.supp"
export TSAN_OPTIONS="halt_on_error=1:suppressions=${SAN_SUPP_DIR}/tsan.supp"
export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins
cd "${INSTALL_DIR}"
# etcd-dependent stages run first, while start_etcd_server's NIXL_ETCD_ENDPOINTS
# is still exported. A nixlAgent auto-starts an etcd comm thread whenever that var
# is set (detectEtcd() in nixl_agent.cpp); that thread opens a gRPC channel whose
# WorkStealingThreadPool calls pthread_create, which flakily trips the GCC 13.3
# libsanitizer thread-registry CHECK (the same toolchain bug gusli /
# MetadataExchange are excluded for). nixl_etcd_example is the only install-dir
# binary that needs etcd, so run it here, then drop the var.
run_stage "nixl_etcd_example" ./bin/nixl_etcd_example

# No remaining test needs etcd. Unset the endpoints so no agent starts the
# etcd/gRPC comm thread: this is what makes nixl_posix_test (a local file backend
# with no metadata exchange) and the other single-process binaries deterministic
# under the sanitizers, instead of flakily aborting inside libsanitizer's
# pthread_create interceptor.
unset NIXL_ETCD_ENDPOINTS NIXL_ETCD_NAMESPACE NIXL_ETCD_PEER_URLS

run_stage "desc_example" ./bin/desc_example
run_stage "agent_example" ./bin/agent_example
run_stage "nixl_example" ./bin/nixl_example
run_stage "ucx_backend_test" ./bin/ucx_backend_test
run_stage "nixl_posix_test" ./bin/nixl_posix_test -n 128 -s 1048576
# nixl_gusli_test is excluded from the sanitizer run entirely: under TSan it
# deterministically crashes in the GUSLI backend's registerMem (the gusli client
# library's block-device/direct-IO memory model vs the TSan runtime), and under
# ASan its GUSLIc thread init flakily trips the GCC 13.3 libsanitizer
# thread-registry crash. Both are toolchain / third-party issues, not NIXL bugs.
run_stage "ucx_backend_multi" ./bin/ucx_backend_multi
run_stage "serdes_test" ./bin/serdes_test
run_stage "test_plugin" ./bin/test_plugin

if [ -n "${SANITIZER_FAILURES}" ]; then
    echo "==== Sanitizer test FAILURES:${SANITIZER_FAILURES} ===="
    exit 1
fi
echo "==== All sanitizer tests passed ===="
