#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
# SPDX-License-Identifier: Apache-2.0
# Brings up an SPDK nvmf target with an in-memory KV namespace over VFIOUSER,
# then runs the RADOS_NKV direct-engine round-trip test against it.
set -euo pipefail

SPDK_ROOT="${SPDK_ROOT:-/mnt/spdk}"
TEST_BIN="$(dirname "$0")/../../../builddir/src/plugins/rados-nkv/rados_nkv_roundtrip_test"
TEST_BIN="$(readlink -f "$TEST_BIN")"

nqn="nqn.2026-06.io.spdk:rados-kv-cnode0"
kvdev_name="RadosNkvMem0"
sock_dir="$(mktemp -d /tmp/rados_nkv_rt.XXXXXX)"
muser_dir="$sock_dir/domain/muser0/0"
rpc_sock="$sock_dir/rpc.sock"
rpc_py="$SPDK_ROOT/scripts/rpc.py -s $rpc_sock"
mkdir -p "$muser_dir"

nvmfpid=""
cleanup() {
    [[ -n "$nvmfpid" ]] && kill "$nvmfpid" 2>/dev/null || true
    rm -rf "$sock_dir"
}
trap cleanup EXIT

echo "== starting nvmf_tgt =="
"$SPDK_ROOT/build/bin/nvmf_tgt" -r "$rpc_sock" -m 0x1 --no-huge -s 512 &
nvmfpid=$!

# Wait for the RPC socket to be ready.
for _ in $(seq 1 50); do
    if $rpc_py rpc_get_methods >/dev/null 2>&1; then break; fi
    sleep 0.2
done

echo "== configuring KV namespace =="
$rpc_py nvmf_create_transport -t VFIOUSER
$rpc_py kvdev_mem_create "$kvdev_name"
$rpc_py nvmf_create_subsystem "$nqn" -s SPDKKV001 -a
$rpc_py nvmf_subsystem_add_kv_ns "$nqn" "$kvdev_name"
$rpc_py nvmf_subsystem_add_listener "$nqn" -t VFIOUSER -a "$muser_dir" -s 0

echo "== running round-trip test =="
"$TEST_BIN" "$muser_dir"
rc=$?
echo "== test exit code: $rc =="
exit $rc
