#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Brings up an SPDK nvmf target with a *librados-backed* KV namespace over
# VFIOUSER (real Ceph/RADOS), then runs the RADOS_NKV direct-engine round-trip
# test against it. This proves the SAME plugin/test that round-trips against the
# in-memory kvdev (run_roundtrip.sh) also works unchanged against a librados KV
# namespace, and that the stored value actually lands as an object in the
# tenant rados namespace.
#
# Modeled on:
#   - src/plugins/rados-nkv/run_roundtrip.sh                  (the in-mem bring-up)
#   - $SPDK_ROOT/test/nvmf/kv_rados/kv_rados_vfio_user.sh     (librados kvdev wiring)
#
# The librados kvdev maps the NVMe KV key to a lowercase-hex oid
# (kvdev_rados_key_to_oid). The engine derives that key by hashing the token
# sequence in the descriptor's metaInfo (radosNkvDeriveKey: 128-bit FNV-1a).
# The round-trip test uses the token sequence "nixl-key-0000001"; its oid is the
# hex of the 16-byte hash. After the WRITE we assert via the rados CLI that this
# oid exists in $KV_POOL/$KV_NS with a size matching the stored value length,
# then the test's READ + queryMem prove Retrieve and Exist. The test object is
# removed at the end so the pool is not polluted.
set -euo pipefail

SPDK_ROOT="${SPDK_ROOT:-/mnt/spdk}"
TEST_BIN="$(dirname "$0")/../../../builddir/src/plugins/rados-nkv/rados_nkv_roundtrip_test"
TEST_BIN="$(readlink -f "$TEST_BIN")"

# --- Ceph / RADOS environment (override via env to match your cluster) -----
# Defaults assume a standard Ceph install; point these at your cluster's config,
# keyring, and rados binary as needed.
CEPH_CONF="${CEPH_CONF:-/etc/ceph/ceph.conf}"
CEPH_KEYRING="${CEPH_KEYRING:-/etc/ceph/ceph.client.admin.keyring}"
RADOS_BIN="${RADOS_BIN:-rados}"
CEPH_USER="${CEPH_USER:-admin}"
KV_POOL="${KV_POOL:-kvpool}"
KV_NS="${KV_NS:-kvns}"

# The round-trip test (rados_nkv_roundtrip_test.cpp) uses this token sequence
# and this exact payload; keep them in sync with that file.
KV_KEY="nixl-key-0000001"
PAYLOAD="RADOS_NKV-NIXL-roundtrip-slice2-0123456789"
VAL_LEN=${#PAYLOAD}
# oid = lowercase hex of the derived 16-byte KV key. The engine derives the key
# as a 128-bit FNV-1a hash of the token sequence; reproduce it here (this MUST
# stay in sync with radosNkvDeriveKey in rados_nkv_backend.cpp).
OID_HEX=$(python3 -c '
import sys
seq = sys.argv[1].encode()
h = 0x6c62272e07bb014262b821756295c58d            # FNV-1a 128-bit offset basis
prime = 0x0000000001000000000000000000013b        # FNV-1a 128-bit prime
mask = (1 << 128) - 1
for b in seq:
    h ^= b
    h = (h * prime) & mask
print(h.to_bytes(16, "big").hex())               # 16-byte key, big-endian, as hex
' "$KV_KEY")

rados() { "$RADOS_BIN" -c "$CEPH_CONF" -k "$CEPH_KEYRING" -p "$KV_POOL" -N "$KV_NS" "$@"; }

nqn="nqn.2026-06.io.spdk:rados-kv-cnode0"
cluster_name="ceph0"
kvdev_name="RadosNkv0"
sock_dir="$(mktemp -d /tmp/rados_nkv_rados_rt.XXXXXX)"
muser_dir="$sock_dir/domain/muser0/0"
rpc_sock="$sock_dir/rpc.sock"
mkdir -p "$muser_dir"

# rpc.py needs to talk to the (possibly sudo-owned) RPC socket. RPC_PFX is set
# to "sudo" when the target runs under sudo so the helper can reach the socket.
RPC_PFX=""
rpc_py() { $RPC_PFX "$SPDK_ROOT/scripts/rpc.py" -s "$rpc_sock" "$@"; }

nvmfpid=""
SUDO_RUN=""
cleanup() {
    if [[ -n "$nvmfpid" ]]; then
        $SUDO_RUN kill "$nvmfpid" 2>/dev/null || true
        wait "$nvmfpid" 2>/dev/null || true
    fi
    # Remove the test object so we do not pollute the pool.
    rados rm "$OID_HEX" >/dev/null 2>&1 || true
    $SUDO_RUN rm -rf "$sock_dir" 2>/dev/null || rm -rf "$sock_dir" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_rpc() {
    for _ in $(seq 1 50); do
        if rpc_py rpc_get_methods >/dev/null 2>&1; then return 0; fi
        sleep 0.2
    done
    return 1
}

echo "== Ceph target: pool=$KV_POOL ns=$KV_NS user=$CEPH_USER =="
echo "== oid for key '$KV_KEY' = $OID_HEX (hex-encoded), expected size=$VAL_LEN =="

# Pre-clean any stale object from a previous aborted run.
rados rm "$OID_HEX" >/dev/null 2>&1 || true

# --- Start the target ------------------------------------------------------
# Try --no-huge first (no special privileges). librados/DMA may require real
# hugepages; if the no-huge target fails to come up, fall back to sudo +
# hugepages exactly like the SPDK kv_rados tests (-m 0x3 --iova-mode=va).
echo "== starting nvmf_tgt (attempt 1: --no-huge -s 1024) =="
"$SPDK_ROOT/build/bin/nvmf_tgt" -r "$rpc_sock" -m 0x1 --no-huge -s 1024 \
    >"$sock_dir/tgt.log" 2>&1 &
nvmfpid=$!

if ! wait_for_rpc; then
    echo "== --no-huge target did not come up; falling back to sudo + hugepages =="
    sed -n '1,200p' "$sock_dir/tgt.log" || true
    kill "$nvmfpid" 2>/dev/null || true
    wait "$nvmfpid" 2>/dev/null || true
    nvmfpid=""

    SUDO_RUN="sudo"
    RPC_PFX="sudo"
    echo "== starting nvmf_tgt (attempt 2: sudo, hugepages, -m 0x3 --iova-mode=va) =="
    sudo "$SPDK_ROOT/build/bin/nvmf_tgt" -r "$rpc_sock" -m 0x3 --iova-mode=va \
        >"$sock_dir/tgt.log" 2>&1 &
    nvmfpid=$!
    if ! wait_for_rpc; then
        echo "FAIL: target did not come up under sudo either"
        sed -n '1,200p' "$sock_dir/tgt.log" || true
        exit 1
    fi
    TGT_MODE="sudo+hugepages"
else
    TGT_MODE="--no-huge -s 1024"
fi
echo "== target up via: $TGT_MODE =="

# --- Wire up the librados KV namespace -------------------------------------
echo "== configuring librados KV namespace =="
rpc_py nvmf_create_transport -t VFIOUSER
rpc_py kvdev_rados_register_cluster "$cluster_name" \
    --user "$CEPH_USER" --config-file "$CEPH_CONF" --key-file "$CEPH_KEYRING"
rpc_py kvdev_rados_create "$kvdev_name" "$cluster_name" "$KV_POOL" --namespace "$KV_NS"
rpc_py nvmf_create_subsystem "$nqn" -s SPDKKVR01 -a
rpc_py nvmf_subsystem_add_kv_ns "$nqn" "$kvdev_name"
rpc_py nvmf_subsystem_add_listener "$nqn" -t VFIOUSER -a "$muser_dir" -s 0

# The VFIOUSER socket dir may be sudo-owned; make sure the (non-sudo) test
# binary can open it.
if [[ -n "$SUDO_RUN" ]]; then
    sudo chmod -R a+rwx "$sock_dir" || true
fi

# --- Run the round-trip test (Store / Retrieve / queryMem over librados) ----
echo "== running round-trip test against librados KV namespace =="
"$TEST_BIN" "$muser_dir"
rc=$?
echo "== round-trip test exit code: $rc =="
if [[ $rc -ne 0 ]]; then
    echo "FAIL: round-trip test failed"
    exit $rc
fi

# --- Verify the value actually landed in RADOS -----------------------------
# The round-trip test's WRITE issued a KV Store; the librados kvdev wrote it as
# object oid=hex(key) in $KV_POOL/$KV_NS. Confirm it exists with the right size.
echo "== RADOS verification: object must exist in $KV_POOL/$KV_NS =="
echo "--- rados ls (grep for $OID_HEX) ---"
ls_out="$(rados ls 2>/dev/null)"
echo "$ls_out" | grep -F "$OID_HEX" || { echo "FAIL: oid $OID_HEX not in rados ls"; exit 1; }

echo "--- rados stat $OID_HEX ---"
stat_out="$(rados stat "$OID_HEX")"
echo "$stat_out"

# Parse the size from `rados stat` output ("... size <N> ...") and assert it
# matches the stored value length.
got_size="$(echo "$stat_out" | grep -oE 'size [0-9]+' | grep -oE '[0-9]+' | head -1)"
if [[ "$got_size" != "$VAL_LEN" ]]; then
    echo "FAIL: rados object size $got_size != expected stored value length $VAL_LEN"
    exit 1
fi
echo "== RADOS object $OID_HEX present in $KV_POOL/$KV_NS, size=$got_size matches stored value length $VAL_LEN =="

echo "run_roundtrip_rados: PASS"
exit 0
