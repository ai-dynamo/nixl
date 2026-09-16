#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail
binary=${1:?Usage: test_spdk_kv_config.sh /path/to/nixlbench}
common=(--backend SPDK --spdk_mode kv --spdk_json_config_file /nonexistent/spdk-kv.json
        --spdk_bdev_name Kv0 --use_hugepages --initiator_seg_type DRAM
        --total_buffer_size 2097152 --start_block_size 4096 --max_block_size 4096)
reject() {
    local expected=$1 output rc
    shift
    rc=0
    output=$("$binary" "${common[@]}" "$@" 2>&1) || rc=$?
    if [[ $rc != 1 || $output != *"$expected"* ]]; then
        echo "Unexpected result ($rc): $output" >&2
        exit 1
    fi
}
reject '--spdk_mode must be' --spdk_mode invalid
reject 'SPDK KV requires shared' --use_hugepages=false
reject 'SPDK KV requires shared' --spdk_bdev_offset 4096
reject 'SPDK KV requires shared' --start_block_size 512
reject 'SPDK KV requires shared' --randomize_location_mode bytealigned
reject 'distinct payload storage' --total_buffer_size 2097152 --max_batch_size 1024
reject 'Consistency check is not supported' --spdk_mode block --check_consistency
echo 'PASS: SPDK KV configuration rejects incompatible modes and overlapping payloads'
