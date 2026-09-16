#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Exercise the environment independently of the backend and its runtime.
spdk=$(realpath "${1:?Usage: test_env.sh /path/to/built/spdk}")
root=$(cd "$(dirname "$0")/../../../.." && pwd)
build=$(mktemp -d "${TMPDIR:-/tmp}/nixl-spdk-env.XXXXXX")
trap 'rm -f "$build"/*.o "$build/test_env"; rmdir "$build"' EXIT
env_dir="$root/src/plugins/spdk/spdk_env"
for source in env init threads memory mem_map pci; do
    "${CC:-cc}" -g -O0 -D_GNU_SOURCE -I"$spdk/include" -I"$env_dir" \
        -c "$env_dir/$source.c" -o "$build/$source.o"
done
"${CXX:-c++}" -g -O0 -I"$spdk/include" \
    "$root/test/unit/plugins/spdk/nixl_spdk_memory_test.cpp" "$build"/*.o \
    -Wl,--start-group "$spdk/build/lib/libspdk_log.a" \
    "$spdk/build/lib/libspdk_util.a" -Wl,--end-group \
    -pthread -lnuma -luuid -o "$build/test_env"
"$build/test_env"
