#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
nixl_include=${1:?NIXL installed include directory}
nixl_lib=${2:?NIXL installed library directory}
cuda_root=${3:-/usr/local/cuda}
bench_dir=$(cd "$(dirname "$0")" && pwd)
bench_build=${PD_BENCH_BUILD:-"$bench_dir/build"}
mkdir -p "$bench_build"
"$cuda_root/bin/nvcc" -std=c++20 -O3 -arch="${PD_BENCH_ARCH:-sm_90}" \
    -c "$bench_dir/payload.cu" -o "$bench_build/payload.o"
"${CXX:-c++}" -std=c++20 -O3 -Wall -Wextra -Werror \
    "$bench_dir/replay.cpp" "$bench_build/payload.o" -o "$bench_build/replay" \
    -isystem "$nixl_include" -isystem "$cuda_root/include" \
    -L"$nixl_lib" -Wl,-rpath,"$nixl_lib" -lnixl -lnixl_build \
    -L"$cuda_root/lib64" -lcudart -ldl -pthread
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror \
    "$bench_dir/payload_test.cpp" -o "$bench_build/payload_test"
"$bench_build/payload_test"
echo "Replay: $bench_build/replay"
