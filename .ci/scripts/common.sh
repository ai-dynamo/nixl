#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Common functions for CI scripts
#

#
# Set initial port number for client/server applications to be updated with
# function below
#
tcp_port_range=1000
min_port_number=10500
max_port_number=65535

# GITLAB CI
if [ -n "$CI_CONCURRENT_ID" ]; then
    nixl_concurrent_id=$CI_CONCURRENT_ID
# Jenkins CI
elif [ -n "$EXECUTOR_NUMBER" ]; then
    nixl_concurrent_id=$EXECUTOR_NUMBER
else
    # Fallback to random number if both CI_CONCURRENT_ID and EXECUTOR_NUMBER are not set
    nixl_concurrent_id=$((RANDOM % $(((max_port_number - min_port_number) / tcp_port_range))))
fi

echo nixl_concurrent_id="$nixl_concurrent_id"

# First half of the port range is used for shell script tests
tcp_port_min=$((min_port_number + nixl_concurrent_id * tcp_port_range))
tcp_port_max=$((tcp_port_min + tcp_port_range / 2))

get_next_tcp_port() {
    local port_file="/tmp/nixl_tcp_port_${nixl_concurrent_id}"

    if [ ! -f "$port_file" ]; then
        echo "$tcp_port_min" > "$port_file"
    fi

    local current_port
    current_port=$(cat "$port_file")
    local next_port=$((current_port + 1))

    # Check if the port is already in use
    while ss -tuln | grep -q :$next_port; do
        next_port=$((next_port + 1))
    done

    if [ "$next_port" -ge "$tcp_port_max" ]; then
        next_port="$tcp_port_min"
    fi

    echo "$next_port" > "$port_file"

    echo "$next_port"
}

# Second half of the port range is used for gtest
gtest_offset=$((tcp_port_range / 2))
# shellcheck disable=SC2034
min_gtest_port=$((tcp_port_min + gtest_offset))
# shellcheck disable=SC2034
max_gtest_port=$((tcp_port_max + gtest_offset))
