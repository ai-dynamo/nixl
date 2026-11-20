#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Run one Python example as target and initiator. Extra NAME=value arguments
# apply only to the initiator (for example, to enable telemetry there).
set -e

script=$1
shift
target_log=$(mktemp)
target_pid=
initiator_pid=

cleanup() {
    local status=$?
    local pid
    for pid in "$initiator_pid" "$target_pid"; do
        if [[ -n "$pid" ]]; then
            kill -KILL "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ $status != 0 ]]; then
        cat "$target_log" >&2
    fi
    rm -f "$target_log"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_peer() {
    local role=$1 pid=$2 status
    wait "$pid" || {
        status=$?
        echo "$role exited with status $status" >&2
        return "$status"
    }
}

NIXL_LOG_LEVEL=INFO python3 "$script" --mode=target --ip=127.0.0.1 --port=0 2>"$target_log" &
target_pid=$!

for ((attempt = 0; attempt < 30; attempt++)); do
    if ! kill -0 "$target_pid" 2>/dev/null; then
        wait_peer target "$target_pid"
        target_pid=
    fi
    port=$(awk '/MD listener is listening on port/ { print $NF; exit }' "$target_log")
    [[ -n "$port" || -z "$target_pid" ]] && break
    sleep 1
done
if [[ -z "$port" ]]; then
    echo "Target did not report its listener port" >&2
    exit 1
fi

env "$@" python3 "$script" --mode=initiator --ip=127.0.0.1 --port="$port" &
initiator_pid=$!

# A failed target can leave the initiator waiting for a notification.
while [[ -n "$target_pid" ]] && kill -0 "$initiator_pid" 2>/dev/null; do
    if ! kill -0 "$target_pid" 2>/dev/null; then
        wait_peer target "$target_pid"
        target_pid=
        break
    fi
    sleep 0.1
done
wait_peer initiator "$initiator_pid"
initiator_pid=
if [[ -n "$target_pid" ]]; then
    wait_peer target "$target_pid"
    target_pid=
fi
