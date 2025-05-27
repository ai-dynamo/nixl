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

# Slurm client runner - executes inside the Slurm-client container
# Required environment variables:
#   SLURM_JOB_NAME   - name of the job
#   PARTITION        - Slurm partition to use
#   TIMEOUT          - HH:MM:SS wall-time
#   WRAP_CMD         - the actual command to run on the compute node

set -euo pipefail

# Configuration
SLURM_USER="svc-nixl"
SLURM_USER_ID="148069"
SLURM_GROUP_ID="30"

# Make sure we exit cleanly on Ctrl-C
trap '[[ -n "${JOB_ID:-}" ]] && scancel "$JOB_ID" 2>/dev/null; exit 130' INT TERM

# Create user if not in Docker image
if ! id "$SLURM_USER" &>/dev/null 2>&1; then
  getent group "$SLURM_GROUP_ID" &>/dev/null 2>&1 || groupadd -g "$SLURM_GROUP_ID" "$SLURM_USER-group" 2>/dev/null
  useradd -u "$SLURM_USER_ID" -g "$SLURM_GROUP_ID" -m "$SLURM_USER" 2>/dev/null
fi

# Start MUNGE if not already running
if ! pgrep munged >/dev/null 2>&1; then
  munged --force 2>&1 | grep -v "Warning: Logfile is insecure" || true
  sleep 2
  munge -n | unmunge >/dev/null || { echo "MUNGE test failed"; exit 1; }
fi

echo "Submitting job \"$SLURM_JOB_NAME\" to partition $PARTITION (timeout $TIMEOUT)…"

# Debug mode
if [[ "${DEBUG:-0}" -gt 0 ]]; then
    echo "Debug: Checking Slurm configuration..."
    ls -la /etc/slurm/ 2>/dev/null || echo "No /etc/slurm directory"
    echo "Debug: Testing scontrol connectivity..."
    scontrol ping || echo "scontrol ping failed"
fi

# Submit job as Slurm user
JOB_ID=$(su "$SLURM_USER" -c "sbatch --parsable \
  -J \"$SLURM_JOB_NAME\" \
  -N 1 \
  -p \"$PARTITION\" \
  -t \"$TIMEOUT\" \
  -o \"\$HOME/slurm-%j.out\" \
  --wrap \"$WRAP_CMD\"")

if [[ -z "$JOB_ID" ]]; then 
    echo "sbatch returned empty job id"
    exit 1
fi

echo "Job $JOB_ID submitted - waiting…"
TERMINAL="COMPLETED FAILED TIMEOUT CANCELLED NODE_FAIL OUT_OF_MEMORY PREEMPTED"
SECONDS=0

sleep 5

while :; do
  STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || true)
  
  if [[ -z "$STATE" ]]; then
    # Try sacct first
    if FINAL_STATE=$(sacct -j "$JOB_ID" -n -o State --parsable2 2>/dev/null | head -1); then
      echo "Job $JOB_ID finished with final state: $FINAL_STATE"
      STATE="$FINAL_STATE"
      break
    fi
    
    # Try scontrol as fallback
    if JOB_INFO=$(scontrol show job "$JOB_ID" 2>/dev/null); then
      if SCONTROL_STATE=$(echo "$JOB_INFO" | grep -oP 'JobState=\K\S+' 2>/dev/null); then
        echo "Job $JOB_ID state from scontrol: $SCONTROL_STATE"
        STATE="$SCONTROL_STATE"
        break
      fi
    fi
    
    # Timeout after 10 minutes
    if (( SECONDS >= 600 )); then
      echo "Job $JOB_ID status unavailable after 10 minutes - assuming failed"
      STATE="FAILED"
      break
    fi
    
    echo "Waiting for job status update (${SECONDS}s elapsed)..."
    sleep 60
    continue
  fi
  
  echo "Job $JOB_ID status: $STATE"
  
  if [[ " $TERMINAL " == *" $STATE "* ]]; then 
      break
  fi
  sleep 60
done

# Show job output
echo "=== Job Output ==="
OUTPUT_FILE="$(su "$SLURM_USER" -c "echo \$HOME")/slurm-${JOB_ID}.out"
if [[ -f "$OUTPUT_FILE" ]]; then
    cat "$OUTPUT_FILE"
    rm -f "$OUTPUT_FILE"
else
    echo "(no output file found)"
fi

echo "Job finished with state: $STATE"

# Exit based on state
case "$STATE" in
    COMPLETED)
        exit 0
        ;;
    CANCELLED|TIMEOUT)
        exit 2
        ;;
    *)
        exit 1
        ;;
esac
