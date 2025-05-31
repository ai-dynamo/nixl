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

# Slurm utility functions for job management
# Source this file to use the functions: source slurm-functions.sh

# Directory for job tracking
SLURM_TRACK_DIR="${SLURM_TRACK_DIR:-$HOME/.slurm_track}"
# Directory for Slurm output logs
SLURM_OUTPUT_DIR="${SLURM_OUTPUT_DIR:-$HOME/slurm_logs}"

# Initialize tracking directory
slurm_init_tracking() {
    mkdir -p "$SLURM_TRACK_DIR"
    mkdir -p "$SLURM_TRACK_DIR/jobs"
    mkdir -p "$SLURM_TRACK_DIR/last"
    # Initialize output directory
    mkdir -p "$SLURM_OUTPUT_DIR"
}

# Function to store job information
# Usage: slurm_track_job <job_id> [job_name] [output_file]
slurm_track_job() {
    local job_id="$1"
    local job_name="${2:-}"
    local output_file="${3:-}"

    if [[ -z "$job_id" ]]; then
        echo "Error: Job ID required" >&2
        return 1
    fi

    # Initialize tracking if not already done
    slurm_init_tracking

    # Store the last job ID
    echo "$job_id" > "$SLURM_TRACK_DIR/last/job_id"
    [[ -n "$job_name" ]] && echo "$job_name" > "$SLURM_TRACK_DIR/last/job_name"

    # Store job info in its own file
    local job_file="$SLURM_TRACK_DIR/jobs/$job_id"
    {
        echo "job_id=$job_id"
        echo "job_name=$job_name"
        echo "output_file=$output_file"
        echo "submit_time=$(date +%s)"
        echo "exit_status="  # Initialize empty exit status
    } > "$job_file"
}

# Function to update job exit status
# Usage: slurm_update_job_status <job_id> <exit_status>
slurm_update_job_status() {
    local job_id="$1"
    local exit_status="$2"

    if [[ -z "$job_id" ]] || [[ -z "$exit_status" ]]; then
        echo "Error: Job ID and exit status required" >&2
        return 1
    fi

    local job_file="$SLURM_TRACK_DIR/jobs/$job_id"
    if [[ -f "$job_file" ]]; then
        # Update or add exit status
        if grep -q "^exit_status=" "$job_file"; then
            sed -i "s/^exit_status=.*/exit_status=$exit_status/" "$job_file"
        else
            echo "exit_status=$exit_status" >> "$job_file"
        fi
    fi
}

# Function to check if a job completed successfully
# Usage: slurm_check_job_success [job_id]
slurm_check_job_success() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    local exit_status=

    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi

    local job_file="$SLURM_TRACK_DIR/jobs/$job_id"
    if [[ ! -f "$job_file" ]]; then
        echo "Error: Job $job_id not found in tracking" >&2
        return 1
    fi

    # Check if job is still running
    status=$(squeue -j "$job_id" -h &>/dev/null)
    if [[ -n "$status" ]]; then
        echo "Job $job_id is still running"
        return 2
    fi

    # Get exit status from job file
    exit_status=$(grep "^exit_status=" "$job_file" | cut -d'=' -f2 | tr -d ' ')

    # If no exit status in tracking, try to get it from sacct
    if [[ -z "$exit_status" ]]; then
        exit_status=$(sacct -j "$job_id" -n -o ExitCode | head -n1 | cut -d':' -f1 | tr -d ' ')
        if [[ -n "$exit_status" ]]; then
            slurm_update_job_status "$job_id" "$exit_status"
        fi
    fi

    if [[ -z "$exit_status" ]]; then
        echo "Error: Could not determine exit status for job $job_id" >&2
        return 1
    fi

    if [[ "$exit_status" == "0" ]]; then
        echo "Job $job_id completed successfully"
        return 0
    else
        echo "Job $job_id failed with exit status $exit_status"
        return 1
    fi
}

# Function to get the last job ID
# Usage: slurm_get_last_job_id
slurm_get_last_job_id() {
    if [[ -f "$SLURM_TRACK_DIR/last/job_id" ]]; then
        cat "$SLURM_TRACK_DIR/last/job_id"
    else
        return 1
    fi
}

# Function to list tracked jobs
# Usage: slurm_list_tracked_jobs
slurm_list_tracked_jobs() {
    if [[ ! -d "$SLURM_TRACK_DIR/jobs" ]]; then
        echo "No jobs currently tracked"
        return 0
    fi

    local job_count=""
    job_count=$(ls "$SLURM_TRACK_DIR/jobs" 2>/dev/null | wc -l)
    if [[ $job_count -eq 0 ]]; then
        echo "No jobs currently tracked"
        return 0
    fi

    echo "Currently tracked jobs:"
    printf "%-10s %-20s %-10s %-20s\n" "JOB ID" "NAME" "STATUS" "SUBMIT TIME"
    echo "--------------------------------------------------------"

    for job_file in "$SLURM_TRACK_DIR/jobs"/*; do
        if [[ -f "$job_file" ]]; then
            local job_id=""
            job_id=$(basename "$job_file")
            local job_name=""
            job_name=$(grep "^job_name=" "$job_file" | cut -d'=' -f2)
            local submit_time=""
            submit_time=$(grep "^submit_time=" "$job_file" | cut -d'=' -f2)
            local status=""
            status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null || echo "COMPLETED")
            local submit_date=""
            submit_date=$(date -d "@$submit_time" "+%Y-%m-%d %H:%M:%S")
            printf "%-10s %-20s %-10s %-20s\n" "$job_id" "$job_name" "$status" "$submit_date"
        fi
    done
}

# Function to clear job tracking
# Usage: slurm_clear_job_tracking [job_id]
slurm_clear_job_tracking() {
    local job_id="$1"

    if [[ -n "$job_id" ]]; then
        # Remove specific job
        rm -f "$SLURM_TRACK_DIR/jobs/$job_id"
        # Update last job if needed
        if [[ -f "$SLURM_TRACK_DIR/last/job_id" ]] && [[ $(cat "$SLURM_TRACK_DIR/last/job_id") == "$job_id" ]]; then
            rm -f "$SLURM_TRACK_DIR/last/job_id" "$SLURM_TRACK_DIR/last/job_name"
        fi
    else
        # Clear all tracking
        rm -rf "$SLURM_TRACK_DIR/jobs"/*
        rm -f "$SLURM_TRACK_DIR/last"/*
    fi
}

# Function to submit a job to Slurm
# Usage: slurm_submit_job <script_path> [job_name] [partition] [time_limit] [memory] [cpus] [env_vars]
# env_vars should be a space-separated list of KEY=VALUE pairs
slurm_submit_job() {
    local script_path="$1"
    local job_name="${2:-$(basename "$script_path" .sh)}"
    local partition="${3:-normal}"
    local time_limit="${4:-01:00:00}"
    local memory="${5:-4G}"
    local cpus="${6:-1}"
    local env_vars="${7:-}"
    local output_dir="$SLURM_OUTPUT_DIR"

    if [[ ! -f "$script_path" ]]; then
        echo "Error: Script file not found: $script_path" >&2
        return 1
    fi

    # Ensure output directory exists
    mkdir -p "$output_dir"

    # Build sbatch command with environment variables if provided
    local sbatch_cmd="sbatch --job-name=\"$job_name\" \
           --partition=\"$partition\" \
           --time=\"$time_limit\" \
           --mem=\"$memory\" \
           --cpus-per-task=\"$cpus\" \
           --output=\"$output_dir/%j_%x.log\" \
           --error=\"$output_dir/%j_%x.log\""

    # Add environment variables if provided
    if [[ -n "$env_vars" ]]; then
        # Convert space-separated KEY=VALUE pairs to --export format
        local export_vars=""
        for var in $env_vars; do
            if [[ "$var" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
                export_vars="${export_vars:+$export_vars,}$var"
            else
                echo "Warning: Invalid environment variable format: $var" >&2
            fi
        done
        if [[ -n "$export_vars" ]]; then
            sbatch_cmd="$sbatch_cmd --export=\"$export_vars\""
        fi
    fi

    # Add script path and execute
    sbatch_cmd="$sbatch_cmd \"$script_path\""
    local job_id=""
    job_id=$(eval "$sbatch_cmd" | grep -o '[0-9]\+')

    if [[ -n "$job_id" ]]; then
        slurm_track_job "$job_id" "$job_name" "$output_dir/${job_id}_${job_name}.log"
        echo "Submitted job $job_id ($job_name)"
        echo "Output will be written to: $output_dir/${job_id}_${job_name}.log"
    fi
}

# Function to submit a job array
# Usage: slurm_submit_job_array <script_path> <array_range> [job_name] [partition] [time_limit] [memory] [cpus]
slurm_submit_job_array() {
    local script_path="$1"
    local array_range="$2"
    local job_name="${3:-$(basename "$script_path" .sh)}"
    local partition="${4:-normal}"
    local time_limit="${5:-01:00:00}"
    local memory="${6:-4G}"
    local cpus="${7:-1}"
    local output_dir="$SLURM_OUTPUT_DIR"

    if [[ ! -f "$script_path" ]]; then
        echo "Error: Script file not found: $script_path" >&2
        return 1
    fi

    # Ensure output directory exists
    mkdir -p "$output_dir"

    local job_id=""
    job_id=$(sbatch --job-name="$job_name" \
           --partition="$partition" \
           --time="$time_limit" \
           --mem="$memory" \
           --cpus-per-task="$cpus" \
           --array="$array_range" \
           --output="$output_dir/%A_%a_%x.log" \
           --error="$output_dir/%A_%a_%x.log" \
           "$script_path" | grep -o '[0-9]\+')

    if [[ -n "$job_id" ]]; then
        slurm_track_job "$job_id" "$job_name" "$output_dir/${job_id}_%a_${job_name}.log"
        echo "Submitted job array $job_id ($job_name) with range $array_range"
        echo "Output will be written to: $output_dir/${job_id}_%a_${job_name}.log"
    fi
}

# Function to submit a job with dependencies
# Usage: slurm_submit_dependent_job <script_path> <dependency_type> <dependency_ids> [job_name] [partition] [time_limit] [memory] [cpus]
slurm_submit_dependent_job() {
    local script_path="$1"
    local dep_type="$2"
    local dep_ids="$3"
    local job_name="${4:-$(basename "$script_path" .sh)}"
    local partition="${5:-normal}"
    local time_limit="${6:-01:00:00}"
    local memory="${7:-4G}"
    local cpus="${8:-1}"
    local output_dir="$SLURM_OUTPUT_DIR"
    local job_id=""

    if [[ ! -f "$script_path" ]]; then
        echo "Error: Script file not found: $script_path" >&2
        return 1
    fi

    # Ensure output directory exists
    mkdir -p "$output_dir"

    job_id=$(sbatch --job-name="$job_name" \
           --partition="$partition" \
           --time="$time_limit" \
           --mem="$memory" \
           --cpus-per-task="$cpus" \
           --dependency="$dep_type:$dep_ids" \
           --output="$output_dir/%j_%x.log" \
           --error="$output_dir/%j_%x.log" \
           "$script_path" | grep -o '[0-9]\+')

    if [[ -n "$job_id" ]]; then
        slurm_track_job "$job_id" "$job_name" "$output_dir/${job_id}_${job_name}.log"
        echo "Submitted dependent job $job_id ($job_name) with dependency $dep_type:$dep_ids"
        echo "Output will be written to: $output_dir/${job_id}_${job_name}.log"
    fi
}

# Function to cancel a job
# Usage: slurm_cancel_job [job_id]
slurm_cancel_job() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi
    scancel "$job_id"
    slurm_clear_job_tracking "$job_id"
    echo "Cancelled job $job_id"
}

# Function to check job status
# Usage: slurm_check_job_status [job_id]
slurm_check_job_status() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi
    squeue -j "$job_id" -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"
}

# Function to get job output
# Usage: slurm_get_job_output [job_id]
slurm_get_job_output() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    local output_dir="$SLURM_OUTPUT_DIR"
    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi

    # Try to find the output file
    local job_name=""
    job_name=$(grep "^job_name=" "$SLURM_TRACK_DIR/jobs/$job_id" 2>/dev/null | cut -d'=' -f2)
    local output_file="$output_dir/${job_id}_${job_name}.log"

    if [[ -f "$output_file" ]]; then
        cat "$output_file"
    else
        echo "No output file found for job $job_id in $output_dir"
        return 1
    fi
}

# Function to get job efficiency metrics
# Usage: slurm_get_job_efficiency [job_id]
slurm_get_job_efficiency() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi
    seff "$job_id"
}

# Function to hold a job
# Usage: slurm_hold_job [job_id]
slurm_hold_job() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi
    scontrol hold "$job_id"
}

# Function to release a held job
# Usage: slurm_release_job [job_id]
slurm_release_job() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi
    scontrol release "$job_id"
}

# Function to get job dependencies
# Usage: slurm_get_job_dependencies [job_id]
slurm_get_job_dependencies() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi
    scontrol show job "$job_id" | grep "Dependency"
}

# Function to check cluster status
# Usage: slurm_check_cluster_status [partition]
slurm_check_cluster_status() {
    local partition="${1:-}"
    if [[ -n "$partition" ]]; then
        sinfo -p "$partition"
    else
        sinfo
    fi
}

# Function to request an interactive allocation
# Usage: slurm_allocate_resources [partition] [time_limit] [memory] [cpus] [command]
slurm_allocate_resources() {
    local partition="${1:-normal}"
    local time_limit="${2:-01:00:00}"
    local memory="${3:-4G}"
    local cpus="${4:-1}"
    local command="${5:-}"

    local salloc_args=(
        "--partition=$partition"
        "--time=$time_limit"
        "--mem=$memory"
        "--cpus-per-task=$cpus"
    )

    if [[ -n "$command" ]]; then
        # If a command is provided, execute it in the allocation
        salloc "${salloc_args[@]}" "$command"
    else
        # Otherwise, start an interactive shell
        echo "Requesting interactive allocation on partition $partition"
        echo "Time limit: $time_limit, Memory: $memory, CPUs: $cpus"
        echo "Type 'exit' to end the allocation"
        salloc "${salloc_args[@]}" bash
    fi
}

# Function to wait for job completion
# Usage: slurm_wait_for_job_completion [job_id] [poll_interval] [verbose]
slurm_wait_for_job_completion() {
    local job_id="${1:-$(slurm_get_last_job_id)}"
    local poll_interval="${2:-5}"
    local verbose="${3:-true}"

    if [[ -z "$job_id" ]]; then
        echo "Error: No job ID provided and no last job tracked" >&2
        return 1
    fi

    if [[ "$verbose" == "true" ]]; then
        echo "Waiting for job $job_id to complete..."
        echo "Polling every ${poll_interval} seconds"
        echo "Press Ctrl+C to stop monitoring"
        echo "----------------------------------------"
    fi

    while true; do
        local status=
        status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        if [[ -z "$status" ]]; then
            # Job is no longer in queue, check if it completed successfully
            local job_name=""
            job_name=$(grep "^job_name=" "$SLURM_TRACK_DIR/jobs/$job_id" 2>/dev/null | cut -d'=' -f2)
            local output_file="$SLURM_OUTPUT_DIR/${job_id}_${job_name}.log"

            if [[ -f "$output_file" ]]; then
                # Get exit status from sacct
                local exit_status=
                exit_status=$(sacct -j "$job_id" -n -o ExitCode | head -n1 | cut -d':' -f1 | tr -d ' ')
                slurm_update_job_status "$job_id" "$exit_status"

                if [[ "$verbose" == "true" ]]; then
                    if [[ "$exit_status" == "0" ]]; then
                        echo "Job $job_id has completed successfully!"
                    else
                        echo "Job $job_id has completed with exit status $exit_status"
                    fi
                    echo "----------------------------------------"
                fi
                return "$exit_status"
            else
                if [[ "$verbose" == "true" ]]; then
                    echo "Job $job_id has ended but no output file found"
                    echo "----------------------------------------"
                fi
                return 1
            fi
        fi

        if [[ "$verbose" == "true" ]]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Job status: $status"
        fi

        sleep "$poll_interval"
    done
}

# Print usage information
slurm_help() {
    echo "Available Slurm utility functions:"
    echo "  slurm_submit_job <script_path> [job_name] [partition] [time_limit] [memory] [cpus] [env_vars]"
    echo "    env_vars: Space-separated list of KEY=VALUE pairs (e.g., \"CUDA_VISIBLE_DEVICES=0,1 PATH=/usr/local/bin\")"
    echo "  slurm_submit_job_array <script_path> <array_range> [job_name] [partition] [time_limit] [memory] [cpus]"
    echo "  slurm_check_job_status [job_id]"
    echo "  slurm_cancel_job [job_id]"
    echo "  slurm_get_job_output [job_id]"
    echo "  slurm_check_cluster_status [partition]"
    echo "  slurm_get_job_efficiency [job_id]"
    echo "  slurm_hold_job [job_id]"
    echo "  slurm_release_job [job_id]"
    echo "  slurm_get_job_dependencies [job_id]"
    echo "  slurm_submit_dependent_job <script_path> <dependency_type> <dependency_ids> [job_name] [partition] [time_limit] [memory] [cpus]"
    echo "  slurm_allocate_resources [partition] [time_limit] [memory] [cpus] [command]"
    echo "  slurm_track_job <job_id> [job_name]"
    echo "  slurm_list_tracked_jobs"
    echo "  slurm_clear_job_tracking [job_id]"
    echo "  slurm_get_last_job_id"
    echo "  slurm_check_job_success [job_id]"
    echo "  slurm_wait_for_job_completion [job_id] [poll_interval] [verbose]"
    echo "  slurm_help"
}

# Initialize tracking directory when script is sourced
slurm_init_tracking
