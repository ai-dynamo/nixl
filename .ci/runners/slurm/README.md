# Slurm Utility Functions

A collection of bash functions for managing Slurm cluster jobs with persistent job tracking.

## Installation

Source the functions in your shell:

```bash
source contrib/runners/slurm/slurm-functions.sh
```

## Configuration

The functions use two main directories for managing jobs:

1. Job Tracking Directory (`$SLURM_TRACK_DIR`, defaults to `$HOME/.slurm_track/`)
   - Stores job metadata and tracking information
   - Directory structure:
     ```
     .slurm_track/
     ├── jobs/           # Individual job files
     │   ├── 12345      # Job file containing job details
     │   └── 12346
     └── last/          # Last submitted job info
         ├── job_id
         └── job_name
     ```

2. Output Directory (`$SLURM_OUTPUT_DIR`, defaults to `$HOME/slurm_logs/`)
   - Stores all job output logs
   - Combined stdout and stderr in a single log file
   - File naming format:
     - Regular jobs: `{job_id}_{job_name}.log`
     - Array jobs: `{array_id}_{task_id}_{job_name}.log`
   - Example:
     ```
     slurm_logs/
     ├── 12345_myjob.log
     ├── 12346_array_job_1.log
     └── 12346_array_job_2.log
     ```

You can customize these directories by setting environment variables:
```bash
export SLURM_TRACK_DIR="/path/to/tracking/dir"
export SLURM_OUTPUT_DIR="/path/to/output/dir"
```

## Job Tracking

The functions use a file-based tracking system to maintain job information across sessions. All job data is stored in `$HOME/.slurm_track/` (configurable via `SLURM_TRACK_DIR`).

Directory structure:
```
.slurm_track/
├── jobs/           # Individual job files
│   ├── 12345      # Job file containing job details
│   └── 12346
└── last/          # Last submitted job info
    ├── job_id
    └── job_name
```

## Available Functions

### Job Submission

- `slurm_submit_job <script_path> [job_name] [partition] [time_limit] [memory] [cpus] [env_vars]`
  - Submit a single job
  - Parameters:
    - script_path: Path to the job script
    - job_name: Optional, defaults to script filename without extension
    - partition: Optional, defaults to "normal"
    - time_limit: Optional, defaults to "01:00:00"
    - memory: Optional, defaults to "4G"
    - cpus: Optional, defaults to 1
    - env_vars: Optional, space-separated list of KEY=VALUE pairs for environment variables
  - Output: Combined stdout and stderr in `$SLURM_OUTPUT_DIR/{job_id}_{job_name}.log`
  - Examples:
    ```bash
    # Basic job submission
    slurm_submit_job my_script.sh "my_job" "normal" "02:00:00" "8G" 4
    # Output will be in: $SLURM_OUTPUT_DIR/12345_my_job.log

    # Job with environment variables
    slurm_submit_job my_script.sh "gpu_job" "gpu" "01:00:00" "16G" 2 "CUDA_VISIBLE_DEVICES=0,1 PATH=/usr/local/bin"
    # Output will be in: $SLURM_OUTPUT_DIR/12346_gpu_job.log
    ```

- `slurm_submit_job_array <script_path> <array_range> [job_name] [partition] [time_limit] [memory] [cpus]`
  - Submit a job array
  - Output: Combined stdout and stderr in `$SLURM_OUTPUT_DIR/{array_id}_{task_id}_{job_name}.log`
  - Example:
    ```bash
    slurm_submit_job_array my_script.sh "1-10" "array_job" "normal" "01:00:00" "4G" 1
    # Output will be in: $SLURM_OUTPUT_DIR/12347_1_array_job.log, 12347_2_array_job.log, etc.
    ```

- `slurm_submit_dependent_job <script_path> <dependency_type> <dependency_ids> [job_name] [partition] [time_limit] [memory] [cpus]`
  - Submit a job with dependencies
  - Example: `slurm_submit_dependent_job my_script.sh afterok 12345 "dependent_job"`

### Job Management

- `slurm_check_job_status [job_id]`
  - Check current status of a job
  - Example: `slurm_check_job_status` or `slurm_check_job_status 12345`

- `slurm_wait_for_job_completion [job_id] [poll_interval] [verbose]`
  - Monitor a job until it completes
  - Parameters:
    - job_id: Optional, defaults to last submitted job
    - poll_interval: Optional, polling interval in seconds (default: 5)
    - verbose: Optional, show status updates (default: true)
  - Example: `slurm_wait_for_job_completion` or `slurm_wait_for_job_completion 12345 10 false`

- `slurm_cancel_job [job_id]`
  - Cancel a running job
  - Example: `slurm_cancel_job` or `slurm_cancel_job 12345`

- `slurm_get_job_output [job_id]`
  - Display job output
  - Example: `slurm_get_job_output` or `slurm_get_job_output 12345`

- `slurm_get_job_efficiency [job_id]`
  - Show job efficiency metrics
  - Example: `slurm_get_job_efficiency` or `slurm_get_job_efficiency 12345`

- `slurm_hold_job [job_id]`
  - Hold a job (prevent it from starting)
  - Example: `slurm_hold_job` or `slurm_hold_job 12345`

- `slurm_release_job [job_id]`
  - Release a held job
  - Example: `slurm_release_job` or `slurm_release_job 12345`

- `slurm_get_job_dependencies [job_id]`
  - Show job dependencies
  - Example: `slurm_get_job_dependencies` or `slurm_get_job_dependencies 12345`

- `slurm_check_job_success [job_id]`
  - Check if a job completed successfully
  - Returns 0 if job succeeded, 1 if failed, 2 if still running
  - Stores exit status in job tracking for future reference
  - Example: `slurm_check_job_success 12345`

### Interactive Sessions

- `slurm_salloc [partition] [time_limit] [memory] [cpus] [command]`
  - Request an interactive allocation
  - Example: `slurm_salloc "normal" "01:00:00" "4G" 1` or `slurm_salloc "gpu" "02:00:00" "16G" 4 "python my_script.py"`

### Cluster Information

- `slurm_check_cluster_status [partition]`
  - Check cluster or partition status
  - Example: `slurm_check_cluster_status` or `slurm_check_cluster_status "gpu"`

### Job Tracking

- `slurm_track_job <job_id> [job_name]`
