#!/bin/bash
set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function usage()
{
cat <<EOF
Usage: $0 <[options=value]>
Options:
--slurm_job_id_file           File path to read SLURM job ID from
--slurm_job_id                SLURM job ID (direct)
--slurm_head_node             SLURM head node
--workspace                   Workspace directory
EOF
exit 1
}

while getopts ":h-:" optchar; do
    case "${optchar}" in
        -)
            case "${OPTARG}" in
                slurm_job_id_file=*)
                    slurm_job_id_file=${OPTARG#*=}
                    ;;
                slurm_job_id=*)
                    slurm_job_id=${OPTARG#*=}
                    ;;
                slurm_head_node=*)
                    slurm_head_node=${OPTARG#*=}
                    ;;
                workspace=*)
                    workspace=${OPTARG#*=}
                    ;;
                *)
                    echo "Unknown option --${OPTARG}" >&2
                    exit 1
                    ;;
            esac;;
        h | *)
            usage
            exit 0
            ;;
    esac
done

slurm_job_id=${slurm_job_id:-${SLURM_JOB_ID}}
slurm_job_id_file=${slurm_job_id_file:-${SLURM_JOB_ID_FILE}}
slurm_head_node=${slurm_head_node:-${SLURM_HEAD_NODE}}
workspace=${workspace:-${WORKSPACE}}

# Set default job ID file path if not specified
if [ -z "${slurm_job_id_file}" ] && [ -n "${workspace}" ]; then
    slurm_job_id_file="${workspace}/job_id.txt"
fi

# Get job ID from file if not provided directly
if [ -z "${slurm_job_id}" ]; then
    if [ -n "${slurm_job_id_file}" ] && [ -f "${slurm_job_id_file}" ]; then
        slurm_job_id=$(cat "${slurm_job_id_file}")
        echo "INFO: Read job ID from ${slurm_job_id_file}: ${slurm_job_id}"
    else
        echo "ERROR: No job ID provided and job ID file not found: ${slurm_job_id_file}"
        exit 1
    fi
fi

: ${slurm_job_id:?Missing SLURM job ID}
: ${slurm_head_node:?Missing SLURM head node}

readonly SLURM_STOP_ALLOCATION_CMD="scancel ${slurm_job_id}"

echo "INFO: Stopping SLURM job: ${slurm_job_id}"

case "${slurm_head_node}" in
    scctl)
        echo "INFO: Using scctl client to stop Slurm resources"
        export SCCTL_USER=${SERVICE_USER_USERNAME}
        export SCCTL_PASSWORD=${SERVICE_USER_PASSWORD}
        scctl -v
        scctl --raw-errors upgrade
        scctl --raw-errors login
        result=$(scctl --raw-errors client exists)
        if [ "$result" == "client does not exist" ]; then
            echo "INFO: Creating scctl client"
            scctl --raw-errors client create
        fi
        echo "INFO: Executing scancel for job ${slurm_job_id}"
        scctl --raw-errors client connect -- "${SLURM_STOP_ALLOCATION_CMD}"
        ;;
    *)
        echo "ERROR: Invalid SLURM_HEAD_NODE value: ${slurm_head_node}"
        exit 1
        ;;
esac

echo "INFO: SLURM job ${slurm_job_id} stopped successfully"
