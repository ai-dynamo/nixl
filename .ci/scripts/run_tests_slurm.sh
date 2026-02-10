#!/bin/bash -xe

set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function usage()
{
cat <<EOF
Usage: $SCRIPT <[options=value]>
Options:
--test_script_path            Path to the test script
--nixl_install_dir            Path to the NixL install directory
--docker_image                Docker image name
--slurm_job_id                SLURM job ID
--slurm_nodes                 Number of SLURM nodes
EOF
exit 1
}

[ $# -eq 0 ] && usage
while getopts ":h-:" optchar; do
    case "${optchar}" in
        -)
            case "${OPTARG}" in
                test_script_path=*)
                    test_script_path=${OPTARG#*=}
                    ;;
                nixl_install_dir=*)
                    nixl_install_dir=${OPTARG#*=}
                    ;;
                docker_image=*)
                    docker_image=${OPTARG#*=}
                    ;;
                slurm_job_id=*)
                    slurm_job_id=${OPTARG#*=}
                    ;;
                slurm_nodes=*)
                    slurm_nodes=${OPTARG#*=}
                    ;;
                *)
                    if [ "$OPTERR" = 1 ] && [ "${optspec:0:1}" != ":" ]; then
                        echo "Unknown option --${OPTARG}" >&2
                        exit 1
                    fi
                    ;;
            esac;;
        h | *)
            usage
            exit 0
            ;;
    esac
done

: ${test_script_path:?Missing --test_script_path}
nixl_install_dir=${nixl_install_dir:-${NIXL_INSTALL_DIR}}
docker_image=${docker_image:-${DOCKER_IMAGE_NAME}}
slurm_job_id=${slurm_job_id:-${SLURM_JOB_ID}}
slurm_nodes=${slurm_nodes:-${SLURM_NODES}}

readonly SLURM_COMMAND="srun --jobid=${slurm_job_id} --nodes=${slurm_nodes} --mpi=pmix --container-writable --container-image=${docker_image} ${test_script_path} ${nixl_install_dir}"

# Validate SLURM_HEAD_NODE is set
if [ -z "${SLURM_HEAD_NODE}" ]; then
    echo "ERROR: SLURM_HEAD_NODE is not set or empty"
    exit 1
fi

# Execute based on head node type
case "${SLURM_HEAD_NODE}" in
    scctl)
        echo "Using scctl client to connect and execute slurm command"
        scctl --raw-errors client connect -- "${SLURM_COMMAND}"
        ;;
    *)
        echo "ERROR: Invalid SLURM_HEAD_NODE value: ${SLURM_HEAD_NODE}"
        exit 1
        ;;
esac
