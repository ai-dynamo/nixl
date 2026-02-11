#!/bin/bash -xe

set -o pipefail

function usage()
{
cat <<EOF
Usage: $0 <[options=value]>
Options:
--test_script_path            Path to the test script
--nixl_install_dir            Path to the NixL install directory
--docker_image                Docker image name
--slurm_job_id                SLURM job ID
--slurm_nodes                 Number of SLURM nodes
--slurm_head_node             SLURM head node (optional, uses SLURM_HEAD_NODE env if not set)
--container_name              Container name (optional, uses "nixl-\${BUILD_NUMBER}" if not set)
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
                slurm_head_node=*)
                    slurm_head_node=${OPTARG#*=}
                    ;;
                container_name=*)
                    container_name=${OPTARG#*=}
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


# Use environment variables as fallback
nixl_install_dir=${nixl_install_dir:-${NIXL_INSTALL_DIR}}
docker_image=${docker_image:-${DOCKER_IMAGE_NAME}}
slurm_job_id=${slurm_job_id:-${SLURM_JOB_ID}}
slurm_nodes=${slurm_nodes:-${SLURM_NODES}}
slurm_head_node=${slurm_head_node:-${SLURM_HEAD_NODE}}
container_name=${container_name:-"nixl-${BUILD_NUMBER}"}

# Validate required parameters
: ${docker_image:?Missing --docker_image}
: ${nixl_install_dir:?Missing --nixl_install_dir}
: ${test_script_path:?Missing --test_script_path}

# Build SLURM command using bash arrays (professional approach)
SLURM_CMD=(
    "srun"
    "--jobid=${slurm_job_id}"
    "--nodes=${slurm_nodes}"
    "--mpi=pmix"
    "--container-writable"
    "--container-name=${container_name}"
    "--container-image=${docker_image}"
    "${test_script_path}"
    "${nixl_install_dir}"
)

echo "INFO: Executing test script: ${test_script_path}"
echo "INFO: Using SLURM job ID: ${slurm_job_id}"
echo "INFO: Using Docker image: ${docker_image}"
echo "INFO: Container name: ${container_name}"
echo "INFO: SLURM command: ${SLURM_CMD[*]}"

# Validate SLURM_HEAD_NODE is set
if [ -z "${slurm_head_node}" ]; then
    echo "ERROR: SLURM_HEAD_NODE is not set or empty"
    exit 1
fi

# Execute based on head node type
case "${slurm_head_node}" in
    scctl)
        echo "INFO: Using scctl client to connect and execute SLURM command"
        scctl --raw-errors client connect -- "${SLURM_CMD[@]}"
        ;;
    *)
        echo "ERROR: Invalid SLURM_HEAD_NODE value: ${slurm_head_node}"
        exit 1
        ;;
esac

echo "INFO: Test execution completed successfully"
