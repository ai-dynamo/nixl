#!/bin/bash
#SBATCH --job-name=pre_both
#SBATCH --partition=ISR1-PRE
#SBATCH --reservation=nonRA
#SBATCH --nodes=2
#SBATCH --nodelist=hgx-isr1-pre-01,hgx-isr1-pre-13
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/vast/users/mtennenhaus/kvbench_results/pre_rdma_storage_%j.out

set -e

export TMPDIR=/mnt/vast/users/mtennenhaus/tmp
export ENROOT_TEMP_DIR=/mnt/vast/users/mtennenhaus/tmp

VAST_BASE="/mnt/vast/users/mtennenhaus"
STORAGE_PATH="$VAST_BASE/kvbench_storage"
CONFIG_DIR="$VAST_BASE/nixl/benchmark/kvbench/test_plans/isr1_pre_test"
TEST_DIR="$VAST_BASE/kvbench_results"
LOCAL_KVBENCH="$VAST_BASE/nixl/benchmark/kvbench"

CONTAINER="/.autodirect/mswg2/E2E/Regression_logs/squash/nixlbench/gitlab-master.nvidia.com+eshukrun+warehouse+nixlbench+v0.1.0.dev.69b633aa.sqsh"
CONTAINER_PYTHON="/workspace/nixl/.venv/bin/python"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=2379

export NIXL_ETCD_ENDPOINTS="http://$MASTER_ADDR:$MASTER_PORT"
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/pre_both_${SLURM_JOB_ID}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export UCX_TLS=rc,cuda_copy

echo "=============================================="
echo "ISR1-PRE RDMA + Storage Test (GDS on Vast)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

srun -n 2 --ntasks-per-node=1 bash -c "mkdir -p $TMPDIR && mkdir -p $TEST_DIR && mkdir -p $STORAGE_PATH && rm -rf $STORAGE_PATH/*"

BIND_SCRIPT="$CONFIG_DIR/bind_8gpus.sh"
MOUNTS="$LOCAL_KVBENCH:/workspace/nixl/benchmark/kvbench,$CONFIG_DIR:$CONFIG_DIR,$TEST_DIR:$TEST_DIR,$BIND_SCRIPT:$BIND_SCRIPT,$STORAGE_PATH:$STORAGE_PATH"

echo "Loading container..."
srun -n 2 --ntasks-per-node=1 --container-image="$CONTAINER" --container-mounts="$MOUNTS" true

echo "Starting ETCD..."
srun -N1 -n1 -w $MASTER_ADDR bash -c "
    docker run -d --name etcd_$SLURM_JOB_ID --network host \
        quay.io/coreos/etcd:v3.5.0 \
        etcd --listen-client-urls http://0.0.0.0:$MASTER_PORT \
             --advertise-client-urls http://$MASTER_ADDR:$MASTER_PORT \
             --data-dir /tmp/etcd_$SLURM_JOB_ID 2>/dev/null || true
"
sleep 5

echo "Running RDMA + Storage benchmark..."
srun --mpi=pmix --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    bash -c "
        $BIND_SCRIPT $CONTAINER_PYTHON /workspace/nixl/benchmark/kvbench/main.py \
            sequential-ct-perftest $CONFIG_DIR/config_rdma_storage.yaml \
            --storage-backend GDS_MT \
            --storage-path $STORAGE_PATH \
            --storage-direct-io \
            --json-output-path $TEST_DIR/pre_rdma_storage_$SLURM_JOB_ID.json
    "

srun -N1 -n1 -w $MASTER_ADDR bash -c "docker stop etcd_$SLURM_JOB_ID; docker rm etcd_$SLURM_JOB_ID" 2>/dev/null || true
echo "Done: $TEST_DIR/pre_rdma_storage_${SLURM_JOB_ID}.json"
