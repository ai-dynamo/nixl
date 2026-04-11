#!/bin/bash
#SBATCH --job-name=pre_posix_lg
#SBATCH --partition=ISR1-PRE
#SBATCH --reservation=nonRA
#SBATCH --nodes=2
#SBATCH --nodelist=hgx-isr1-pre-01,hgx-isr1-pre-13
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/mtrsysgwork/mtennenhaus/kvbench_results/pre_posix_lg_%j.out

set -e

export TMPDIR=/mnt/vast/users/mtennenhaus/tmp
export ENROOT_TEMP_DIR=/mnt/vast/users/mtennenhaus/tmp

# mtrsysgwork is mounted at /auto/mtrsysgwork on isr1-pre nodes
MTRSYS_BASE="/auto/mtrsysgwork/mtennenhaus"
VAST_BASE="/mnt/vast/users/mtennenhaus"

# Code and config from mtrsysgwork (no need to copy to VAST)
LOCAL_KVBENCH="$MTRSYS_BASE/nixl/benchmark/kvbench"
CONFIG_DIR="$MTRSYS_BASE/nixl/benchmark/kvbench/test_plans/isr1_pre_test"
TEST_DIR="$MTRSYS_BASE/kvbench_results"

# Storage on VAST (what we're benchmarking)
STORAGE_PATH="$VAST_BASE/kvbench_storage"

CONTAINER="/.autodirect/mswg2/E2E/Regression_logs/squash/nixlbench/gitlab-master.nvidia.com+eshukrun+warehouse+nixlbench+v0.1.0.dev.69b633aa.sqsh"
CONTAINER_PYTHON="/workspace/nixl/.venv/bin/python"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=2379

export NIXL_ETCD_ENDPOINTS="http://$MASTER_ADDR:$MASTER_PORT"
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/pre_posix_lg_${SLURM_JOB_ID}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=============================================="
echo "ISR1-PRE Large POSIX Storage Test (O_DIRECT)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Code: $LOCAL_KVBENCH (mtrsysgwork)"
echo "Storage: $STORAGE_PATH (VAST)"
echo "Backend: POSIX with O_DIRECT"
echo "Sizes: 64MB, 256MB, 1GB per rank"
echo "=============================================="

# Setup directories on all nodes
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

echo "Running POSIX Storage benchmark (CPU memory, O_DIRECT)..."
srun --mpi=pmix --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    bash -c "
        $BIND_SCRIPT $CONTAINER_PYTHON /workspace/nixl/benchmark/kvbench/main.py \
            sequential-ct-perftest $CONFIG_DIR/config_storage_large.yaml \
            --storage-backend POSIX \
            --storage-path $STORAGE_PATH \
            --storage-direct-io \
            --json-output-path $TEST_DIR/pre_posix_lg_$SLURM_JOB_ID.json
    "

srun -N1 -n1 -w $MASTER_ADDR bash -c "docker stop etcd_$SLURM_JOB_ID; docker rm etcd_$SLURM_JOB_ID" 2>/dev/null || true
echo "Done: $TEST_DIR/pre_posix_lg_${SLURM_JOB_ID}.json"

