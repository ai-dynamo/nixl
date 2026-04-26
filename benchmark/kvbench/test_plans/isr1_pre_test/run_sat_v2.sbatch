#!/bin/bash
# Parameterized saturation test sbatch — called by run_all_sat_v2.sh
# Env vars set by caller: SAT_NAME, SAT_NODES, SAT_NTASKS_PER_NODE, SAT_CONFIG, SAT_NODELIST
#SBATCH --partition=ISR1-PRE
#SBATCH --reservation=nonRA
#SBATCH --gpus-per-node=0
#SBATCH --time=00:30:00
#SBATCH --exclusive

export TMPDIR=/tmp
export ENROOT_TEMP_DIR=/tmp
STORAGE_PATH="/mnt/vast/mtennenhaus/kvbench_sat_v2_${SAT_NAME}_${SLURM_JOB_ID}"

MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=2379
export NIXL_ETCD_ENDPOINTS="http://$MASTER_ADDR:$MASTER_PORT"
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/sat_v2_${SAT_NAME}_${SLURM_JOB_ID}"
export UCX_TLS=rc,cuda_copy
export UCX_IB_GID_INDEX=3

CONTAINER="/.autodirect/mswg2/E2E/Regression_logs/squash/nixlbench/gitlab-master.nvidia.com+eshukrun+warehouse+nixlbench+v0.1.0.dev.69b633aa.sqsh"
CONTAINER_PYTHON="/workspace/nixl/.venv/bin/python"

NIXL_REPO="/auto/mtrsysgwork/mtennenhaus/nixl"
LOCAL_KVBENCH="$NIXL_REPO/benchmark/kvbench"
CONFIG_FILE="$SAT_CONFIG"
TEST_DIR="/auto/mtrsysgwork/mtennenhaus/kvbench_results/spcx_v2"
OUTPUT="$TEST_DIR/${SAT_NAME}_$SLURM_JOB_ID.json"
BIND_SCRIPT="$LOCAL_KVBENCH/test_plans/isr1_pre_test/bind_8gpus.sh"
CONFIG_PARENT=$(dirname $(dirname $CONFIG_FILE))

echo "=============================================="
echo "VAST Saturation v2: $SAT_NAME"
echo "  block_size=1M, posix_api=uring, O_DIRECT"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR"
echo "Config: $CONFIG_FILE"
echo "Storage: $STORAGE_PATH"
echo "=============================================="

srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p $STORAGE_PATH"

echo "Starting ETCD on $MASTER_ADDR..."
srun -N1 -n1 -w $MASTER_ADDR \
    docker run -d --name etcd_sat_v2_${SAT_NAME}_$SLURM_JOB_ID --network host \
        quay.io/coreos/etcd:v3.5.0 \
        etcd --listen-client-urls http://0.0.0.0:$MASTER_PORT \
             --advertise-client-urls http://$MASTER_ADDR:$MASTER_PORT \
             --data-dir /tmp/etcd_sat_v2_${SAT_NAME}_$SLURM_JOB_ID \
             --log-level warn

sleep 5

MOUNTS="$LOCAL_KVBENCH:/workspace/nixl/benchmark/kvbench,$CONFIG_PARENT:$CONFIG_PARENT,$TEST_DIR:$TEST_DIR,$STORAGE_PATH:$STORAGE_PATH,$BIND_SCRIPT:$BIND_SCRIPT"

srun -n $SLURM_NNODES --ntasks-per-node=1 --container-image="$CONTAINER" --container-mounts="$MOUNTS" true

echo "Running optimized saturation test..."
srun --mpi=pmix --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    bash -c "
        $CONTAINER_PYTHON /workspace/nixl/benchmark/kvbench/main.py \
            sequential-ct-perftest $CONFIG_FILE \
                --storage-backend POSIX \
                --storage-path $STORAGE_PATH \
                --storage-direct-io \
                --storage-block-size 1M \
                --storage-posix-api uring \
                --warmup-iters 5 \
                --isolation-iters 5 \
                --json-output-path $OUTPUT
    "

srun -N1 -n1 -w $MASTER_ADDR bash -c "docker stop etcd_sat_v2_${SAT_NAME}_$SLURM_JOB_ID; docker rm etcd_sat_v2_${SAT_NAME}_$SLURM_JOB_ID" 2>/dev/null || true
echo "Done: $OUTPUT"
