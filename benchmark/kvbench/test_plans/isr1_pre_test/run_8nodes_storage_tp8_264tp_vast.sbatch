#!/bin/bash
#SBATCH --job-name=8n_st_tp8_264
#SBATCH --partition=ISR1-PRE
#SBATCH --reservation=nonRA
#SBATCH --nodes=8
#SBATCH --nodelist=hgx-isr1-pre-[02,05,07,10-12,14-15]
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/auto/mtrsysgwork/mtennenhaus/kvbench_results/8n_st_tp8_264_pre_%j.out

export TMPDIR=/tmp
export ENROOT_TEMP_DIR=/tmp
STORAGE_PATH="/mnt/vast/mtennenhaus/kvbench_storage_8n_st_tp8_264_pre_${SLURM_JOB_ID}"

# etcd configuration
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=2379
export NIXL_ETCD_ENDPOINTS="http://$MASTER_ADDR:$MASTER_PORT"
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/8n_st_tp8_264_pre_${SLURM_JOB_ID}"

# UCX settings for RDMA (ISR1-PRE)
export UCX_TLS=rc,cuda_copy
export UCX_IB_GID_INDEX=3

CONTAINER="/.autodirect/mswg2/E2E/Regression_logs/squash/nixlbench/gitlab-master.nvidia.com+eshukrun+warehouse+nixlbench+v0.1.0.dev.69b633aa.sqsh"
CONTAINER_PYTHON="/workspace/nixl/.venv/bin/python"

NIXL_REPO="/mtrsysgwork/mtennenhaus/nixl"
LOCAL_KVBENCH="$NIXL_REPO/benchmark/kvbench"
CONFIG_DIR="$NIXL_REPO/benchmark/kvbench/test/decode_centric_12nodes"
CONFIG_FILE="$CONFIG_DIR/storage_70b_8nodes_tp8_264tp_read_only/metadata.yaml"
TEST_DIR="/auto/mtrsysgwork/mtennenhaus/kvbench_results"
OUTPUT="$TEST_DIR/8n_st_tp8_264_pre_$SLURM_JOB_ID.json"
BIND_SCRIPT="$LOCAL_KVBENCH/test_plans/isr1_pre_test/bind_8gpus.sh"

echo "=============================================="
echo "8-Node Storage-Only Test - TP8 264 TPs (ISR1-PRE + VAST)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR"
echo "Config: $CONFIG_FILE"
echo "Storage: $STORAGE_PATH (VAST + GDS)"
echo "=============================================="

srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p $STORAGE_PATH"
rm -rf $STORAGE_PATH/* 2>/dev/null || true

# Start etcd on master node
echo "Starting ETCD on $MASTER_ADDR..."
srun -N1 -n1 -w $MASTER_ADDR \
    docker run -d --name etcd_8n_st_tp8_264_pre_$SLURM_JOB_ID --network host \
        quay.io/coreos/etcd:v3.5.0 \
        etcd --listen-client-urls http://0.0.0.0:$MASTER_PORT \
             --advertise-client-urls http://$MASTER_ADDR:$MASTER_PORT \
             --data-dir /tmp/etcd_8n_st_tp8_264_pre_$SLURM_JOB_ID \
             --log-level warn

sleep 5

MOUNTS="$LOCAL_KVBENCH:/workspace/nixl/benchmark/kvbench,$CONFIG_DIR:$CONFIG_DIR,$TEST_DIR:$TEST_DIR,$STORAGE_PATH:$STORAGE_PATH,$BIND_SCRIPT:$BIND_SCRIPT"

# Pre-pull container
srun -n $SLURM_NNODES --ntasks-per-node=1 --container-image="$CONTAINER" --container-mounts="$MOUNTS" true

echo "Running 8-node storage-only benchmark (TP8 264 TPs, VAST GDS)..."
srun --mpi=pmix --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    bash -c "
        $BIND_SCRIPT $CONTAINER_PYTHON /workspace/nixl/benchmark/kvbench/main.py \
            sequential-ct-perftest $CONFIG_FILE \
                --storage-backend GDS \
                --storage-path $STORAGE_PATH \
                --warmup-iters 1 \
                --isolation-iters 5 \
                --json-output-path $OUTPUT
    "

# Cleanup etcd
srun -N1 -n1 -w $MASTER_ADDR bash -c "docker stop etcd_8n_st_tp8_264_pre_$SLURM_JOB_ID; docker rm etcd_8n_st_tp8_264_pre_$SLURM_JOB_ID" 2>/dev/null || true

echo "Done: $OUTPUT"
