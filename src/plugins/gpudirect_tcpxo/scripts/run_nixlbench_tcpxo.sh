#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

# Source utility functions
source ./nixl_test_utils.sh

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Environment variables that MUST be set:"
  echo "  NIXLBENCH_IMAGE    Docker image for NIXL Bench"
  echo "  NIXLBENCH_TAG      Docker tag for NIXL Bench"
  echo "  RXDM_IMAGE         Docker image for RxDM"
  echo "  RXDM_TAG           Docker tag for RxDM"
  echo "  RXDM_FLAGS         Flags to pass to RxDM"
  echo "  NODE_RANK          Rank of this node (0 for initiator, >0 for target)"
  echo "  ETCD_ADDR          IP address of the ETCD server"
  echo ""
  echo "Options:"
  echo "  --initiator_seg_type <type> (default: VRAM)"
  echo "  --target_seg_type <type>    (default: VRAM)"
  echo "  --scheme <scheme>"
  echo "  --mode <mode>"
  echo "  --op_type <op>"
  echo "  --check_consistency"
  echo "  --total_buffer_size <size>"
  echo "  --start_block_size <size>"
  echo "  --max_block_size <size>"
  echo "  --start_batch_size <size>"
  echo "  --max_batch_size <size>"
  echo "  --num_iter <num>"
  echo "  --warmup_iter <num>"
  echo "  --large_blk_iter_ftr <num>"
  echo "  --num_threads <num>         (default: 4)"
  echo "  --num_initiator_dev <num>"
  echo "  --num_target_dev <num>"
  echo "  --enable_pt"
  echo "  --progress_threads <num>"
  echo "  --enable_vmm"
  echo "  --device_list <list>"
  echo "  --log_level <level>         (default: ERROR)"
  echo "  --test_timeout <secs>       (default: 1200)"
  echo "  --dump_rxdm_logs            Dump RxDM logs"
  echo "  -h, --help                Show this help message"
  exit 1
}

# Job parameters.
if [[ -z "${NIXLBENCH_IMAGE:-}" || -z "${NIXLBENCH_TAG:-}" ||
  -z "${RXDM_IMAGE:-}" || -z "${RXDM_TAG:-}" ||
  -z "${RXDM_FLAGS:-}" || -z "${NODE_RANK:-}" ||
  -z "${ETCD_ADDR:-}" ]]; then
  echo "Error: Missing required environment variables."
  usage
fi

# NIXL Bench Flags
INITIATOR_SEG_TYPE="VRAM"
TARGET_SEG_TYPE="VRAM"
NUM_THREADS="4"
SCHEME=""
MODE=""
OP_TYPE=""
CHECK_CONSISTENCY=""
TOTAL_BUFFER_SIZE=""
START_BLOCK_SIZE=""
MAX_BLOCK_SIZE=""
START_BATCH_SIZE=""
MAX_BATCH_SIZE=""
NUM_ITER=""
WARMUP_ITER=""
LARGE_BLK_ITER_FTR=""
NUM_INITIATOR_DEV=""
NUM_TARGET_DEV=""
ENABLE_PT=""
PROGRESS_THREADS=""
ENABLE_VMM=""
DEVICE_LIST=""
LOG_LEVEL="ERROR"
TEST_TIMEOUT="1200"
DUMP_RXDM_LOGS=false
RUN_MODE="default"

gather_nixlbench_flags() {
  while [[ $# -gt 0 ]]; do
    case $1 in
    --run_mode | --nixlbench-run-mode)
      RUN_MODE="$2"
      shift 2
      ;;
    --initiator_seg_type)
      INITIATOR_SEG_TYPE="$2"
      shift 2
      ;;
    --target_seg_type)
      TARGET_SEG_TYPE="$2"
      shift 2
      ;;
    --scheme)
      SCHEME="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --op_type)
      OP_TYPE="$2"
      shift 2
      ;;
    --check_consistency)
      CHECK_CONSISTENCY="1"
      shift 1
      ;;
    --total_buffer_size)
      TOTAL_BUFFER_SIZE="$2"
      shift 2
      ;;
    --start_block_size)
      START_BLOCK_SIZE="$2"
      shift 2
      ;;
    --max_block_size)
      MAX_BLOCK_SIZE="$2"
      shift 2
      ;;
    --start_batch_size)
      START_BATCH_SIZE="$2"
      shift 2
      ;;
    --max_batch_size)
      MAX_BATCH_SIZE="$2"
      shift 2
      ;;
    --num_iter)
      NUM_ITER="$2"
      shift 2
      ;;
    --warmup_iter)
      WARMUP_ITER="$2"
      shift 2
      ;;
    --large_blk_iter_ftr)
      LARGE_BLK_ITER_FTR="$2"
      shift 2
      ;;
    --num_threads)
      NUM_THREADS="$2"
      shift 2
      ;;
    --num_initiator_dev)
      NUM_INITIATOR_DEV="$2"
      shift 2
      ;;
    --num_target_dev)
      NUM_TARGET_DEV="$2"
      shift 2
      ;;
    --enable_pt)
      ENABLE_PT="1"
      shift 1
      ;;
    --progress_threads)
      PROGRESS_THREADS="$2"
      shift 2
      ;;
    --enable_vmm)
      ENABLE_VMM="1"
      shift 1
      ;;
    --device_list)
      DEVICE_LIST="$2"
      shift 2
      ;;
    --log_level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --test_timeout)
      TEST_TIMEOUT="$2"
      shift 2
      ;;
    --dump_rxdm_logs)
      DUMP_RXDM_LOGS=true
      shift 1
      ;;
    -h | --help) usage ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
    esac
  done

  NIXLBENCH_ARGS=(
    "--etcd_endpoints" "http://${ETCD_ADDR}:2379"
    "--backend" "TCPXO"
  )

  if [ -n "${INITIATOR_SEG_TYPE}" ]; then NIXLBENCH_ARGS+=("--initiator_seg_type" "${INITIATOR_SEG_TYPE}"); fi
  if [ -n "${TARGET_SEG_TYPE}" ]; then NIXLBENCH_ARGS+=("--target_seg_type" "${TARGET_SEG_TYPE}"); fi
  if [ -n "${SCHEME}" ]; then NIXLBENCH_ARGS+=("--scheme" "${SCHEME}"); fi
  if [ -n "${MODE}" ]; then NIXLBENCH_ARGS+=("--mode" "${MODE}"); fi
  if [ -n "${OP_TYPE}" ]; then NIXLBENCH_ARGS+=("--op_type" "${OP_TYPE}"); fi
  if [ -n "${CHECK_CONSISTENCY}" ]; then NIXLBENCH_ARGS+=("--check_consistency"); fi
  if [ -n "${TOTAL_BUFFER_SIZE}" ]; then NIXLBENCH_ARGS+=("--total_buffer_size" "${TOTAL_BUFFER_SIZE}"); fi
  if [ -n "${START_BLOCK_SIZE}" ]; then NIXLBENCH_ARGS+=("--start_block_size" "${START_BLOCK_SIZE}"); fi
  if [ -n "${MAX_BLOCK_SIZE}" ]; then NIXLBENCH_ARGS+=("--max_block_size" "${MAX_BLOCK_SIZE}"); fi
  if [ -n "${START_BATCH_SIZE}" ]; then NIXLBENCH_ARGS+=("--start_batch_size" "${START_BATCH_SIZE}"); fi
  if [ -n "${MAX_BATCH_SIZE}" ]; then NIXLBENCH_ARGS+=("--max_batch_size" "${MAX_BATCH_SIZE}"); fi
  if [ -n "${NUM_ITER}" ]; then NIXLBENCH_ARGS+=("--num_iter" "${NUM_ITER}"); fi
  if [ -n "${WARMUP_ITER}" ]; then NIXLBENCH_ARGS+=("--warmup_iter" "${WARMUP_ITER}"); fi
  if [ -n "${LARGE_BLK_ITER_FTR}" ]; then NIXLBENCH_ARGS+=("--large_blk_iter_ftr" "${LARGE_BLK_ITER_FTR}"); fi
  if [ -n "${NUM_THREADS}" ]; then NIXLBENCH_ARGS+=("--num_threads" "${NUM_THREADS}"); fi
  if [ -n "${NUM_INITIATOR_DEV}" ]; then NIXLBENCH_ARGS+=("--num_initiator_dev" "${NUM_INITIATOR_DEV}"); fi
  if [ -n "${NUM_TARGET_DEV}" ]; then NIXLBENCH_ARGS+=("--num_target_dev" "${NUM_TARGET_DEV}"); fi
  if [ -n "${ENABLE_PT}" ]; then NIXLBENCH_ARGS+=("--enable_pt"); fi
  if [ -n "${PROGRESS_THREADS}" ]; then NIXLBENCH_ARGS+=("--progress_threads" "${PROGRESS_THREADS}"); fi
  if [ -n "${ENABLE_VMM}" ]; then NIXLBENCH_ARGS+=("--enable_vmm"); fi
  if [ -n "${DEVICE_LIST}" ]; then NIXLBENCH_ARGS+=("--device_list" "${DEVICE_LIST}"); fi
}

gather_nixlbench_flags "$@"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NIXLBENCH_LOG_DIR_NAME="${NIXLBENCH_LOG_DIR_NAME:-nixlbench_${TIMESTAMP}}"
HOST_LOG_DIR="${HOME}/test_logs/${NIXLBENCH_LOG_DIR_NAME}"
mkdir -p "${HOST_LOG_DIR}" && chmod 777 "${HOST_LOG_DIR}"

INITIATOR_GPU=""
TARGET_GPU=""

# Preprocess RUN_MODE string for consistent regex matching
MODE_CLEAN="${RUN_MODE#cross_}"
MODE_CLEAN="${MODE_CLEAN#same_}"
MODE_CLEAN="${MODE_CLEAN//_to_/_}"

if [[ "${MODE_CLEAN}" == "default" || "${MODE_CLEAN}" == "pairwise_sg" ]]; then
  # Baseline 1x1 Single GPU
  :
elif [[ "${MODE_CLEAN}" =~ ^rail_([0-9]+)_([0-9]+)$ ]]; then
  INITIATOR_GPU="${BASH_REMATCH[1]}"
  TARGET_GPU="${BASH_REMATCH[2]}"
elif [[ "${MODE_CLEAN}" =~ ^pairwise_mg_([0-9]+)_([0-9]+)$ ]]; then
  NIXLBENCH_ARGS+=("--mode" "MG" "--scheme" "pairwise" "--num_initiator_dev" "${BASH_REMATCH[1]}" "--num_target_dev" "${BASH_REMATCH[2]}")
elif [[ "${MODE_CLEAN}" =~ ^pairwise_mg_([0-9]+)$ ]]; then
  NIXLBENCH_ARGS+=("--mode" "MG" "--scheme" "pairwise" "--num_initiator_dev" "${BASH_REMATCH[1]}" "--num_target_dev" "${BASH_REMATCH[1]}")
elif [[ "${MODE_CLEAN}" =~ ^fan_out_([0-9]+)_([0-9]+)$ || "${MODE_CLEAN}" =~ ^onetomany_([0-9]+)_([0-9]+)$ ]]; then
  NIXLBENCH_ARGS+=("--mode" "MG" "--scheme" "onetomany" "--num_initiator_dev" "${BASH_REMATCH[1]}" "--num_target_dev" "${BASH_REMATCH[2]}")
elif [[ "${MODE_CLEAN}" =~ ^fan_in_([0-9]+)_([0-9]+)$ || "${MODE_CLEAN}" =~ ^manytoone_([0-9]+)_([0-9]+)$ ]]; then
  NIXLBENCH_ARGS+=("--mode" "MG" "--scheme" "manytoone" "--num_initiator_dev" "${BASH_REMATCH[1]}" "--num_target_dev" "${BASH_REMATCH[2]}")
elif [[ "${MODE_CLEAN}" =~ ^tensor_parallel_([0-9]+)_([0-9]+)$ || "${MODE_CLEAN}" =~ ^tp_([0-9]+)_([0-9]+)$ ]]; then
  NIXLBENCH_ARGS+=("--mode" "MG" "--scheme" "tp" "--num_initiator_dev" "${BASH_REMATCH[1]}" "--num_target_dev" "${BASH_REMATCH[2]}")
else
  echo "Error: Unknown NIXLBench run mode format: ${RUN_MODE}"
  usage
fi

INIT_DEVS=1
TARGET_DEVS=1

for ((idx = 0; idx < ${#NIXLBENCH_ARGS[@]}; idx++)); do
  if [[ "${NIXLBENCH_ARGS[idx]}" == "--num_initiator_dev" ]]; then
    INIT_DEVS="${NIXLBENCH_ARGS[idx + 1]}"
  elif [[ "${NIXLBENCH_ARGS[idx]}" == "--num_target_dev" ]]; then
    TARGET_DEVS="${NIXLBENCH_ARGS[idx + 1]}"
  fi
done

TOTAL_BUFFER_SIZE="${TOTAL_BUFFER_SIZE:-8589934592}"
NIXLBENCH_ARGS+=("--total_buffer_size" "${TOTAL_BUFFER_SIZE}")

echo "=========================================================================="
echo "[Rank ${NODE_RANK:-0}] NIXLBench Parameter Sweep Info:"
echo "[Rank ${NODE_RANK:-0}]   Run Mode         : ${RUN_MODE}"
echo "[Rank ${NODE_RANK:-0}]   Initiator Devs   : ${INIT_DEVS}"
echo "[Rank ${NODE_RANK:-0}]   Target Devs      : ${TARGET_DEVS}"
echo "[Rank ${NODE_RANK:-0}]   Threads          : ${NUM_THREADS}"
echo "[Rank ${NODE_RANK:-0}]   Initiator GPU    : ${INITIATOR_GPU:-all}"
echo "[Rank ${NODE_RANK:-0}]   Target GPU       : ${TARGET_GPU:-all}"
echo "[Rank ${NODE_RANK:-0}]   Total Buffer Size: ${TOTAL_BUFFER_SIZE}"
echo "[Rank ${NODE_RANK:-0}]   Full Arguments   : ${NIXLBENCH_ARGS[*]}"
echo "=========================================================================="

cleanup() {
  local exit_code=$?
  echo "[Rank ${NODE_RANK}] Cleaning up NIXLBench containers and resources..."

  if [ "${DUMP_RXDM_LOGS}" == "true" ]; then
    echo "[Rank ${NODE_RANK}] Collecting RxDM logs..."
    sudo docker logs rxdm >"${HOST_LOG_DIR}/rxdm-${RUN_MODE}.log" 2>&1 || true
  fi

  if [ "${NODE_RANK}" -eq 0 ]; then
    sudo docker rm -f rxdm etcd-server nixlbench-fastrak-initiator 2>/dev/null || true
  else
    sudo docker rm -f rxdm nixlbench-fastrak-target 2>/dev/null || true
  fi

  # Make all files (especially perf files) readable by all so that SCP succeeds
  sudo chmod -R a+rX "${HOST_LOG_DIR}" 2>/dev/null || true
  sudo chown -R "${USER:-$LOGNAME}" "${HOST_LOG_DIR}" 2>/dev/null || true
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

cleanup_and_prepare_host

launch_rxdm "${RXDM_IMAGE}" "${RXDM_TAG}" "${RXDM_FLAGS}"
wait_for_container_running "rxdm"

if [ "${NODE_RANK}" -eq 0 ]; then
  # Launch EtcD
  echo "Starting etcd ..."
  sudo docker run -d --rm --name etcd-server \
    -p 2379:2379 -p 2380:2380 \
    quay.io/coreos/etcd:v3.5.18 \
    /usr/local/bin/etcd \
    --data-dir=/etcd-data \
    --listen-client-urls=http://0.0.0.0:2379 \
    --advertise-client-urls="http://${ETCD_ADDR}:2379" \
    --listen-peer-urls=http://0.0.0.0:2380 \
    --initial-advertise-peer-urls="http://${ETCD_ADDR}:2380" \
    --initial-cluster="default=http://${ETCD_ADDR}:2380"
  wait_for_container_running "etcd-server"

  # Launch NIXLBENCH Initiator
  PERF_VOL=""
  NIXLBENCH_CMD=(nixlbench "${NIXLBENCH_ARGS[@]}")
  echo "Starting nixlbench initiator..."

  CUDA_DEV_ENV=()
  if [ -n "${INITIATOR_GPU}" ]; then
    CUDA_DEV_ENV=("--env" "CUDA_VISIBLE_DEVICES=${INITIATOR_GPU}")
    echo "Configuring Initiator GPU ${INITIATOR_GPU} (CUDA_VISIBLE_DEVICES=${INITIATOR_GPU})"
  fi

  # Not running as root so that the user's credentials can be used to validate
  # against their package repository.
  docker pull "${NIXLBENCH_IMAGE}:${NIXLBENCH_TAG}"
  sudo docker run \
    --name nixlbench-fastrak-initiator \
    --network host \
    --detach \
    --privileged \
    --security-opt seccomp=unconfined \
    -u 0 \
    --cap-add=NET_ADMIN \
    --userns=host \
    --shm-size=1g \
    --volume "${HOST_LOG_DIR}:/workspace/logs" \
    --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
    ${DEVICE_FLAGS} \
    ${PERF_VOL} \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/dmabuf_import_helper:/dev/dmabuf_import_helper \
    --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
    --env NIXL_LOG_LEVEL="${LOG_LEVEL}" \
    "${CUDA_DEV_ENV[@]}" \
    "${NIXLBENCH_IMAGE}:${NIXLBENCH_TAG}" \
    "${NIXLBENCH_CMD[@]}"
  wait_for_container_running "nixlbench-fastrak-initiator"
else
  # Launch NIXLBENCH Target
  PERF_VOL=""
  NIXLBENCH_CMD=(nixlbench "${NIXLBENCH_ARGS[@]}")
  echo "Starting nixlbench target..."

  CUDA_DEV_ENV=()
  if [ -n "${TARGET_GPU}" ]; then
    CUDA_DEV_ENV=("--env" "CUDA_VISIBLE_DEVICES=${TARGET_GPU}")
    echo "Configuring Target GPU ${TARGET_GPU} (CUDA_VISIBLE_DEVICES=${TARGET_GPU})"
  fi

  docker run --pull=always \
    --name nixlbench-fastrak-target \
    --network host \
    --detach \
    --privileged \
    --security-opt seccomp=unconfined \
    -u 0 \
    --cap-add=NET_ADMIN \
    --userns=host \
    --shm-size=1g \
    --volume "${HOST_LOG_DIR}:/workspace/logs" \
    --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
    ${DEVICE_FLAGS} \
    ${PERF_VOL} \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/dmabuf_import_helper:/dev/dmabuf_import_helper \
    --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
    --env NIXL_LOG_LEVEL="${LOG_LEVEL}" \
    "${CUDA_DEV_ENV[@]}" \
    "${NIXLBENCH_IMAGE}:${NIXLBENCH_TAG}" \
    "${NIXLBENCH_CMD[@]}"
  wait_for_container_running "nixlbench-fastrak-target"
fi

# Wait for test to run and collect logs
ROLE=$([ "${NODE_RANK}" -eq 0 ] && echo "initiator" || echo "target")
CONTAINER_NAME="nixlbench-fastrak-${ROLE}"
echo "[Rank ${NODE_RANK}] Collecting NIXLBench logs for mode ${RUN_MODE} (timeout: ${TEST_TIMEOUT}s)..."
timeout "${TEST_TIMEOUT}s" sudo docker logs -f "${CONTAINER_NAME}" \
  > >(tee "${HOST_LOG_DIR}/nixlbench-${RUN_MODE}.report.log") \
  2> >(tee "${HOST_LOG_DIR}/nixlbench-${RUN_MODE}.system.log" >&2)

CONTAINER_EXIT_CODE=$(sudo docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}" 2>/dev/null || echo 1)
if [ "${CONTAINER_EXIT_CODE}" -ne 0 ]; then
  echo "[Rank ${NODE_RANK}] Error: ${CONTAINER_NAME} exited with code ${CONTAINER_EXIT_CODE}" >&2
  exit "${CONTAINER_EXIT_CODE}"
fi
