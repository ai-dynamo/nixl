#!/bin/bash

set -e
set -x

if [ -n "$TEST_RUN" ]
then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y python3.12 python3.12-venv python3-pip
  # Install infiniband libraries
  apt-get install -y libibverbs1 rdma-core ibverbs-utils ibverbs-providers libibumad3 libnuma1 librdmacm1t64
  python3.12 -m venv /root/venv
  source /root/venv/bin/activate
  pip install --upgrade pip
  pip install /wheel/nixl*cp312*.whl
  export UCX_PROTO_INFO=y
  python /nixl/examples/python/nixl_api_example.py
  exit 0
fi

IMG="nvidia/cuda:12.8.0-base-ubuntu24.04"

NIXL_DIR=$(dirname $(dirname $(readlink -f "$0")))
WHEEL_DIR="$1"

if [ -z "$WHEEL_DIR" ]
then
  echo "Usage: $0 <path to wheel directory>"
  exit 1
fi

docker run --rm --privileged --device=/dev/infiniband --net=host --ipc=host --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$NIXL_DIR:/nixl:ro" \
  -v "$WHEEL_DIR:/wheel:ro" \
  -w /nixl \
  -it $IMG \
  bash -c "TEST_RUN=1 /nixl/contrib/wheel-test.sh"
