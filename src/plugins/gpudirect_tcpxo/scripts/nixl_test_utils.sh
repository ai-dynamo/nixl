#!/bin/bash

is_container_running_docker() {
  local container_name="$1"

  if [ -z "$container_name" ]; then
    echo "Error: Container name not provided." >&2
    echo "Usage: is_container_running_docker <container_name>" >&2
    return 2
  fi

  if ! command -v docker &>/dev/null; then
    echo "Error: docker command not found." >&2
    return 3
  fi

  # docker ps -q returns only container IDs.
  # --filter "name=^${container_name}$" matches the exact container name.
  # --filter "status=running" ensures it's running.
  local running_id
  running_id=$(docker ps -q --filter "name=^${container_name}$" --filter "status=running")

  if [ -n "$running_id" ]; then
    # echo "Container '$container_name' is running with ID $running_id."
    return 0 # Success: Container is running
  else
    # echo "Container '$container_name' is not running or does not exist."
    return 1 # Failure: Container is not running
  fi
}

wait_for_container_running() {
  local container_name="$1"
  local timeout_secs="${2:-30}"
  local elapsed=0

  echo "Waiting for ${container_name} to start running..."
  while ! is_container_running_docker "${container_name}"; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "${elapsed}" -ge "${timeout_secs}" ]; then
      echo "Error: Timed out waiting for ${container_name} to start." >&2
      exit 1
    fi
  done
  echo "${container_name} is running."
}

cleanup_and_prepare_host() {
  echo "Cleaning up and preparing host environment..."

  # Update iptables.
  sudo /sbin/iptables -I INPUT -p tcp -m tcp -j ACCEPT

  # Import dmabuf helper.
  sudo modprobe import-helper

  # Cleanup other containers including the RxDM.
  while IFS= read -r line; do
    CONTAINER_NAME=$(echo "$line" | awk '{print $2}')
    sudo docker stop "${CONTAINER_NAME}"
    sudo docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    echo "Removing ${CONTAINER_NAME}..."
  done < <(sudo docker container ls -a --format "{{.ID}} {{.Names}}")

  # Wait for container removal to complete.
  while [[ $(sudo docker container ls -a | tail -n +2) ]]; do sleep 1; done
}

launch_rxdm() {
  local rxdm_image="$1"
  local rxdm_tag="$2"
  local rxdm_flags="$3"

  # Launch the RxDM.
  DEVICE_FLAGS=$(find /dev -type c -regex "\/dev\/nvidia[0-9]*" \
    -printf "--device %p:%p ")
  eval "RXDM_FLAGS_ARR=(${rxdm_flags})"

  echo "Pulling rxdm..."
  docker pull "${rxdm_image}:${rxdm_tag}"

  echo "Starting rxdm..."
  # Disabling linter for unquoted expansions.
  # shellcheck disable=SC2086
  sudo docker run --rm \
    --name rxdm \
    --detach \
    --privileged \
    --cap-add=NET_ADMIN \
    --network=host \
    --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
    ${DEVICE_FLAGS} \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/dmabuf_import_helper:/dev/dmabuf_import_helper \
    --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
    "${rxdm_image}:${rxdm_tag}" \
    "${RXDM_FLAGS_ARR[@]}"

  echo "Started rxdm"
}
