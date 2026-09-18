# NIXLBench TCPXO Guide

## A3 NIXLBench Component Overview

[NIXLBench](https://github.com/ai-dynamo/nixl/tree/main/benchmark/nixlbench) is
a good way to ensure that your environment is setup correctly to use
GPUDirect-TCPXO. This guide provides instructions for running NIXLBench with the
aid of scripts we have developed.

These are the components we use to run NIXLBench:

*   **RxDM:** Standalone manager responsible for making GPU memory accessible to
    NIC for RDMA transfers.
*   **ETCD:** A distributed key-value coordinator service that facilitates
    inter-node synchronization, rank discovery, and barrier coordination.
*   **NIXLBench:** The actual NIXLBench binary itself which gets launched on the
    VMs.

## Execution Overview

1.  Launch the RxDM container on both nodes.
2.  Launch the ETCD server on one node.
3.  Launch the NIXL containers on both nodes.

We've created [`run_nixlbench_tcpxo.sh`](../scripts/run_nixlbench_tcpxo.sh)
(which uses [`nixl_test_utils.sh`](../scripts/nixl_test_utils.sh)) to simplify
launching these components. `run_nixlbench_tcpxo.sh` will take care of all three
steps on each VM.

## Prerequisites

Before executing the benchmarking script, ensure your workstation and target
cluster meet the following requirements:

1.  **IAM Permissions**

    *   Compute Admin or Instance Admin privileges in the target GCP project are
        required to interact with and deploy containers to the virtual machines.
    *   Access to pull RxDM images from
        [gce-ai-infra](us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo).
    *   Access to pull and push NIXLBench images to your GCP project’s artifact
        directory or some other docker artifact repository.

2.  **Gcloud SDK Authentication**

    Ensure your local workstation Google Cloud SDK
    [installed](https://docs.cloud.google.com/sdk/docs/install-sdk) and is
    authenticated

    ```sh
    $ gcloud auth login
    $ gcloud config set project <YOUR PROJECT ID>
    ```

3.  **Repository Checkout**

    For access to the scripts, clone the repository to your workstation

    ```sh
    $ git clone https://github.com/ai-dynamo/nixl
    $ cd nixl/src/plugins/gpudirect_tcpxo/scripts
    ```

4.  **Target VM State**

    Verify that the target virtual machines are active, in the case of A3-High
    VMs have FasTrak enabled (A3-Mega VMs are enabled by default), and have
    matching NVIDIA kernel drivers properly loaded.

## Quick Start (Pairwise Execution)

### 1. Copy Scripts

Copy `run_nixlbench_tcpxo.sh` and `nixl_test_utils.sh` to your target VMs

```sh
$ gcloud compute scp run_nixlbench_tcpxo.sh nixl_test_utils.sh "<VM1>:~/" --project="<PROJECT>" --zone="<ZONE>"
$ gcloud compute scp run_nixlbench_tcpxo.sh nixl_test_utils.sh "<VM2>:~/" --project="<PROJECT>" --zone="<ZONE>"
```

### 2. Launch NIXLBench

SSH into both VMs and launch RxDM.

For each VM:

```sh
$ gcloud compute ssh <VM> --project="<PROJECT>" --zone="<ZONE>"

# Do some environment setup
$ export NODE_RANK=<0|1>  # Choose one VM to be "Rank 0". This VM will run the ETCD server
$ export PRIMARY_ADDR=<IP>  # Set this to the internal IP of Rank 0's host NIC
$ export NIXLBENCH_LOG_DIR_NAME=<path/to/logs>  # Choose a place the logs will be written to
$ export NIXLBENCH_IMAGE=<my.docker.registry/repository/image>
$ export IMAGE_TAG=<NIXLBENCH_IMAGE_TAG>
$ export RXDM_IMAGE=us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/tcpgpudmarxd-dev
$ export RXDM_TAG=latest
$ export RXDM_FLAGS="--num_hops 2"
$ NODE_RANK=$NODE_RANK ETCD_ADDR=$PRIMARY_ADDR NIXLBENCH_LOG_DIR_NAME=$NIXLBENCH_LOG_DIR_NAME NIXLBENCH_IMAGE=$NIXLBENCH_IMAGE NIXLBENCH_TAG=$IMAGE_TAG RXDM_IMAGE=$RXDM_IMAGE RXDM_TAG=$RXDM_TAG RXDM_FLAGS=$RXDM_FLAGS MAX_BLOCK_SIZE=4096 bash ./run_nixlbench_tcpxo.sh --run_mode default
```

There are different run modes and tweaks to the NIXLBench parameters possible
with `run_nixlbench_tcpxo.sh`. See
[`nixlbench_modes.txt`](../scripts/nixlbench_modes.txt) for examples of
different run modes. Run `./run_nixlbench_tcpxo -h` to see all the options you
can tweak.

## Under the Hood & Advanced Orchestration

For troubleshooting, customizing NIXLBench command-line arguments for manual
optimization, or container validation, it is helpful to understand the
underlying mechanics configured automatically by the `run_nixlbench_tcpxo.sh`
script. This section goes into some detail about what the script is doing.

### RxDM and NIXLBench Docker Container Configuration

To access raw GPU hardware and kernel interfaces directly without the wrapper
script, these docker containers must be spawned with custom system limits,
volume mounts, and device nodes.

*   **System Resource Flags:** The RxDM docker container needs
    `--cap-add=NET_ADMIN` permissions in order to interact directly with the
    IPU. Both docker containers need extra shared memory allocated with
    `--shm-size=1g` (or `256g` on Ubuntu instances).
*   **Device & Driver Mappings:** Host libraries and device nodes must be
    exposed explicitly:
    *   NVIDIA Host libraries: `--volume /var/lib/nvidia:/usr/local/nvidia`
    *   NVIDIA GPU nodes: `/dev/nvidia0` through `/dev/nvidia7`,
        `/dev/nvidia-uvm`, and `/dev/nvidiactl`
    *   RxDM Memory Mapping Helper: `/dev/dmabuf_import_helper`

Refer directly to [run_nixlbench_tcpxo.sh](../scripts/run_nixlbench_tcpxo.sh)
and [nixl_test_utils.sh](../scripts/nixl_test_utils.sh) for more details and
up-to-date configuration.

### ETCD Node Orchestration & Barrier Synchronisation

Our scripts launch the ETCD server on the main node. To spin up the ETCD server
manually:

```shell
# Within NIXLBench container instance on the  main node (Node 0)
pkill etcd
nohup etcd --listen-client-urls=http://0.0.0.0:2379 \
  --advertise-client-urls=http://0.0.0.0:2379 &
```

### Direct NIXLBench Invocation (with TCPXO Backend)

Our script the runs the benchmark inside the containers with these default
parameters:

```shell
./nixlbench \
  --etcd-endpoints "http://${SERVER_HOST_NAME}:2379" \
  --backend TCPXO \
  --initiator_seg_type VRAM \
  --target_seg_type VRAM \
  --num_initiator_dev 1 \
  --num_target_dev 1 \
  --mode SG \
  --op_type WRITE \
  --start_block_size=4096 \
  --max_block_size=67108864 \
  --start_batch_size=1 \
  --max_batch_size=1 \
  --num_iter=1024 \
  --warmup_iter=128 \
  --num_threads=4
```

## Common Issues & Warnings

*   **Thread Distribution Normalization:** The engine will occasionally scale or
    realign `num_iter` or `warmup_iter` parameters internally during launch to
    ensure evenly balanced loops across the requested thread pool. This is
    expected behavior.
*   **NVIDIA Driver Mismatches:** Ensure that the host-level NVIDIA kernel
    driver is correctly configured. Failures to resolve matching library
    dependencies will generate warnings during initialization, which leads to
    running in unsupported modes.
*   **Barrier Synchronization Timeout / Hanging:** If the test execution hangs
    during the initial bootstrap, verify that all participating container ranks
    have completed registration with the central ETCD server.
