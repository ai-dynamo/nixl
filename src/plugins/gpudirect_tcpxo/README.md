# NIXL GPUDirect TCPXO Plugin

This plugin provides a high-performance RDMA backend for NIXL running on Google
Cloud
[A3-High and A3-Mega](https://docs.cloud.google.com/compute/docs/gpus#gpu-models)
virtual machines.

## Table Of Contents

## Overview

This plugin supports remote, VRAM<->VRAM transfers. These transfers are handled
by the
[IPUs](https://cloud.google.com/blog/products/compute/introducing-a3-supercomputers-with-nvidia-h100-gpus)
paired with a given GPU. Using the IPUs as a hardware-offload, we are able to
achieve zero-copy, non-blocking transfers between remote GPUs. This
hardware-offload is provided by two more Google libraries,
[RxDM and DXS](https://github.com/google/nccl-plugin-gpudirect-tcpxo). The
**Receive Datapath Manager** (RxDM) pins memory on the GPU and makes it
available to the IPU while the **Data Transfer Service** (DXS) actually
transfers the data between GPUs.

This plugin also supports rail-aligned (e.g. GPU 1 on Host A to GPU 1 on Host B)
or cross-rail traffic (e.g. GPU 3 on Host A to GPU 7 on Host B).

For a high-level view, see: <https://github.com/ai-dynamo/nixl/issues/2227>.

## Dependencies

This plugin relies on the following libraries:

-   [CUDA](https://developer.nvidia.com/cuda/toolkit)
-   [protobuf](https://protobuf.dev/)
-   [abseil](https://github.com/abseil/abseil-cpp)
-   [WebRTC](https://webrtc.googlesource.com/src/) (via
    [WebRTC Builds](https://github.com/vsimon/webrtcbuilds))
-   [DXS](https://github.com/google/nccl-plugin-gpudirect-tcpxo/tree/master/dxs)
-   [RxDM](https://github.com/google/nccl-plugin-gpudirect-tcpxo/tree/master/buffer_mgmt_daemon)

The DXS and RxDM dependencies additionally require:

-   [gRPC](https://grpc.io/)
-   [or-tools](https://github.com/google/or-tools)
-   [json](https://github.com/nlohmann/json)
-   [tensorflow serving](https://github.com/tensorflow/serving)
-   [ng-log](https://github.com/ng-log/ng-log/tree/master)
-   [libevent](https://github.com/libevent/libevent)
-   [gflags](https://github.com/gflags/gflags)
-   [googleapis](https://github.com/googleapis/googleapis)
-   [zlib](https://zlib.net/)
-   [boost.asio](https://www.boost.org/doc/libs/latest/doc/html/boost_asio.html)

## Building

### Prerequisites

Before building, ensure you have the following dependencies installed:

-   CUDA toolkit
-   protobuf
-   abseil and gRPC (for DXS and RxDM)
-   [WebRTC devel](https://github.com/google/nccl-plugin-gpudirect-tcpxo/tree/master/webrtc)

See the [Dockerfile](../../../contrib/Dockerfile) for details on how these are
installed and placed.

All the other dependencies are automatically retrieved during the `meson setup`
process.

### Meson

```sh
$ meson setup <builddir> <options>...
$ ninja -C <builddir> src/plugins/gpudirect_tcpxo/libplugin_TCPXO.so
```

Options for `meson`:

-   `-Ddisable_rxdm_dxs=[true|false]`
    -   Uses stubs for RxDM and DXS. Useful for running unit tests without real
        hardware.
-   `-Ddisable_gpudirect_tcpxo_cuda=[true|false]`
    -   Should be kept in sync with `-Ddisable_rxdm_dxs=[true|false]`.
    -   Disables CUDA integrations. Useful for doing development without real
        hardware.
-   `-Dskip_bazel_clean=[true|false]`
    -   Skips cleaning the bazel output directory when building DXS and RxDM.
        Keeping set as true offers subsequent `meson --reconfigure` calls.

## Environment Variables

The following [parameters](params.h) can be configured either via environment
variables or as NIXL engine initialization parameters:

### Quick Reference Table

| Environment Variable | Default Value | Valid Range / Type | Description |
| :--- | :--- | :--- | :--- |
| `FASTRAK_DATA_TRANSFER_TIMEOUT_MS` | `7200000` (2 hours) | `[0, 86400000]` ms | Timeout threshold for in-progress data transfers. `0` disables timeout. |
| `FASTRAK_DATA_TRANSFER_SLOWNESS_MS` | `300000` (5 minutes) | `[1, 86400000]` ms | Initial warning threshold for slow/pending data transfers (doubles each event). |
| `FASTRAK_PLUGIN_CONNECT_TIMEOUT_MS` | `300000` (5 minutes) | `[0, 86400000]` ms | Timeout threshold for the local node attempting to establish an outgoing connection to a peer GPU. |
| `FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS` | `900000` (15 minutes) | `[0, 86400000]` ms | Timeout threshold for the local node waiting for an incoming connection from a peer GPU. |
| `FASTRAK_DXS_LISTEN_TIMEOUT_MS` | `1000` (1 second) | `[10, 2000]` ms | Timeout threshold for DXS client listen socket readiness. |
| `FASTRAK_NUM_FLOWS` | `2` | `[1, 8]` | Number of concurrent network flows (network connections) per DXS connection. |
| `FASTRAK_NUM_CONTROL_CHANNEL_WORKERS` | `8` | `[1, 16]` | Number of worker threads processing control channel tasks and events. |
| `FASTRAK_HEARTBEAT_SEND_PERIOD_MS` | `15000` (15 seconds) | `[100, 900000]` ms | Interval between heartbeat messages sent to connected peers. |
| `FASTRAK_HEARTBEAT_TIMEOUT_MS` | `60000` (1 minute) | `[400, 7200000]` ms | Timeout threshold to disconnect an unresponsive peer if no heartbeat is received. |
| `FASTRAK_CTRL_DEV` | `"eth0"` | String | Network interface the control channel will bind to. This should be the non-GPU fabric NIC. |
| `FASTRAK_RXDM_INIT_TIMEOUT_SEC` | `30` (30 seconds) | `[0, 3600]` seconds | Timeout waiting for RxDM daemon readiness during init (`0` waits up to 1 hour). |
| `NUM_GPUS_PER_NODE` | `8` | `[1, 8]` | Number of GPUs per node to manage. |
| `FASTRAK_USE_LLCM` | `true` | Boolean (`true`/`false`) | Use a [fast communication device from the IPU](https://github.com/google/nccl-plugin-gpudirect-tcpxo/blob/04815d1bb89971577c7330f9873b70644b3a5934/dxs/client/guest_llcm/guest_llcm.h#L25) to exchange RxDM/DXS commands. |
| `FASTRAK_LLCM_DEVICE_DIRECTORY` | `/sys/bus/pci/devices` | String | Path where LLCM devices are exposed. |
| `FASTRAK_CLOSE_SEND_ON_DONE` | `false` | Boolean (`true`/`false`) | If the LLCM device is not used, should we send DXS a command to close its connections when the plugin is tearing down. |
| `FASTRAK_LOOPBACK_ONLY` | `false` | Boolean (`true`/`false`) | Restrict communication exclusively to the loopback (`lo`) interface. Useful only for tests (that aren't yet present). |
| `FASTRAK_IFNAME` | `""` (empty) | String | The explicit network interface name(s) of the NICs on the GPU fabric or the prefix for these NICs. |
| `FASTRAK_SOCKET_IFNAME` | `""` (empty) | String | Deprecated. Similar to the above, and only takes effect if the above is not supplied. Also respects `FASTRAK_SOCKET_FAMILY` below, and will fail if the devices don't support the specified IP version. |
| `FASTRAK_SOCKET_FAMILY` | `""` (empty) | String (`AF_INET`/`AF_INET6`) | Force socket address family selection to IPv4 (`AF_INET`) or IPv6 (`AF_INET6`). |
| `FASTRAK_COMM_ID` | `""` (empty) | String (`<IP>:<PORT>`) | Fallback for `FASTRAK_SOCKET_IFNAME` and `FASTRAK_SOCKET_FAMILY`. If neither is provided, we will use NICs on the same subnet as this endpoint. |

### Select Variable Descriptions

Most of these parameters will be left at their default. We'll cover a few that a
user is most likely to change:

-   `FASTRAK_LLCM_DEVICE_DIRECTORY`

    -   **What it modifies**: Directory path scanned to discover LLCM PCIe
        devices.
    -   **Impact**:
    -   Useful in containerized environments or setups where LLCM device nodes
        reside in non-standard filesystem paths.

-   `FASTRAK_IFNAME`

    -   **What it modifies**: Overrides automatic interface discovery by
        explicitly specifying the network interface name or pattern to use for
        the GPU Fabric NICs.
    -   **Impact**:
    -   Useful if the network interface names are different than what the
        auto-mapping detects (or if the auto-mapping isn't available). An
        example:

        ```sh
        FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
        ```

-   `FASTRAK_CTRL_DEV`

    -   **What it modifies**: Sets the network interface to bind the control
        channel to. The control channel will establish a listening socket on
        this device and connect from this device to other hosts. This should be
        an IP accessible by the other hosts on your VPC. It should not be
        exposed to the internet.
    -   **Impact**:
    -   If the normal network NIC has a different name than `eth0`, you can
        specify what NIC to use by changing this variable.

-   `FASTRAK_DATA_TRANSFER_TIMEOUT_MS`

    -   **What it modifies**: Sets the maximum time allowed for an individual
        data transfer operation (`NIXL_WRITE` / `NIXL_READ`) to complete before
        being marked as timed out (`NIXL_ERR_CANCELED`). If set to `0`, timeout
        enforcement is disabled.
    -   **Impact**:
    -   *Increasing / Disabling (`0`)*: Prevents premature failures on massive
        transfers or congested GPUs, but will cause stalled or stalled transfers
        to block indefinitely or delay failure recovery.
    -   *Decreasing*: Fails faster when transfers stall or when a remote worker
        drops, allowing callers to recover or failover sooner, but risks
        aborting legitimate slow transfers.

-   `FASTRAK_DXS_LISTEN_TIMEOUT_MS`

    -   **What it modifies**: Maximum time spent waiting for a DXS listen socket
        to report readiness during local endpoint setup.
    -   **Impact**:
    -   *Increasing*: Gives DXS more time to bind and listen if the system is
        under high thread contention or CPU load.
    -   *Decreasing*: Accelerates failure detection if the DXS daemon or driver
        is unresponsive during listen socket binding.

-   `FASTRAK_NUM_FLOWS`

    -   **What it modifies**: Number of concurrent network flows (channels)
        established per DXS peer connection.
    -   **Impact**:
    -   *Increasing (up to 8)*: Spreads transfer traffic across multiple network
        flows, improving bandwidth utilization at the cost of longer GPU
        connection establishment.
    -   *Decreasing (e.g., to 1)*: Speeds up GPU connection establishment at the
        cost reduced bandwith utilization.

-   `FASTRAK_NUM_CONTROL_CHANNEL_WORKERS`

    -   **What it modifies**: The number of background worker threads spawned to
        process control channel events, including connection establishment, peer
        handshakes, address exchanges, notifications, and transfer coordination
        messages.
    -   **Impact**:
    -   *Increasing*: Increases parallelization of processing control channel
        events. More events from different peers can be processed
        simultaneously, improving control channel responsiveness when the
        workload has a lot of peers at the cost of increased thread contention.
    -   *Decreasing*: Reduces thread contention and increases determinism.

-   `FASTRAK_RXDM_INIT_TIMEOUT_SEC`

    -   **What it modifies**: Maximum time (in seconds) to wait for the RxDM
        (Receive Datapath Manager) service to report healthy readiness during
        plugin initialization. Setting to `0` defaults to waiting up to 1 hour
        (3600 seconds).
    -   **Impact**:
    -   *Increasing / Setting to 0*: Allows the plugin to wait for RxDM daemons
        that take longer to initialize (e.g. during container startup or system
        reboot).
    -   *Decreasing*: Fails initialization quickly if RxDM is not running or
        misconfigured.

## Running RxDM

The Receive Datapath Manager (RxDM) is a required daemon/sidecar container that
pins GPU memory and makes it accessible to the IPU (SmartNIC) for zero-copy
GPUDirect TCPXO transfers. It must be running on each host prior to running your
workload using NIXL GPUDirect-TCPXO.

### GKE

For GKE, the
[NCCL instructions](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx#add-gpudirect-manifests)
cover the needed manifest changes to include RxDM.

### GCE

Inside the script
[`scripts/nixl_test_utils.sh`](scripts/nixl_test_utils.sh), the two
functions `launch_rxdm` and `cleanup_and_prepare_host` show the setup needed.

The following is a short explainer on what these functions do. First, some
one-time setup:

1.  **Load the `import-helper` kernel module**:

    ```sh
    sudo modprobe import-helper
    ```

    This creates the `/dev/dmabuf_import_helper` device node that RxDM uses for
    GPU memory pinning.

2.  **Permit TCP ingress traffic**:

    ```sh
    sudo /sbin/iptables -I INPUT -p tcp -m tcp -j ACCEPT
    ```

Then, launch the RxDM container with host networking, net admin capabilities,
and access to the NVIDIA GPUs and dmabuf device:

```sh
# Discover all GPU character devices on the host
DEVICE_FLAGS=$(find /dev -type c -regex "\/dev\/nvidia[0-9]*" -printf "--device %p:%p ")

# Launch the RxDM daemon container
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
  "${RXDM_IMAGE}:${RXDM_TAG}" \
  --num_hops 2
```

For verbose debugging logs, launch the container with `--stderrthreshold=0`.

## Class Overview

### Diagram

The following is a high-level overview of the classes and their organization.

```
+-----------------------------------------------------------------------------------------+
|                                    nixlTcpxoEngine                                      |
|                                                                                         |
|  +--------------------+   +-----------------------+   +------------------------------+  |
|  |   ControlChannel   |   |   DxsEndpointManager  |   |     Registered Memory        |  |
|  |  (EpollThread)     |   |                       |   |         Caches               |  |
|  |                    |   |  +-----------------+  |   +------------------------------+  |
|  |  - Peer Handshake  |   |  |   DxsEndpoint   |  |                                     |
|  |  - Address Exchange|   |  | (GPU-NIC Pair)  |  |   +------------------------------+  |
|  |  - Notifications   |   |  |                 |  |   |       Progress Thread        |  |
|  +---------+----------+   |  | - RxDM Client   |  |   |                              |  |
|            |              |  | - DXS Client    |  |   |  - Continuous DXS Test()     |  |
|            |              |  +-----------------+  |   |  - Transfer cleanup          |  |
|            v              +-----------------------+   +------------------------------+  |
|  +--------------------------+                                                           |
|  |   Worker Pool            |                                                           |
|  | (Thread-safe Task Queue) |                                                           |
|  +---------+----------------+                                                           |
|            |                                                                            |
|            v                                                                            |
|  +-----------------------------------------------------------------------------------+  |
|  |                              Remote Agents Map                                    |  |
|  |                                                                                   |  |
|  |  +-----------------------------------------------------------------------------+  |  |
|  |  |                               HostConnection                                |  |  |
|  |  |                                                                             |  |  |
|  |  |  - DxsOp Queue (Pending & In-Flight Operations)                             |  |  |
|  |  |  - Slowness & Timeout Enforcement                                           |  |  |
|  |  |  - Active Endpoint Connections:                                             |  |  |
|  |  |    * Send DxsConnection (DxsFlows 0..N)                                     |  |  |
|  |  |    * Recv DxsConnection (DxsFlows 0..N)                                     |  |  |
|  |  +-----------------------------------------------------------------------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
```

### Core Classes

-   [`nixlTcpxoEngine`](tcpxo_backend.h)

    Our plugin core. It coordinates DXS and RxDM initialization, manages the
    local memory registration cache, orchestrates remote agent connection
    metadata
    ([`nixlTcpxoConnection`](tcpxo_backend.h)),
    initiates transfers via `postXfer()`, and runs the background
    `ProgressThread()` to poll transfer completions.

-   [`ControlChannel`](control_channel.h)

    Manages out-of-band TCP socket communication between peers. It runs an epoll
    event loop on a dedicated thread (`EpollThread`) to handle asynchronous
    network I/O, process protobuf messages (identity handshakes, DXS address
    exchanges, workload notifications, transfer messages), and monitor peer
    liveness via heartbeats.

-   [`Worker`](control_channel.h)

    A thread pool of background workers (sized by
    `FASTRAK_NUM_CONTROL_CHANNEL_WORKERS`). Workers dequeue events dispatched by
    the `ControlChannel` and `HostConnection`—such as connection events, DXS
    address exchanges, incoming transfer requests, and disconnections—ensuring
    heavy setup tasks (e.g. establishing DXS sockets) do not block the control
    channel's epoll thread.

-   [`DxsEndpointManager`](dxs_endpoint.h)

    Manages the collection of [`DxsEndpoint`](dxs_endpoint.h) objects on the
    local host (one per GPU-NIC pair, up to `kMaxGpuDevices = 8`). Discovers
    available GPU fabirc NICs and queries RxDM for the GPU-to-NIC topology
    mapping.

-   [`DxsEndpoint`](dxs_endpoint.h)

    Encapsulates a single GPU and its paired IPU. Holds an RxDM client
    (`BufferManagerClient`) used to pin and register GPU memory and a
    `DxsClient` used to bind listening sockets and setup data connections.

-   [`HostConnection`](host_connection.h)

    Represents an active communication session with a specific remote peer. It
    tracks active endpoint connections [`EndpointConnection`](host_connection.h)
    and manages pending and issued transfer operations
    ([`DxsOp`](host_connection.h)).

-   [`DxsConnection`](dxs_endpoint.h) and [`DxsFlow`](dxs_endpoint.h)

    Represents an established DXS connection consisting of one or more
    concurrent network flows (`DxsFlow`, configured by `FASTRAK_NUM_FLOWS`).
    Each flow wraps a low-level send socket (`dxs::SendSocketInterface`) and
    receive socket (`dxs::LinearizedRecvSocketInterface`) between two GPUs.

-   [`nixlTcpxoLocalMemoryMetadata`](tcpxo_nixl_memory_metadata.h) and
    [`nixlTcpxoRemoteMemoryMetadata`](tcpxo_nixl_memory_metadata.h)

    Represent registered memory regions. Local metadata holds the dmabuf file
    descriptor and RxDM registration handle (`MemoryHandle`) generated when
    registering local GPU memory. Remote metadata encapsulates the serialized
    descriptor received from a peer, allowing DXS to target remote memory
    buffers.

## Common Issues

### RxDM Daemon Not Ready / Initialization Timeout

-   **Symptom**:

    ```
    RxDM not ready after 30 seconds (status: ...), retrying... (timeout 30 s)
    Timeout: RxDM not ready on <hostname> after 30 seconds
    ```
-   **Cause**: The RxDM daemon container (`rxdm` / `tcpxo-daemon`) is not
    running, crashed, or was delayed in starting up.
-   **Resolution**:

    1.  Check if the container is running: `docker ps --filter "name=rxdm"`.
    2.  Inspect container logs: `docker logs rxdm` (or `docker logs
        tcpxo-daemon`).
    3.  Ensure the container has full GPU and driver access (`--volume
        /var/lib/nvidia/lib64:/usr/local/nvidia/lib64`, `--device
        /dev/nvidia-uvm`, `--device /dev/nvidiactl`, `--privileged`,
        `--network=host`).
    4.  If the daemon starts slowly, increase `FASTRAK_RXDM_INIT_TIMEOUT_SEC`
        (or set to `0` to wait up to 1 hour).

### Missing Kernel Module or `/dev/dmabuf_import_helper`

-   **Symptom**: Memory registration fails during `registerMem()` or RxDM
    container startup with errors indicating `/dev/dmabuf_import_helper` is
    missing or cannot be opened.
-   **Cause**: The `import-helper` kernel module was not loaded on the host, or
    the device was not mounted into the container.
-   **Resolution**:
    1.  Load the module on the host: `sudo modprobe import-helper`.
    2.  Verify the device exists: `ls -l /dev/dmabuf_import_helper`.
    3.  Ensure `--device /dev/dmabuf_import_helper:/dev/dmabuf_import_helper` is
        passed to both the RxDM container and workload containers.

### Host Firewall Dropping TCP Traffic

-   **Symptom**: Remote hosts can't connect to the ETCD server on another host,
    the control channel connect times out or fails, or DXS socket connections
    stall indefinitely or fail with connection refused/timed out.
-   **Cause**: Host `iptables` or firewall policies are blocking incoming TCP
    traffic on dynamic ports used by the control channel or DXS endpoints.
-   **Resolution**:

    -   Add an iptables rule on the host to permit incoming TCP traffic:

    ```sh
    sudo /sbin/iptables -I INPUT -p tcp -m tcp -j ACCEPT
    ```
