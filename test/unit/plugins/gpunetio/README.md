# DOCA plugin unit test

## Native pool qualification

`native_qp_count=1` is the default. Native mode accepts 1..32 active data
channels per backend. Standard GET, PUT and atomicAdd select `channel_id` per
call using one prepared view pair. The backend readSession extension retains
`hint.customParam="native_channel=N"` and requires matching selected views.
Device-view ABI 2 requires rebuilding CUDA consumers with the matching backend.
Channels are independently ordered; native pools support one peer per engine.

Set `NIXL_GPUNETIO_TEST_QPS=4` or32 to exercise the owned pool in the hardware
device API test below. It checks distinct lane/QP identities, batch mixed-view
rejection, standard per-call channel selection, generation preservation on
re-preparation, copied-status safety, fixed-mode submissions and immediate GPU
consumption. Standard GET/PUT/atomic/status/flush checks include nullable status,
defer, reverse polling, queue wrap, same-channel THREAD/WARP/BLOCK producers,
GPU-produced PUT data, channel isolation and Graph replay. The default remains
the one-QP test. GRID is unsupported; RDMA getPtr returns nullptr.

The following are disposable-process fault modes; failed resources stay alive
until process exit, not normal resource reclamation or recovery:

- `NIXL_GPUNETIO_TEST_PEER_QPS=N` differing from the reader count requires
  both counts to be at least two and checks `NIXL_ERR_INVALID_PARAM` from the
  metadata load that validates the pool header. Equal counts are invalid for this
  mismatch mode; omit the variable for the normal matching-count path.
- `NIXL_GPUNETIO_TEST_RETENTION=1` requires `NIXL_GPUNETIO_TEST_QPS=2` or
  greater, poisons the last channel and checks failed-channel retention. Run
  under memory checking.
- Add `NIXL_GPUNETIO_TEST_CQ_ERROR=1` to use a real invalid-rkey READ instead
  of synthetic poisoning. It requires a CQ error, sticky refusal and continued
  healthy-channel progress before testing retention.
- Add `NIXL_GPUNETIO_TEST_STANDARD_CQ_ERROR=1` with the retention and CQ-error
  flags to put a bad key in the middle of a standard deferred request sequence.
  Polling the last request must detect the error while the checked prefix stays
  successful. Use `NIXL_GPUNETIO_TEST_FIFO_CQ_ERROR=1` instead for the separate
  readSession FIFO regression; run the two modes in separate processes.

`nixl_gpunetio_native_pool_options_test` is a CPU-only parser test with no
GPU/network dependency. It covers all supported sizes/channels and invalid
syntax/bounds/binary blobs.

Unit test application for the DOCA GPUNetIO backend plugin. It enables a GPU to GPU communications between initiator and target using the GPUDirect Async Kernel-Initiated (GDAKI) technology through the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) library.

Initiator and target communicate over the network using the DOCA GPUNetIO plugin, so both processes should run on a system with a GPU. Initiator and target exchange a sequence of metadata and notifications for the setup then initiator posts twice the same transfer request with 32 buffers, with a value for the first transfer and updated value for the second transfer.

It's possible to execute in two modes:
- Stream attached: when creating a transfer request, the application can specify a CUDA stream where the transfer request (executed by a CUDA kernel implemented within the plugin) must be posted. Useful in situations where data is prepared in a GPU CUDA kernel and right after, asynchronously, it can be sent by the GPU without requiring any CPU synchronization.
- Stream pool: at transfer request creation time, no CUDA stream is provided by the app. The DOCA GPUNetIO plugin can still be used as it uses an internal pool of CUDA streams used in a round-robin fashion.

Please note in this unit test, data used for transfers is allocated in GPU memory. In real-world applications this is not mandatory: the DOCA GPUNetIO plugin can send/receive data from the CUDA kernel if it resides in CPU memory.

## System requirements

Please look at the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) programming guide to properly configure youy system for the correct execution of this application.

## Execute

The following instructions assume DOCA libraries and NIXL libraries are set in LD_LIBRARY_PATH environment variable.

The native GET test also exercises FIFO pipelining and partial retirement over
90 rounds per channel; it skips this part when MCST consistency is required.
An ordered graph test holds a session across kernel nodes, verifies a second
acquisition is rejected while the lane is held, then retires and reuses the lane
across replays without assuming concurrent block scheduling.
With two or more QPs, two separately captured Graphs also publish independent
READs on distinct lanes before either retires; destination bytes are checked.

For the native GET test, set `NIXL_GPUNETIO_TEST_QPS=2` or greater and add
`NIXL_GPUNETIO_TEST_FIFO_CQ_ERROR=1` together with
`NIXL_GPUNETIO_TEST_RETENTION=1` and `NIXL_GPUNETIO_TEST_CQ_ERROR=1` to
inject a bad key between two valid READs and check sticky FIFO error handling.
Run this only in a disposable process; failed resources are retained until
process exit. The public FIFO completion path is the default. Define
`NIXL_GPUNETIO_PUBLIC_FIFO_ONLY=0` consistently across CUDA consumers only
when testing the unsupported private parallel CQ helper.

With `build_tests=true`, a non-release build, and `gpunetio_device_api=enabled`,
`nixl_gpunetio_device_get_test` exercises standard GPU-initiated operations and
the CTA batch extension against registered pinned Host or VRAM memory. It
creates two agents on the selected GPUs and checks payloads, guards, completion,
ordering and Graph replay. The DRAM case also samples the receiver counter and
checks the preceding PUT payload immediately after observing the increment.
Run it manually on a configured GPUNETIO system with two free consecutive OOB
ports; the GPU index is relative to `CUDA_VISIBLE_DEVICES`:

```
./nixl_gpunetio_device_get_test <rdma-device> <oob-interface> <gid-index> <base-port> <owner-gpu> [DRAM|VRAM] [reader-gpu]
```

The source type defaults to `DRAM`, and the reader GPU defaults to the owner
GPU. The OOB interface must have an active IPv4 address inside the execution
environment; an RDMA sysfs netdev name alone does not establish that address.
The test requires the explicitly enabled DOCA 3.5+ device API. It prints a skip
when the hardware requires MCST consistency, which this implementation reports
as unsupported. Its same-host result covers component correctness.

Set `NIXL_GPUNETIO_TEST_LARGE=1` for one additional GET and PUT of
`DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE + 64` bytes, with a full GPU byte check
and 64-byte guards on each end. It allocates two additional VRAM buffers
(about 2 GiB total on DOCA 3.5), regardless of the small test's source type.
This mode requires matching QP counts and cannot be combined with the peer-count
mismatch mode. No large host mirror is allocated.

Target should start first. A command line example:

```
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/gdrcopy/src:/opt/mellanox/doca NIXL_PLUGIN_DIR=/path/to/nixl/lib/x86_64-linux-gnu/plugins CUDA_MODULE_LOADING=EAGER ./nixl_gpunetio_stream_test target <initiator IP address> <connection port> <stream mode ('attached' or 'pool')>
```

To start the initiator:

```
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/gdrcopy/src:/opt/mellanox/doca NIXL_PLUGIN_DIR=/path/to/nixl/lib/x86_64-linux-gnu/plugins CUDA_MODULE_LOADING=EAGER ./nixl_gpunetio_stream_test initiator <target IP address> <connection port> <stream mode ('attached' or 'pool')>
```

As specified in [DOCA GPUNetIO programming guide](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html#src-3670647880_id-.DOCAGPUNetIOv3.0.0-RunningwithoutRootPrivileges) this application uses GDAKI technology thus can be executed without sudo/root privileges if the NVIDIA driver has been correctly configured with the right option. If not, the application must be executed with sudo/root privileges.

To test this application is highly recommented to have target and initiator running on two different machines with a network connection between the two.

## Output

To ensure everything works correctly as expected, tagert should print out 32 lines like the following ones for the first transfer:

```
>>>>>>> CUDA target kernel, buffer 0, all bytes received! val=187
>>>>>>> CUDA target kernel, buffer 1, all bytes received! val=187
...
>>>>>>> CUDA target kernel, buffer 30, all bytes received! val=187
>>>>>>> CUDA target kernel, buffer 31, all bytes received! val=187
```

and another set of 32 lines for the second tranfer:

```
>>>>>>> CUDA target kernel, buffer 0, all bytes received! val=188
>>>>>>> CUDA target kernel, buffer 1, all bytes received! val=188
...
>>>>>>> CUDA target kernel, buffer 30, all bytes received! val=188
>>>>>>> CUDA target kernel, buffer 31, all bytes received! val=188
```
