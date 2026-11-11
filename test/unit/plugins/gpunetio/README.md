# DOCA plugin unit test

## Native pool qualification

`native_qp_count=1` is the default. Native mode accepts1..32 active READ
channels per backend, selected once at view preparation with
`hint.customParam="native_channel=N"`. The GPU API/device-view ABI is unchanged.
Use matching channel views; channels are independently ordered, even with
strict registrations. Native pools still support one peer per engine.

Set `NIXL_GPUNETIO_TEST_QPS=4` or32 to exercise the owned pool in the hardware
GET test below. It checks distinct lane/QP identities, mixed-channel rejection,
generation preservation on re-preparation/MD reload, immediate GPU consumption,
and repeated SQ/CQ wrap on every channel. The default remains the one-QP test.

The following are disposable-process fault modes; failed resources stay alive
until process exit, not normal resource reclamation or recovery:

- `NIXL_GPUNETIO_TEST_PEER_QPS=N` differing from the reader count requires
  setup rejection during metadata/OOB exchange.
- `NIXL_GPUNETIO_TEST_RETENTION=1` poisons the last channel and checks idle
  view release plus failed-channel retention. Run under memory checking.
- Add `NIXL_GPUNETIO_TEST_CQ_ERROR=1` to use a real invalid-rkey READ instead
  of synthetic poisoning. It requires a CQ error, sticky refusal and continued
  healthy-channel progress before testing retention.

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

For the native GET test, add `NIXL_GPUNETIO_TEST_FIFO_CQ_ERROR=1` together
with `NIXL_GPUNETIO_TEST_RETENTION=1` and `NIXL_GPUNETIO_TEST_CQ_ERROR=1` to
inject a bad key between two valid READs and check sticky FIFO error handling.
Run this only in a disposable process; failed resources are retained until
process exit. Define `NIXL_GPUNETIO_PUBLIC_FIFO_ONLY=1` consistently when
compiling a consumer that uses only the public FIFO completion path.

With `build_tests=true`, a non-release build, and `gpunetio_device_api=enabled`,
`nixl_gpunetio_device_get_test` exercises GPU-initiated GET from registered
pinned Host or VRAM memory. It creates two agents on the selected GPUs and checks scalar and
CTA-batched reads, individual completions, result bytes, and guard bytes.
It also rejects a malformed metadata section and verifies that the same live
peer can reload valid metadata and execute the reads. The malformed-section
check emits an expected `NIXL_ERR_MISMATCH` log.
Run it manually on a configured GPUNETIO system with two free consecutive OOB
ports; the GPU index is relative to `CUDA_VISIBLE_DEVICES`:

```
./nixl_gpunetio_device_get_test <rdma-device> <oob-interface> <gid-index> <base-port> <owner-gpu> [DRAM|VRAM] [reader-gpu]
```

The source type defaults to `DRAM`, and the reader GPU defaults to the owner
GPU. The test requires the explicitly enabled DOCA 3.5 device API. Its
same-host result does not establish cross-node correctness or performance.

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
