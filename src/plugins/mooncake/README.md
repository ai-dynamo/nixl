# Mooncake Backend Plugin [Preview]

[Mooncake](https://github.com/kvcache-ai/Mooncake) is a KVCache-centric disaggregated architecture for LLM serving.
The core of Mooncake is the Transfer Engine, which provides a unified interface for batched data transfer across various storage devices and network links. Supporting multiple protocols including TCP, RDMA, CXL/shared-memory, and NVMe over Fabric (NVMe-of), Transfer Engine is designed to enable fast and reliable data transfer for AI workloads. Compared to Gloo (used by Distributed PyTorch) and traditional TCP, Transfer Engine achieves significantly lower I/O latency, making it a superior solution for efficient data transmission.

Mooncake transfer engine is a high-performance, zero-copy data transfer library. To achieve better performance in NIXL, we have designed an new backend based on Mooncake Transfer Engine.

## Usage Guide
1. Build the install Mooncake manually. You can refer to the [installation guide here](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries).

    ```cpp
    git clone https://github.com/kvcache-ai/Mooncake.git
    cd Mooncake
    bash dependencies.sh
    mkdir build
    cd build
    cmake .. -DBUILD_SHARED_LIBS=ON
    make -j
    sudo make install
    ```

    > [!IMPORTANT]
    > You must build and install the shared library (`-DBUILD_SHARED_LIBS=ON`) before building NIXL with the Mooncake backend.

2. Build NIXL, ensuring that the option `disable_mooncake_backend` is set as `false`.

3. To test the Mooncake backend, you can run the unit test in `test/unit/plugins/mooncake/mooncake_backend_test`.

4. To use the Notify feature, you need to download the latest main branch of Mooncake.

## Known Issues
1. The `ProgTh[read]` features are not supported.
2. The current version of Mooncake Transfer Engine manages metadata exchange by itself, which is different from NIXL.
3. The sum of the number of release requests for each handle allocated by `prepXfer()` should be less than `kMaxRequestCount(1024)`.
4. CUDA-graph-stable checkpoint pause/resume is a prototype path gated by strict preconditions. It requires CUDA VMM allocations for graph-visible VRAM, stable graph-visible indirection buffers above the backend, no active transfer batches, and fresh remote connection metadata after resume.

## CUDA-Graph-Stable Checkpointing

Preserving CUDA virtual addresses with reserved/mapped CUDA VMM memory is mandatory for graph-visible buffers, but it is not the whole checkpoint/resume contract. Mooncake also owns opaque transport state, including QPs, memory registrations, rkeys, IPC handles, remote segments, segment cache entries, and active batch IDs. A graph-stable integration must keep CUDA graph-visible addresses stable while destroying/recreating transport/application/communicator state and refreshing opaque handles behind stable graph-visible indirection buffers.

The NIXL Mooncake backend exposes prototype `checkpointPauseGraphStable()` and `checkpointResumeGraphStable()` hooks for the local quiesce/pre-pause and local resume phases:

- `checkpointPauseGraphStable()` rejects the pause if any transfer batch is active or if a registered VRAM buffer is not recognized as CUDA VMM memory. On success it marks the backend paused, rejects new transfers, drops cached remote segment IDs, and retains local registered virtual addresses.
- `checkpointResumeGraphStable()` requires a previous successful pause, revalidates registered VRAM addresses, clears the paused state, and intentionally leaves remote agents disconnected. The caller must reload fresh remote connection information for the restored node/IP before posting transfers.
- `prepXfer()` and `postXfer()` fail while paused. `postXfer()` also fails until fresh remote connection information has been loaded after resume, preventing stale segment IDs from being reused.

This makes graph-stable replay feasible under the VMM/stable-indirection model without recapturing CUDA graphs, but production integrations still need additional plumbing above this backend: coordinated quiesce across all workers, fresh bootstrap/address metadata distribution on the destination node, stable device-side indirection buffers for rkeys/QP/IPC data, and end-to-end validation that the lower Mooncake Transfer Engine refreshes all opaque handles before graph replay.

> [!IMPORTANT]
> We are working for refactoring Mooncake Transfer Engine to make it more adaptful and useful.
