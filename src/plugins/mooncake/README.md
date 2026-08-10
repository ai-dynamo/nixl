# Mooncake Backend Plugin [Preview]

[Mooncake](https://github.com/kvcache-ai/Mooncake) is a KVCache-centric disaggregated architecture for LLM serving.
The core of Mooncake is the Transfer Engine, which provides a unified interface for batched data transfer across various storage devices and network links. Supporting multiple protocols including TCP, RDMA, CXL/shared-memory, and NVMe over Fabric (NVMe-of), Transfer Engine is designed to enable fast and reliable data transfer for AI workloads. Compared to Gloo (used by Distributed PyTorch) and traditional TCP, Transfer Engine achieves significantly lower I/O latency, making it a superior solution for efficient data transmission.

Mooncake transfer engine is a high-performance, zero-copy data transfer library. To achieve better performance in NIXL, we have designed an new backend based on Mooncake Transfer Engine.

## Engine modes

The backend can drive either Mooncake engine, selected by the `mooncake_mode` backend parameter (or the `NIXL_MOONCAKE_MODE` environment variable, which takes precedence):

| Mode | Engine | Notes |
|---|---|---|
| `classic` (default) | Transfer Engine (`transfer_engine_c.h`) | Unchanged behavior, works with every Mooncake release |
| `tent` | TENT, the next-generation engine, through its native C API (`tent/transfer_engine.h`) | Requires Mooncake built with `-DUSE_TENT=ON` |

`tent` mode additionally provides:

- a `releaseReqH()` that follows the BackendGuide cancellation protocol (best-effort `tent_cancel_task()` on every in-flight task, non-blocking poll, and a refusal to release until every task reached a terminal state) instead of the classic path, where the engine offers no cancellation primitive;
- `O(1)` completion polling through the aggregated batch status, which also drives engine-internal progress and failover;
- memory registration that carries the transport type and, for `VRAM_SEG`, the `cuda:<devId>` location.

If NIXL is built against a Mooncake without TENT, the plugin still builds and offers `classic` only; requesting `tent` then fails engine creation with a clear error.

## Usage Guide
1. Build and install Mooncake. You can refer to the [installation guide here](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries). Add `-DUSE_TENT=ON` if you want the `tent` mode.

    ```bash
    git clone https://github.com/kvcache-ai/Mooncake.git
    cd Mooncake
    bash dependencies.sh
    mkdir build
    cd build
    cmake .. -DBUILD_SHARED_LIBS=ON -DUSE_TENT=ON
    make -j
    sudo make install
    ```

    > [!IMPORTANT]
    > You must build and install the shared library (`-DBUILD_SHARED_LIBS=ON`) before building NIXL with the Mooncake backend.

2. Build NIXL, ensuring that the option `disable_mooncake_backend` is set as `false`.

3. To test the Mooncake backend, you can run the unit test in `test/unit/plugins/mooncake/mooncake_backend_test`. Run it twice, with and without `NIXL_MOONCAKE_MODE=tent`, to cover both engines.

4. To use the Notify feature, you need to download the latest main branch of Mooncake.

## Configuration
- `mooncake_mode` (backend parameter) / `NIXL_MOONCAKE_MODE` (environment): `classic` (default) or `tent`.
- `tent_config_path` (backend parameter): optional TENT configuration file, read in `tent` mode before engine creation. The settings the plugin requires (`metadata_type=p2p`, `local_segment_name`) override the file.
- `NIXL_MOONCAKE_IP_ADDR` (environment): the IP address advertised to remote agents. The plugin otherwise picks the first non-loopback interface that is UP and RUNNING, which may not be routable from the peers on multi-NIC hosts.

## Known Issues
1. The `ProgTh[read]` features are not supported.
2. The current version of Mooncake Transfer Engine manages metadata exchange by itself, which is different from NIXL.
3. The sum of the number of requests posted on one handle allocated by `prepXfer()` between two completions should be less than `kMaxRequestCount(1024)`.
4. In `classic` mode, releasing a handle whose transfer is still in flight cannot cancel it and leaks the batch; the engine offers no cancellation primitive. Use `tent` mode where that matters.
5. In `tent` mode, notification payloads are limited to 4095 bytes (and agent names to 255 bytes) by the TENT C API; oversized messages are rejected with `NIXL_ERR_INVALID_PARAM`.
6. In p2p mode both the RPC port and the data-plane ports are chosen at runtime, so firewalled deployments must open the corresponding ranges (see the Mooncake deployment guide, e.g. `MC_MIN_RPC_PORT`/`MC_MAX_RPC_PORT`).
7. VRAM registration in `tent` mode passes `cuda:<devId>` as the buffer location; this path is not yet validated on GPU hardware.
