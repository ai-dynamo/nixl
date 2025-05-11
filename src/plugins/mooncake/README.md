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

## Known Issues
1. The `Notif[ication]` and `ProgTh[read]` features are not supported.
2. The current version of Mooncake Transfer Engine manages metadata exchange by itself, which is different from NIXL.
3. The sum of the number of release requests for each handle allocated by `prepXfer()` should be less than `kMaxRequestCount(1024)`.

> [!IMPORTANT]
> We are working for refactoring Mooncake Transfer Engine to make it more adaptful and useful.
