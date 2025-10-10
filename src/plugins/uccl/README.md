## UCCL Backend Plugin [Preview]

[UCCL](https://github.com/uccl-project/uccl) is an efficient communication library to perform GPU memory transfers, with a focus on flexibility (evolving ML workloads) and portability (heteregenous GPUs).
UCCL supports collectives, p2p communication and gpu-driven communication for expert parallelism.

## Capabilities

Currently, the UCCL backend supports internode communication over RDMA. Intranode communication will be added soon.

## Installation Guide

1. Install UCCL's p2p engine manually. You can refer to the [installation guide here](https://https://github.com/uccl-project/uccl).

    ```cpp
    git clone https://github.com/uccl-project/uccl.git
    cd uccl/p2p
    make -j
    sudo make install
    ```

2. Build NIXL using regular method as in [README](https://github.com/ai-dynamo/nixl/blob/main/README.md) ensuring `disable_uccl_backend` is set to `false`.

## Usage Guide

### Additional Parameters
1. `device_idx` : Specifies which GPU the UCCL engine will be affined to.
Example Usage to create a NIXL agent with uccl engine on GPU 0: 
    ```python
    config = nixl_agent_config(device_idx=0, backends=["Uccl"])
    agent = nixl_agent("agent-name", config)
    ```
UCCL engine would auto discover the right NIC to be used for the GPU based on the PCIe distance

### Environment Variables
1. `NCCL_IB_GID_INDEX` : GID Index of the device to be used. Usually, its auto-detected. 
2. `UCCL_SOCKET_IFNAME` : The ethernet interface to be used for control socket communication.
3. `UCCL_IB_HCA` : HCAs to be used for UCCL connection.
4. `UCCL_RCMODE` : Set to either 0 or 1. To enable RDMA RC (Reliable Connection), set to 1. For `NIXL_READ` operations, set `UCCL_RCMODE` to 1.

### Usage References

1) [NIXL Benchmark](https://github.com/uccl-project/uccl/blob/main/p2p/benchmarks/benchmark_nixl.py) in UCCL: Refer to  this [README](https://github.com/uccl-project/uccl/tree/main/p2p) on how to run the script.

2) [NIXL connector](https://github.com/praveingk/vllm/commit/fa67cd7edff076fee4914cc316a9833c2311a65d) in vLLM.

### Road Map

- [ ] Add Intra-node communication support

- [ ] Add asynchronous posting of reads over multiple workers to mitigate latency increase upon fragmentation

- [ ] Add support for other transport (TCP, TCP-X, etc.)