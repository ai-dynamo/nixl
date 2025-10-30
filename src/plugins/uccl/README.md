## UCCL Backend Plugin [Preview]

[UCCL](https://github.com/uccl-project/uccl) is an efficient communication library to perform GPU memory transfers, with a focus on flexibility (evolving ML workloads) and portability (heteregenous GPUs). UCCL provides a software transport stack which runs on the CPUs and are easily extensible to support different techniques like congestion control, multipathing, efficient loss recovery, etc.
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

Example Usage to create a NIXL agent with uccl engine:

    ```python
    config = nixl_agent_config(backends=["UCCL"])
    agent = nixl_agent("agent-name", config)
    ```
UCCL engine would auto discover the right NIC to be used for the GPU based on the PCIe distance during memory registration based on the data locality.

### Environment Variables

Refer to [README](https://github.com/uccl-project/uccl/tree/main/collective/rdma#environment-variables-in-uccl) for the complete list of environment variables that can be set to customize UCCL.

**Important**: For `NIXL_READ` operations set `UCCL_RCMODE=1`. By default, UCCL uses RDMA UC (Unreliable Connection). However, `READ` operations need to operate on RDMA RC (Reliable Connection).

### Usage References

1) [NIXL Benchmark](https://github.com/uccl-project/uccl/blob/main/p2p/benchmarks/benchmark_nixl.py) in UCCL: Refer to  this [README](https://github.com/uccl-project/uccl/tree/main/p2p) on how to run the script.

2) [NIXL connector](https://github.com/praveingk/vllm/commit/fa67cd7edff076fee4914cc316a9833c2311a65d) in vLLM. vLLM's NIXL connector uses `NIXL_READ` operations, hence set env `UCCL_RCMODE` to 1.

### Road Map

- [ ] Add Intra-node communication support

- [ ] Add Progress Thread support

- [ ] Add asynchronous posting of reads over multiple workers to mitigate latency increase upon fragmentation

- [ ] Add support for other transport (TCP, TCP-X, etc.)