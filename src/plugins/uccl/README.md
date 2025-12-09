## UCCL Backend Plugin [Preview]

[UCCL](https://github.com/uccl-project/uccl) is an efficient communication library to perform GPU memory transfers, with a focus on flexibility (evolving ML workloads) and portability (heteregenous GPUs/NICs). UCCL provides a software transport stack which runs on the CPUs and are easily extensible to support different techniques like congestion control, multipathing, efficient loss recovery, etc.
UCCL supports collectives for training, P2P communication for PD disaggregation and gpu-driven communication for expert parallelism. This backend plugin adds support for UCCL P2P.

## Capabilities

Currently, the UCCL P2P backend supports internode communication over RDMA. Intranode communication will be added soon.

## Installation Guide

1. Install UCCL's P2P engine manually. You can refer to the [installation guide here](https://github.com/uccl-project/uccl/p2p).

    ```cpp
    git clone https://github.com/uccl-project/uccl.git
    cd uccl/p2p
    make -j
    sudo make install
    ```

2. Build NIXL using regular method as in [README](https://github.com/ai-dynamo/nixl/blob/main/README.md).

## Usage Guide

Example Usage to create a NIXL agent with UCCL P2P engine:

    ```python
    config = nixl_agent_config(backends=["UCCL"])
    agent = nixl_agent("agent-name", config)
    ```
UCCL engine would auto discover the right NIC to be used for the GPU based on the PCIe distance during memory registration based on the data locality.

### Environment Variables

Refer to [README](https://github.com/uccl-project/uccl/tree/main/collective/rdma#environment-variables-in-uccl) for the complete list of environment variables that can be set to customize UCCL.

**Important**: For `NIXL_READ` operations set `UCCL_RCMODE=1`. By default, UCCL uses RDMA UC (Unreliable Connection). However, `READ` operations need to operate on RDMA RC (Reliable Connection). `WRITE` operations can work on both RDMA UC and RC.

### Usage References

1) [NIXL Benchmark](https://github.com/uccl-project/uccl/blob/main/p2p/benchmarks/benchmark_nixl.py) in UCCL P2P: Refer to  this [README](https://github.com/uccl-project/uccl/tree/main/p2p) on how to run the script.

2) [NIXL connector](https://github.com/vllm-project/vllm/commit/e731733d30d0aed3252dc60427927768bfc0ca73) in vLLM. vLLM's NIXL connector uses `NIXL_READ` operations, hence set env `UCCL_RCMODE` to 1.

### Road Map

- [ ] Add Intra-node communication support

- [ ] Add Progress Thread support

- [ ] Add asynchronous posting of reads over multiple workers to mitigate latency increase upon fragmentation

- [ ] Add support for other transport (TCP, TCP-X, etc.)