# UCCL Backend Plugin

[UCCL](https://github.com/uccl-project/uccl) is an efficient communication library to perform GPU memory transfers, with a focus on flexibility (evolving ML workloads) and portability (heteregenous GPUs/NICs). UCCL provides a software transport stack which runs on the CPUs and are easily extensible to support different techniques like congestion control, multipathing, efficient loss recovery, etc.
UCCL supports collectives for training, P2P communication for PD disaggregation and gpu-driven communication for expert parallelism. This backend plugin adds support for UCCL P2P.

## Capabilities

The UCCL P2P backend supports internode communication over RDMA, TCP, TCP-X, and EFA, and intranode communication (GPU-to-GPU, GPU-to-CPU & vice-versa).

## Installation Guide

1. Install UCCL's P2P engine manually. You can refer to the [installation guide here](https://github.com/uccl-project/uccl/p2p).

```cpp
    git clone https://github.com/uccl-project/uccl.git
    cd uccl/p2p
    make -j
    sudo make install
```

UCCL is built with all transports included (**RDMA, TCP/TCP-X, and EFA**). To select a transport at runtime, set the environment variable while starting NIXL:

| Transport | Runtime ENV                                      |
|-----------|--------------------------------------------------|
| RDMA      | *(default)*                                      |
| TCPX/TCP  | `UCCL_P2P_TRANSPORT=tcp`                         |
| EFA       | `UCCL_P2P_TRANSPORT=efa`                         |

2. Build NIXL using regular method as in [README](https://github.com/ai-dynamo/nixl/blob/main/README.md).

## Usage Guide

Example Usage to create a NIXL agent with UCCL P2P engine:

```python
    config = nixl_agent_config(backends=["UCCL"])
    agent = nixl_agent("agent-name", config)
```
UCCL engine would auto discover the right NIC(s) to be used for the GPU based on the PCIe distance during memory registration based on the data locality.

### Environment Variables

Refer to [README](https://github.com/uccl-project/uccl/tree/main/collective/rdma#environment-variables-in-uccl) for the complete list of environment variables that can be set to customize UCCL.

### Usage References

**Running vLLM with UCCL backend** — To use UCCL for KV-cache transfer in a prefill-decode (PD) disaggregated setup, pass `"backends":["UCCL"]` in the KV transfer config. Here is a minimal example:

```bash
    vllm serve <MODEL> \
      --port <PORT> \
      --tensor-parallel-size <TP_SIZE> \
      --enforce-eager \
      --block-size <BLOCK_SIZE> \
      --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":
        {"backends":["UCCL"]}}'
```

The key part is `"backends": ["UCCL"]` — this tells NIXL to use the UCCL P2P engine instead of the default backend. UCCL will automatically select the best NIC for each GPU based on PCIe topology.

Refer to the [vLLM NIXL connector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py).


## Features & Roadmap

- ✅ Asynchronous posting of reads over multiple workers to mitigate latency increase upon fragmentation

- ✅ Intra-node communication support (IPC)

- ✅ Support for other transport (TCP, TCP-X, EFA.)

- ✅ Dynamic runtime transport selection

- 🚧 Add Progress Thread support

- 🚧 Add Telemetry support

