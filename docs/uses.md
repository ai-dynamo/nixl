# NIXL Usage Examples

## Overview

The purpose of this document is to provide references to NIXL usage examples in different contexts.

## RL Post-Training

### Weight transfers

In reinforcement learning (RL) post-training, weight transfer refers to copying the trainer’s updated model parameters to the inference workers used in the rollout process.
The model weights are typically sharded on both the training workers and inference workers and an efficient weight transfer implementation should be built around the following principles:
- Parallelize the transfers across all the training workers to maximize available NIC BW.
- Avoid gathering the full model onto any training worker, focus on transferring shards that are available locally.
- Take into account what shards each inference worker owns to avoid transferring unused model weights to these workers.

There are multiple designs to achieve these goals with different trade-offs:
1. `Pull Mode`: training workers prepare the model weights and broadcast readiness to inference workers, the inference workers initiate pull transfers and install the weights.
    - Maximum elasticity and fault tolerance:
       - The trainer doesn't need to be aware of the inference setup.
       - Multiple inference engines can operate in parallel with no knowledge of each other.
    - There is a variant `Async Pull Mode` where the inference workers execute the update completely asynchronously from the training workers and even relative to each other.
    - There is a variant `Pull Mode with Storage` where the training workers publish the model weights to some storage workers, from which they are pulled onto the inference workers.
2. `Push Mode`: training workers prepare the model weights to match the inference engine requirements and push them to the inference workers.
    - Simplicity:
      - Outside of pausing/resuming the inference engines, this is fully one-sided.
    - Reduced elasticity and fault tolerance:
      - The training workers need to be aware of the complete inference setup.
3. `Collective Mode`: training workers prepare the model weights, all workers collaboratively execute the transfers, inference workers install the weights.
    - Most opportunity for transfer optimization:
      - Full collaboration between all the workers can expose opportunities from awareness of the full system topology to maximize system BW utilization.
    - Limited elasticity and fault tolerance:
      - Need to maintain a collective group including both the training and inference workers.

NIXL's unique combination of performance, flexibility, and easy-to-use APIs makes it a great backend for implementing any of these strategies, allowing the user to select a design based on their needs.

#### Framework-agnostic libraries/backends

1. [ModelExpress RL](https://github.com/ai-dynamo/modelexpress/tree/main/docs/rl)
    - Implements an `Async Pull Mode` design.
    - Integration example in [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl).
2. [vLLM](https://github.com/vllm-project/vllm) Weight Transfer API [sharded_rdt](https://docs.vllm.ai/en/stable/training/weight_transfer/sharded_rdt/) backend
    - Implements a `Pull Mode` design.
    - Discussed in [Large-Scale Sharded Weight Transfer with Ray Direct Transport (RDT) in vLLM](https://vllm.ai/blog/2026-08-22-rdt-weight-transfer).
    - Integration example in [SkyRL](https://github.com/novasky-ai/skyrl), including usage with Megatron trainer.
3. [TorchStore](https://github.com/meta-pytorch/torchstore)
    - Implements a `Pull Mode with/without Storage` design.
    - Integration example in [TitanRL](https://github.com/pytorch/torchtitan/blob/main/torchtitan/rl/README.md).

#### RL frameworks

1. [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
    - The [MX backend](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/transports/weights/modelexpress.py) integrates ModelExpress RL.
    - The [NIXL backend](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/transports/weights/nixl) implements a `Pull Mode` design with details discussed in [GLM-5.2 RL weight transfer in 4 seconds using NIXL and ModelExpress](https://www.primeintellect.ai/blog/nixl-modelexpress-weight-transfer).
2. [Miles](https://github.com/radixark/miles)
    - The [P2P protocol](https://github.com/radixark/miles/blob/main/docs/advanced/p2p-weight-transfer.md) implements a `Push Mode` design with NIXL as an optional backend.
3. [verl](https://github.com/verl-project/verl)
    - The [WIP sharded_nixl backend](https://github.com/verl-project/verl/pull/8395) implements a `Push Mode` design.
4. [SkyRL](https://github.com/novasky-ai/skyrl)
    - Integrates the vLLM `sharded_rdt` Weight Transfer backend.
5. [TitanRL](https://github.com/pytorch/torchtitan/blob/main/torchtitan/rl/README.md)
    - Integrates TorchStore.
