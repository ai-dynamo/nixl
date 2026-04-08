---
title: KVBench
description: A KV cache benchmarking utility that generates and runs NIXLBench commands to profile LLM inference transfer performance.
---

KVBench is a benchmarking utility that generates and runs [NIXLBench](../nixlbench/index.md) commands for profiling KV cache transfer performance across LLM architectures. The `profile` command invokes `nixlbench` as a subprocess, executing the transfer benchmarks that KVBench plans based on model architecture and access patterns.

## Command Categories

### KVBench Commands

- **plan** -- Display the recommended NIXLBench configuration for a given model architecture and access pattern
- **profile** -- Run NIXLBench with the planned configuration and collect performance results
- **kvcache** -- Display KV cache layout information for a model architecture

### CTP Commands

- **ct-perftest** -- Run custom traffic performance tests using asymmetric transfer matrices
- **sequential-ct-perftest** -- Run sequential custom traffic performance tests for ordered matrix evaluation

## Supported Models

KVBench includes model architecture definitions for several LLM families: DeepSeek R1, Llama 3.1, and more. See [Commands and Examples](./commands.md) for details on defining custom model architectures.

## Next Steps

- **[Building KVBench](./build.md)** -- Docker and Python virtual environment installation
- **[Commands and Examples](./commands.md)** -- Full command reference and usage examples
