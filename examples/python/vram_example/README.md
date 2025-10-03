<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL VRAM Transfer Example

This is an example of VRAM transfer using NIXL, inspired by the [vLLM](https://github.com/vllm-project/vllm) 0.10.0 KV cache transfer algorithm. It demonstrates a basic server-client model for transferring VRAM memory. The server process allocates tensor memory filled with ones, and the client copies these tensors into locally allocated tensors on the client side.

## Memory Alignment

As described above, this example follows vLLM's KV cache memory management. The transfer unit is controlled by a specific memory block size within each tensor.

The memory layout is designed as follows:

- Let **N** be the number of layers. The process creates **N** tensors.
- Each tensor maintains contiguous memory alignment but is logically divided into blocks of `block_size` tokens.
- The memory size of each block is calculated using the following formula:

`KV * Heads * Dimensions * Tokens per block * Precision`

For example, considerating [Mistral-Small-3.1](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/tree/main) for default vLLM config, with:

- N (Attention layers) = 40
- KV = 2
- Heads = 8
- Dimensions = 128
- Tokens per block = 16
- Precision = bf16 (2 bytes)

The block size in bytes is:

`2 * 8 * 128 * 16 * 2 = 65536 (64KB)`

The memory alignment looks like this:

```
tensor 0  [(64KB)|(64KB)|(64KB)| ... |(64KB)]
tensor 1  [(64KB)|(64KB)|(64KB)| ... |(64KB)]
...
tensor 39 [(64KB)|(64KB)|(64KB)| ... |(64KB)]
```

If 4GB of VRAM is available for Mistral-Small-3.1, the number of blocks per tensor is calculated as:

`((4 * 1024 * 1024 * 1024) / (64 * 1024)) // 40 = 1638 (blocks)`


## Transfer Model

This example uses the memory alignment described above. NIXL has a capability to manage memory blocks using indices. Following vLLM's approach, the process first reserves blocks for KV cache. These reserved blocks maintain consistent indices across all layer tensors. Therefore, this example transfers data from the head block up to the number of blocks required for the given input tokens.

## Usage
Start the server process to wait for incoming requests:

```
python server.py
```

Then, launch the client process:

```
python client.py
```

Note that, the server process will be running with while-loop so use `Ctrl-C` or kill the process directly to terminate server process at the end.

If you want to use a different GPU for each process, pass `CUDA_VISIBLE_DEVICES` environment variable to pin the GPU index.

And the variables of the tranfer model can be configured by the argument. Please use `--help` to confirm available arguments.


> [!NOTE]
> The server runs in a `while` loop. Use `Ctrl-C` or terminate the process manually to stop the server.

> [!TIP]
> To use different GPUs for each process, set the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPU index.

You can configure the transfer model parameters using command-line arguments. Use `--help` to view all available options.

> [!NOTE]
> The parameters must be identical between the server and client processes to ensure successful transfer.

