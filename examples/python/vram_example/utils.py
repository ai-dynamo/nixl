# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging


def parse_args():
    """
    Parse commandline argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--available-memory",
        type=int,
        default=4,
        help="Estimated available memory size for kv cache (GB)",
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dims", type=int, default=128)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def get_logger(name):
    """
    Create default logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def calc_memory_blocks(args):
    """
    Calculate tensor_size and num_blocks to allocate for test
    """

    # Emurate vllm kv cache as dense model.
    # In default, the parameter is from Mistral-Small-3.1:
    # 2(kv), 8 heads, 128 dimensions, 16 tokens per block, bf16 (2bytes),
    # 40 layers
    #
    # And, the calculation of the block size in bytes is:
    # 2 * 8 * 128 * 16 * 2 = 65536 (64KB)
    #
    # Then, if GPU has 4GB available memory for KV Cache, the number of
    # blocks per layer is:
    # ((4 * 1024 * 1024 * 1024) / (64 * 1024)) // 40 = 1638 (blocks)

    # Let's caclculate block size and alignment
    gpu_mem = args.available_memory * 1024 * 1024 * 1024
    block_size = args.block_size
    heads = args.heads
    dims = args.dims
    layers = args.layers
    num_blocks = gpu_mem // layers // (2 * heads * dims * block_size * 2)
    shape_len = 2 * block_size * heads * dims
    tensor_size = shape_len * num_blocks
    return tensor_size, shape_len, num_blocks
