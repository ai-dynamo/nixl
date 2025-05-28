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


def add_common_args(subparser: argparse.ArgumentParser):
    subparser.add_argument("--model", type=str, help="Model name (e.g., 'llama3.1-8b')")
    subparser.add_argument(
        "--model_config", type=str, help="Path to a single model config YAML file"
    )
    subparser.add_argument(
        "--model_configs",
        type=str,
        help="Path to multiple model config YAML files (supports glob patterns like 'configs/*.yaml')",
    )


def add_cli_args(subparser: argparse.ArgumentParser):
    subparser.add_argument("--pp", type=int, help="Pipeline parallelism size")
    subparser.add_argument("--tp", type=int, help="Tensor parallelism size")
    subparser.add_argument("--isl", type=int, help="Input sequence length")
    subparser.add_argument("--osl", type=int, help="Output sequence length")
    subparser.add_argument("--num_requests", type=int, help="Number of requests")
    subparser.add_argument("--page_size", type=int, help="Page size")
    subparser.add_argument("--access_pattern", type=str, help="Access pattern")


def add_plan_args(subparser: argparse.ArgumentParser):
    subparser.add_argument(
        "--format",
        default="text",
        type=str,
        help="Output of the nixl command [text, json, csv] (default: text)",
    )


def add_nixl_bench_args(subparser: argparse.ArgumentParser):
    subparser.add_argument(
        "--source",
        default="file",
        type=str,
        help="Source of the nixl descriptors [file, memory, gpu] (default: file)",
    )
    subparser.add_argument(
        "--destination",
        default="memory",
        type=str,
        help="Destination of the nixl descriptors [file, memory, gpu] (default: memory)",
    )
    subparser.add_argument(
        "--backend", type=str, help="Communication backend [UCX, UCX_MO] (default: UCX)"
    )
    subparser.add_argument(
        "--worker_type",
        type=str,
        help="Worker to use to transfer data [nixl, nvshmem] (default: nixl)",
    )
    subparser.add_argument(
        "--initiator_seg_type",
        type=str,
        help="Memory segment type for initiator [DRAM, VRAM] (default: DRAM)",
    )
    subparser.add_argument(
        "--target_seg_type",
        type=str,
        help="Memory segment type for target [DRAM, VRAM] (default: DRAM)",
    )
    subparser.add_argument(
        "--scheme",
        type=str,
        help="Communication scheme [pairwise, manytoone, onetomany, tp] (default: pairwise)",
    )
    subparser.add_argument(
        "--mode",
        type=str,
        help="Process mode [SG (Single GPU per proc), MG (Multi GPU per proc)] (default: SG)",
    )
    subparser.add_argument(
        "--op_type", type=str, help="Operation type [READ, WRITE] (default: WRITE)"
    )
    subparser.add_argument(
        "--check_consistency", action="store_true", help="Enable consistency checking"
    )
    subparser.add_argument(
        "--total_buffer_size", type=int, help="Total buffer size (default: 8GiB)"
    )
    subparser.add_argument(
        "--start_block_size", type=int, help="Starting block size (default: 4KiB)"
    )
    subparser.add_argument(
        "--max_block_size", type=int, help="Maximum block size (default: 64MiB)"
    )
    subparser.add_argument(
        "--start_batch_size", type=int, help="Starting batch size (default: 1)"
    )
    subparser.add_argument(
        "--max_batch_size", type=int, help="Maximum batch size (default: 1)"
    )
    subparser.add_argument(
        "--num_iter", type=int, help="Number of iterations (default: 1000)"
    )
    subparser.add_argument(
        "--warmup_iter", type=int, help="Number of warmup iterations (default: 100)"
    )
    subparser.add_argument(
        "--num_threads",
        type=int,
        help="Number of threads used by benchmark (default: 1)",
    )
    subparser.add_argument(
        "--num_initiator_dev",
        type=int,
        help="Number of devices in initiator processes (default: 1)",
    )
    subparser.add_argument(
        "--num_target_dev",
        type=int,
        help="Number of devices in target processes (default: 1)",
    )
    subparser.add_argument(
        "--enable_pt", action="store_true", help="Enable progress thread"
    )
    subparser.add_argument(
        "--device_list", type=str, help="Comma-separated device names (default: all)"
    )
    subparser.add_argument(
        "--runtime_type", type=str, help="Type of runtime to use [ETCD] (default: ETCD)"
    )
    subparser.add_argument(
        "--etcd-endpoints",
        type=str,
        help="ETCD server URL for coordination (default: http://localhost:2379)",
    )
    subparser.add_argument(
        "--storage_enable_direct",
        type=bool,
        help="Enable direct I/O for storage operations (only used with POSIX backend)",
        default=False,
    )
    subparser.add_argument(
        "--gds_filepath", type=str, help="(File path for GDS operations"
    )
    subparser.add_argument(
        "--enable_vmm",
        action="store_true",
        help="Enable VMM memory allocation when DRAM is requested",
    )
