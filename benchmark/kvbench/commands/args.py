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

import click


def common_args(func):
    """Decorator for common model arguments"""
    func = click.option(
        "--model", type=str, help="Path to a model architecture YAML file"
    )(func)
    func = click.option(
        "--model_config", type=str, help="Path to a single model config YAML file"
    )(func)
    func = click.option(
        "--model_configs",
        type=str,
        help="Path to multiple model config YAML files (supports glob patterns like 'configs/*.yaml')",
    )(func)
    return func


def cli_args(func):
    """Decorator for CLI-specific arguments"""
    func = click.option("--pp", type=int, help="Pipeline parallelism size")(func)
    func = click.option("--tp", type=int, help="Tensor parallelism size")(func)
    func = click.option("--isl", type=int, help="Input sequence length")(func)
    func = click.option("--osl", type=int, help="Output sequence length")(func)
    func = click.option("--num_requests", type=int, help="Number of requests")(func)
    func = click.option("--page_size", type=int, help="Page size")(func)
    func = click.option("--access_pattern", type=str, help="Access pattern")(func)
    func = click.option(
        "--source",
        default="file",
        type=str,
        help="Source of the nixl descriptors [file, memory, gpu] (default: file)",
    )(func)
    func = click.option(
        "--destination",
        default="memory",
        type=str,
        help="Destination of the nixl descriptors [file, memory, gpu] (default: memory)",
    )(func)
    return func


def plan_args(func):
    """Decorator for plan-specific arguments"""
    func = click.option(
        "--format",
        default="text",
        type=str,
        help="Output of the nixl command [text, json, csv] (default: text)",
    )(func)
    return func


def nixl_bench_args(func):
    """Decorator for NIXL benchmark arguments"""
    func = click.option(
        "--backend",
        type=str,
        help="Communication backend [UCX, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, HF3FS, OBJ] (default: UCX)",
    )(func)
    func = click.option(
        "--worker_type",
        type=str,
        help="Worker to use to transfer data [nixl, nvshmem] (default: nixl)",
    )(func)
    func = click.option(
        "--initiator_seg_type",
        type=str,
        help="Memory segment type for initiator [DRAM, VRAM, FILE, OBJ] (default: DRAM)",
    )(func)
    func = click.option(
        "--target_seg_type",
        type=str,
        help="Memory segment type for target [DRAM, VRAM, FILE, OBJ] (default: DRAM)",
    )(func)
    func = click.option(
        "--scheme",
        type=str,
        help="Communication scheme [pairwise, manytoone, onetomany, tp] (default: pairwise)",
    )(func)
    func = click.option(
        "--mode",
        type=str,
        help="Process mode [SG (Single GPU per proc), MG (Multi GPU per proc)] (default: SG)",
    )(func)
    func = click.option(
        "--op_type", type=str, help="Operation type [READ, WRITE] (default: WRITE)"
    )(func)
    func = click.option(
        "--check_consistency", is_flag=True, help="Enable consistency checking"
    )(func)
    func = click.option(
        "--total_buffer_size", type=int, help="Total buffer size (default: 8GiB)"
    )(func)
    func = click.option(
        "--start_block_size", type=int, help="Starting block size (default: 4KiB)"
    )(func)
    func = click.option(
        "--max_block_size", type=int, help="Maximum block size (default: 64MiB)"
    )(func)
    func = click.option(
        "--start_batch_size", type=int, help="Starting batch size (default: 1)"
    )(func)
    func = click.option(
        "--max_batch_size", type=int, help="Maximum batch size (default: 1)"
    )(func)
    func = click.option(
        "--num_iter", type=int, help="Number of iterations (default: 1000)"
    )(func)
    func = click.option(
        "--warmup_iter", type=int, help="Number of warmup iterations (default: 100)"
    )(func)
    func = click.option(
        "--num_threads",
        type=int,
        help="Number of threads used by benchmark (default: 1)",
    )(func)
    func = click.option(
        "--num_initiator_dev",
        type=int,
        help="Number of devices in initiator processes (default: 1)",
    )(func)
    func = click.option(
        "--num_target_dev",
        type=int,
        help="Number of devices in target processes (default: 1)",
    )(func)
    func = click.option("--enable_pt", is_flag=True, help="Enable progress thread")(
        func
    )
    func = click.option(
        "--progress_threads", type=int, help="Number of progress threads (default: 0)"
    )(func)
    func = click.option(
        "--device_list", type=str, help="Comma-separated device names (default: all)"
    )(func)
    func = click.option(
        "--runtime_type", type=str, help="Type of runtime to use [ETCD] (default: ETCD)"
    )(func)
    func = click.option(
        "--etcd_endpoints",
        type=str,
        help="ETCD server URL for coordination (default: http://localhost:2379)",
    )(func)
    func = click.option(
        "--storage_enable_direct",
        type=bool,
        help="Enable direct I/O for storage operations",
        default=False,
    )(func)
    func = click.option(
        "--filepath", type=str, help="File path for storage operations"
    )(func)
    func = click.option(
        "--posix_api_type",
        type=str,
        help="API type for POSIX operations [AIO, URING] (only used with POSIX backend",
    )(func)
    func = click.option(
        "--enable_vmm",
        type=bool,
        help="Enable VMM memory allocation when DRAM is requested",
    )(func)
    func = click.option(
        "--benchmark_group",
        type=str,
        help="Name of benchmark group (default: default). Use different names to run multiple benchmarks in parallel",
        default="default",
    )(func)
    # Missing arguments from nixlbench
    func = click.option(
        "--num_files",
        type=int,
        help="Number of files used by benchmark (default: 1)",
    )(func)
    func = click.option(
        "--large_blk_iter_ftr",
        type=int,
        help="Factor to reduce test iteration when testing large block size(>1MB) (default: 16)",
    )(func)
    func = click.option(
        "--gds_batch_pool_size",
        type=int,
        help="Batch pool size for GDS operations (default: 32, only used with GDS backend)",
    )(func)
    func = click.option(
        "--gds_batch_limit",
        type=int,
        help="Batch limit for GDS operations (default: 128, only used with GDS backend)",
    )(func)
    func = click.option(
        "--gds_mt_num_threads",
        type=int,
        help="Number of threads used by GDS MT plugin (default: 1)",
    )(func)
    func = click.option(
        "--gpunetio_device_list",
        type=str,
        help="Comma-separated GPU CUDA device id to use for communication (only used with GPUNETIO backend)",
    )(func)
    func = click.option(
        "--gpunetio_oob_list",
        type=str,
        help="OOB network interface name for control path (only used with GPUNETIO backend)",
    )(func)
    func = click.option(
        "--hf3fs_iopool_size",
        type=int,
        help="Size of io memory pool for HF3FS backend (default: 64)",
    )(func)
    func = click.option(
        "--obj_access_key",
        type=str,
        help="Access key for S3 backend (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_secret_key",
        type=str,
        help="Secret key for S3 backend (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_session_token",
        type=str,
        help="Session token for S3 backend (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_bucket_name",
        type=str,
        help="Bucket name for S3 backend (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_scheme",
        type=str,
        help="HTTP scheme for S3 backend [http, https] (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_region",
        type=str,
        help="Region for S3 backend (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_use_virtual_addressing",
        type=bool,
        help="Use virtual addressing for S3 backend (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_endpoint_override",
        type=str,
        help="Endpoint override for S3 backend (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_req_checksum",
        type=str,
        help="Required checksum type for S3 backend [supported, required] (only used with OBJ backend)",
    )(func)
    func = click.option(
        "--obj_ca_bundle",
        type=str,
        help="Path to CA bundle for S3 backend (only used with OBJ backend)",
    )(func)
    return func


def ctp_args(func):
    """Decorator for CTP (Custom Traffic Pattern) arguments"""
    func = click.option(
        "--verify-buffers/--no-verify-buffers",
        default=False,
        help="Verify buffer contents after transfer",
    )(func)
    func = click.option(
        "--print-recv-buffers/--no-print-recv-buffers",
        default=False,
        help="Print received buffer contents",
    )(func)
    func = click.option(
        "--json-output-path",
        type=click.Path(),
        help="Path to save JSON output",
        default=None,
    )(func)
    func = click.option(
        "--config-file",
        type=click.Path(exists=True),
        required=True,
        help="Path to YAML config file",
    )(func)
    return func
