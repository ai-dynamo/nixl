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

from models.model_config import ModelConfig
from models.models import BaseModelArch


class NIXLBench:
    """
    NIXL Benchmarking utility for KV cache performance testing.

    This class provides a configurable interface for running benchmarks
    on NIXL with various parameters and configurations. It handles parameter
    validation, default values, and command generation.
    """

    def __init__(
        self,
        model: BaseModelArch,
        model_config: ModelConfig,
        backend="UCX",
        check_consistency=False,
        device_list="all",
        enable_pt=False,
        progress_threads=0,
        etcd_endpoints="http://localhost:2379",
        storage_enable_direct=False,
        filepath="",
        gds_batch_pool_size=32,
        gds_batch_limit=128,
        initiator_seg_type="DRAM",
        enable_vmm=False,
        max_batch_size=None,
        max_block_size=None,
        mode="SG",
        num_files=1,
        num_initiator_dev=1,
        num_iter=1000,
        num_target_dev=1,
        num_threads=1,
        op_type="WRITE",
        posix_api_type="AIO",
        runtime_type="ETCD",
        scheme="pairwise",
        start_batch_size=None,
        start_block_size=None,
        target_seg_type="DRAM",
        total_buffer_size=None,
        warmup_iter=100,
        worker_type="nixl",
        benchmark_group="default",
        gds_mt_num_threads=1,
        gpunetio_device_list="0",
        gpunetio_oob_list="",
        hf3fs_iopool_size=64,
        obj_access_key="",
        obj_secret_key="",
        obj_session_token="",
        obj_bucket_name="",
        obj_scheme="http",
        obj_region="eu-central-1",
        obj_use_virtual_addressing=False,
        obj_endpoint_override="",
        obj_req_checksum="supported",
        # Additional nixlbench arguments
        large_blk_iter_ftr=16,
    ):
        """
        Initialize a NIXLBench instance with benchmark configuration.

        Args:
            model (BaseModelArch): Model architecture specification.
            model_config (ModelConfig): Model runtime and system configuration.
            benchmark_group (str, optional): Name of benchmark group. Defaults to "default".
            backend (str, optional): Communication backend. Defaults to "UCX".
            check_consistency (bool, optional): Whether to check consistency. Defaults to False.
            device_list (str, optional): List of devices to use. Defaults to "all".
            enable_pt (bool, optional): Whether to enable peer-to-peer transfer. Defaults to False.
            progress_threads (int, optional): Number of progress threads (default: 0).
            etcd_endpoints (str, optional): ETCD endpoints for runtime. Defaults to "http://localhost:2379".
            storage_enable_direct (bool, optional): Whether to enable direct I/O for storage operations. Defaults to False.
            filepath (str, optional): Path for GDS and POSIX operations. Defaults to "".
            gds_batch_pool_size (int, optional): Batch pool size for GDS operations. Defaults to 32.
            gds_batch_limit (int, optional): Batch limit for GDS operations. Defaults to 128.
            initiator_seg_type (str, optional): Type of initiator segment. Defaults to "DRAM".
            enable_vmm (bool, optional): Whether to use VMM memory allocation. Defaults to False.
            max_batch_size (int, optional): Maximum batch size for testing. Defaults to model_config value.
            max_block_size (int, optional): Maximum block size for testing. Defaults to tp_size * isl.
            mode (str, optional): Benchmarking mode. Defaults to "SG".
            num_files (int, optional): Number of files. Defaults to 1.
            num_initiator_dev (int, optional): Number of initiator devices. Defaults to 1.
            num_iter (int, optional): Number of iterations. Defaults to 1000.
            num_target_dev (int, optional): Number of target devices. Defaults to 1.
            num_threads (int, optional): Number of threads. Defaults to 1.
            op_type (str, optional): Operation type. Defaults to "WRITE".
            posix_api_type (str, optional): POSIX API type. Defaults to "AIO".
            runtime_type (str, optional): Runtime type. Defaults to "ETCD".
            scheme (str, optional): Communication scheme. Defaults to "pairwise".
            start_batch_size (int, optional): Starting batch size. Defaults to 1.
            start_block_size (int, optional): Starting block size. Defaults to 4096.
            target_seg_type (str, optional): Type of target segment. Defaults to "DRAM".
            total_buffer_size (int, optional): Total buffer size. Defaults to 8589934592.
            warmup_iter (int, optional): Number of warmup iterations. Defaults to 100.
            worker_type (str, optional): Type of worker. Defaults to "nixl".
            gds_mt_num_threads (int, optional): Number of threads for GDS_MT plugin. Defaults to 1.
            gpunetio_device_list (str, optional): GPU device list for GPUNETIO plugin. Defaults to "0".
            gpunetio_oob_list (str, optional): OOB network interface name for control path for GPUNETIO plugin. Defaults to "".
            hf3fs_iopool_size (int, optional): IO pool size for HF3FS plugin. Defaults to 64.
            obj_access_key (str, optional): Access key for OBJ/S3 plugin. Defaults to "".
            obj_secret_key (str, optional): Secret key for OBJ/S3 plugin. Defaults to "".
            obj_session_token (str, optional): Session token for OBJ/S3 plugin. Defaults to "".
            obj_bucket_name (str, optional): Bucket name for OBJ/S3 plugin. Defaults to "".
            obj_scheme (str, optional): HTTP scheme for OBJ/S3 plugin. Defaults to "http".
            obj_region (str, optional): Region for OBJ/S3 plugin. Defaults to "eu-central-1".
            obj_use_virtual_addressing (bool, optional): Use virtual addressing for OBJ/S3. Defaults to False.
            obj_endpoint_override (str, optional): Endpoint override for OBJ/S3. Defaults to "".
            obj_req_checksum (str, optional): Required checksum for OBJ/S3. Defaults to "supported".
            large_blk_iter_ftr (int, optional): Factor to reduce iterations for large blocks. Defaults to 16.
        """
        self.model = model
        self.model_config = model_config
        self.benchmark_group = benchmark_group
        self.backend = backend
        self.check_consistency = check_consistency
        self.device_list = device_list
        self.enable_pt = enable_pt
        self.progress_threads = progress_threads
        self.etcd_endpoints = etcd_endpoints
        self.storage_enable_direct = storage_enable_direct
        self.filepath = filepath
        self.enable_vmm = enable_vmm
        self.gds_batch_pool_size = gds_batch_pool_size
        self.gds_batch_limit = gds_batch_limit
        self.initiator_seg_type = initiator_seg_type
        self.max_batch_size = max_batch_size
        self.max_block_size = max_block_size
        self.mode = mode
        self.num_files = num_files
        self.num_initiator_dev = num_initiator_dev
        self.num_iter = num_iter
        self.num_target_dev = num_target_dev
        self.num_threads = num_threads
        self.op_type = op_type
        self.posix_api_type = posix_api_type
        self.runtime_type = runtime_type
        self.scheme = scheme
        self.start_batch_size = start_batch_size
        self.start_block_size = start_block_size
        self.target_seg_type = target_seg_type
        self.total_buffer_size = total_buffer_size
        self.warmup_iter = warmup_iter
        self.worker_type = worker_type
        self.gds_mt_num_threads = gds_mt_num_threads
        self.gpunetio_device_list = gpunetio_device_list
        self.gpunetio_oob_list = gpunetio_oob_list
        self.hf3fs_iopool_size = hf3fs_iopool_size
        self.obj_access_key = obj_access_key
        self.obj_secret_key = obj_secret_key
        self.obj_session_token = obj_session_token
        self.obj_bucket_name = obj_bucket_name
        self.obj_scheme = obj_scheme
        self.obj_region = obj_region
        self.obj_use_virtual_addressing = obj_use_virtual_addressing
        self.obj_endpoint_override = obj_endpoint_override
        self.obj_req_checksum = obj_req_checksum
        self.large_blk_iter_ftr = large_blk_iter_ftr
        self._override_defaults()

    def set_io_size(self, io_size: int):
        self.start_block_size = io_size
        self.max_block_size = io_size

    def _configure_gds(self, source: str, destination: str):
        """Configure GDS and GDS_MT plugins (same logic for both)"""
        if source == "file":
            self.op_type = "READ"
            self.target_seg_type = "VRAM"
        elif source == "gpu":
            self.op_type = "WRITE"
            self.target_seg_type = "FILE"
        else:
            raise ValueError(f"Invalid source for GDS/GDS_MT: {source}")

    def _configure_posix(self, source: str, destination: str):
        """Configure POSIX and HF3FS plugins (same logic for both)"""
        if source == "file":
            self.op_type = "READ"
            self.target_seg_type = "DRAM"
        elif source == "memory":
            self.op_type = "WRITE"
            self.initiator_seg_type = "DRAM"
        else:
            raise ValueError(f"Invalid source for POSIX/HF3FS: {source}")

    def _configure_ucx(self, backend: str, source: str, destination: str):
        """Configure UCX, GPUNETIO, and Mooncake plugins (same logic for all)"""
        arg_to_seg_type = {
            "memory": "DRAM",
            "gpu": "VRAM",
        }

        backend = backend.upper()
        try:
            self.initiator_seg_type = arg_to_seg_type[source]
        except KeyError:
            raise ValueError(
                f"Invalid source for {backend}: {source}, valid sources are: {arg_to_seg_type.keys()}"
            )
        try:
            self.target_seg_type = arg_to_seg_type[destination]
        except KeyError:
            raise ValueError(
                f"Invalid destination for {backend}: {destination}, valid destinations are: {arg_to_seg_type.keys()}"
            )

    def _configure_obj(self, source: str, destination: str):
        """Configure OBJ plugin for object storage operations"""
        if source == "memory":
            self.target_seg_type = "OBJ"
        elif destination == "memory":
            self.initiator_seg_type = "OBJ"
        else:
            raise ValueError(f"Invalid source for OBJ: {source}")

    def configure_segment_type(self, backend: str, source: str, destination: str):
        backend_lower = backend.lower()

        if backend_lower in ["gds", "gds_mt"]:
            self._configure_gds(source, destination)
        elif backend_lower in ["posix", "hf3fs"]:
            self._configure_posix(source, destination)
        elif backend_lower in ["ucx", "gpunetio", "mooncake"]:
            self._configure_ucx(backend_lower, source, destination)
        elif backend_lower == "obj":
            self._configure_obj(source, destination)
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def configure_scheme(self, scheme: str = "pairwise", direction: str = "isl"):
        """
        Configure the scheme based on the model configuration.
        For ISL (input)
        """
        if scheme == "tp":
            if direction == "isl":
                self.num_initiator_dev = 1
                self.num_target_dev = self.model_config.model.tp_size
            elif direction == "osl":
                self.num_initiator_dev = self.model_config.model.tp_size
                self.num_target_dev = 1

    def set_batch_size(self, batch_size: int):
        """
        Set the batch size for benchmarking.
        """

        self.start_batch_size = batch_size
        self.max_batch_size = batch_size

    def configure_buffer_size(self):
        self.total_buffer_size = self.max_batch_size * self.max_block_size

    def _override_defaults(self):
        """
        Set default values for parameters that were not explicitly provided.

        This method is called during initialization to ensure all required
        parameters have valid values before running benchmarks.
        """
        if self.total_buffer_size is None:
            self.total_buffer_size = 8589934592

    def _params(self):
        """
        Collect all benchmark parameters into a dictionary.

        Returns:
            dict: Dictionary containing all benchmark parameters.
        """
        return {
            "benchmark_group": self.benchmark_group,
            "backend": self.backend,
            "check_consistency": self.check_consistency,
            "device_list": self.device_list,
            "enable_pt": self.enable_pt,
            "progress_threads": self.progress_threads,
            "etcd_endpoints": self.etcd_endpoints,
            "storage_enable_direct": self.storage_enable_direct,
            "filepath": self.filepath,
            "enable_vmm": self.enable_vmm,
            "gds_batch_pool_size": self.gds_batch_pool_size,
            "gds_batch_limit": self.gds_batch_limit,
            "initiator_seg_type": self.initiator_seg_type,
            "max_batch_size": self.max_batch_size,
            "max_block_size": self.max_block_size,
            "mode": self.mode,
            "num_files": self.num_files,
            "num_initiator_dev": self.num_initiator_dev,
            "num_iter": self.num_iter,
            "num_target_dev": self.num_target_dev,
            "num_threads": self.num_threads,
            "op_type": self.op_type,
            "posix_api_type": self.posix_api_type,
            "runtime_type": self.runtime_type,
            "scheme": self.scheme,
            "start_batch_size": self.start_batch_size,
            "start_block_size": self.start_block_size,
            "target_seg_type": self.target_seg_type,
            "total_buffer_size": self.total_buffer_size,
            "warmup_iter": self.warmup_iter,
            "worker_type": self.worker_type,
            "gds_mt_num_threads": self.gds_mt_num_threads,
            "gpunetio_device_list": self.gpunetio_device_list,
            "gpunetio_oob_list": self.gpunetio_oob_list,
            "hf3fs_iopool_size": self.hf3fs_iopool_size,
            "obj_access_key": self.obj_access_key,
            "obj_secret_key": self.obj_secret_key,
            "obj_session_token": self.obj_session_token,
            "obj_bucket_name": self.obj_bucket_name,
            "obj_scheme": self.obj_scheme,
            "obj_region": self.obj_region,
            "obj_use_virtual_addressing": self.obj_use_virtual_addressing,
            "obj_endpoint_override": self.obj_endpoint_override,
            "obj_req_checksum": self.obj_req_checksum,
            # Additional nixlbench parameters
            "large_blk_iter_ftr": self.large_blk_iter_ftr,
        }

    @staticmethod
    def defaults():
        """
        Get the default benchmark parameters.

        This static method provides the default values for all benchmark parameters
        when not explicitly specified.

        Returns:
            dict: Dictionary containing default values for all benchmark parameters.
        """
        return {
            "backend": "UCX",
            "check_consistency": False,
            "device_list": "all",
            "enable_pt": False,
            "progress_threads": 0,
            "etcd_endpoints": "http://localhost:2379",
            "storage_enable_direct": False,
            "filepath": "",
            "enable_vmm": False,
            "gds_batch_pool_size": 32,
            "gds_batch_limit": 128,
            "initiator_seg_type": "DRAM",
            "max_batch_size": 1,  # ios per gpu
            "max_block_size": 67108864,  # io size
            "mode": "SG",
            "num_files": 1,
            "num_initiator_dev": 1,
            "num_iter": 1000,
            "num_target_dev": 1,
            "num_threads": 1,
            "op_type": "WRITE",
            "posix_api_type": "AIO",
            "runtime_type": "ETCD",
            "scheme": "pairwise",
            "start_batch_size": 1,
            "start_block_size": 4096,
            "target_seg_type": "DRAM",
            "total_buffer_size": 8589934592,
            "warmup_iter": 100,
            "worker_type": "nixl",
            "benchmark_group": "default",
            "gds_mt_num_threads": 1,
            "gpunetio_device_list": "0",
            "gpunetio_oob_list": "",
            "hf3fs_iopool_size": 64,
            "obj_access_key": "",
            "obj_secret_key": "",
            "obj_session_token": "",
            "obj_bucket_name": "",
            "obj_scheme": "http",
            "obj_region": "eu-central-1",
            "obj_use_virtual_addressing": False,
            "obj_endpoint_override": "",
            "obj_req_checksum": "supported",
            # Additional nixlbench defaults
            "large_blk_iter_ftr": 16,
        }

    def plan(self, format: str = "text"):
        """
        Generate the nixlbench command with appropriate parameters.

        This method builds a command string for the nixlbench tool,
        including only non-default parameters to keep the command concise.
        The generated command is printed to the console.

        For JSON output, all parameters including defaults are included,
        with configured non-null values overriding defaults.
        """
        defaults = NIXLBench.defaults()
        command_parts = ["nixlbench"]

        def should_include(name, value, include_defaults=False):
            if value is None:
                return False
            if not include_defaults and name in defaults and value == defaults[name]:
                return False

            return True

        params = self._params()
        # For JSON output, include all parameters (including defaults)
        if format == "json" or format == "csv":
            # Start with defaults, then update with actual non-null params to override defaults
            merged_params = defaults.copy()
            # Only update with non-null values from params
            for key, value in params.items():
                if value is not None:
                    merged_params[key] = value
            return merged_params
        else:  # for text format, exclude defaults to keep command concise
            for name, value in params.items():
                if should_include(name, value):
                    command_parts.append(f"--{name} {value}")

            command = " \\\n    ".join(command_parts)
            return command

    def profile(self):
        """
        Run the nixlbench command with appropriate parameters.

        This method builds a command for the nixlbench tool,
        including only non-default parameters to keep the command concise,
        and executes it as a subprocess.
        """
        import os
        import subprocess

        env = os.environ.copy()
        defaults = NIXLBench.defaults()
        command_parts = ["nixlbench"]

        def should_include(name, value):
            if value is None:
                return False
            if name in defaults and value == defaults[name]:
                return False
            return True

        params = self._params()
        for name, value in params.items():
            if should_include(name, value):
                command_parts.append(f"--{name}")
                command_parts.append(f"{value}")
        return subprocess.run(command_parts, capture_output=False, env=env)
