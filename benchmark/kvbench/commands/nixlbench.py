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
        etcd_endpoints="http://localhost:2379",
        storage_enable_direct=False,
        gds_filepath="",
        initiator_seg_type="DRAM",
        enable_vmm=False,
        max_batch_size=None,
        max_block_size=None,
        mode="SG",
        num_initiator_dev=1,
        num_iter=1000,
        num_target_dev=1,
        num_threads=1,
        op_type="WRITE",
        runtime_type="ETCD",
        scheme="pairwise",
        start_batch_size=None,
        start_block_size=None,
        target_seg_type="DRAM",
        total_buffer_size=None,
        warmup_iter=100,
        worker_type="nixl",
    ):
        """
        Initialize a NIXLBench instance with benchmark configuration.

        Args:
            model (BaseModelArch): Model architecture specification.
            model_config (ModelConfig): Model runtime and system configuration.
            backend (str, optional): Communication backend. Defaults to "UCX".
            check_consistency (bool, optional): Whether to check consistency. Defaults to False.
            device_list (str, optional): List of devices to use. Defaults to "all".
            enable_pt (bool, optional): Whether to enable peer-to-peer transfer. Defaults to False.
            etcd_endpoints (str, optional): ETCD endpoints for runtime. Defaults to "http://localhost:2379".
            storage_enable_direct (bool, optional): Whether to enable direct I/O for storage operations. Defaults to False.
            gds_filepath (str, optional): Path for GDS file. Defaults to "".
            enable_vmm (bool, optional): Whether to use VMM memory allocation. Defaults to False.
            initiator_seg_type (str, optional): Type of initiator segment. Defaults to "DRAM".
            max_batch_size (int, optional): Maximum batch size for testing. Defaults to model_config value.
            max_block_size (int, optional): Maximum block size for testing. Defaults to tp_size * isl.
            mode (str, optional): Benchmarking mode. Defaults to "SG".
            num_initiator_dev (int, optional): Number of initiator devices. Defaults to 1.
            num_iter (int, optional): Number of iterations. Defaults to 1000.
            num_target_dev (int, optional): Number of target devices. Defaults to 1.
            num_threads (int, optional): Number of threads. Defaults to 1.
            op_type (str, optional): Operation type. Defaults to "WRITE".
            runtime_type (str, optional): Runtime type. Defaults to "ETCD".
            scheme (str, optional): Communication scheme. Defaults to "pairwise".
            start_batch_size (int, optional): Starting batch size. Defaults to 1.
            start_block_size (int, optional): Starting block size. Defaults to 4096.
            target_seg_type (str, optional): Type of target segment. Defaults to "DRAM".
            total_buffer_size (int, optional): Total buffer size. Defaults to 8589934592.
            warmup_iter (int, optional): Number of warmup iterations. Defaults to 100.
            worker_type (str, optional): Type of worker. Defaults to "nixl".
        """
        self.model = model
        self.model_config = model_config
        self.backend = backend
        self.check_consistency = check_consistency
        self.device_list = device_list
        self.enable_pt = enable_pt
        self.etcd_endpoints = etcd_endpoints
        self.storage_enable_direct = storage_enable_direct
        self.gds_filepath = gds_filepath
        self.enable_vmm = enable_vmm
        self.initiator_seg_type = initiator_seg_type
        self.max_batch_size = max_batch_size
        self.max_block_size = max_block_size
        self.mode = mode
        self.num_initiator_dev = num_initiator_dev
        self.num_iter = num_iter
        self.num_target_dev = num_target_dev
        self.num_threads = num_threads
        self.op_type = op_type
        self.runtime_type = runtime_type
        self.scheme = scheme
        self.start_batch_size = start_batch_size
        self.start_block_size = start_block_size
        self.target_seg_type = target_seg_type
        self.total_buffer_size = total_buffer_size
        self.warmup_iter = warmup_iter
        self.worker_type = worker_type
        self._override_defaults()

    def set_io_size(self, io_size: int):
        self.start_block_size = io_size
        self.max_block_size = io_size

    def configure_segment_type(self, backend: str, source: str, destination: str):
        if backend == "GDS":
            if source == "file":
                # this is a READ from GDS to GPU
                self.op_type = "READ"
                self.target_seg_type = "VRAM"
            elif source == "gpu":
                # this is a WRITE from GPU to GDS
                self.op_type = "WRITE"
                self.target_seg_type = "VRAM"

            elif source == "memory":
                # this is a WRITE from memory to GDS
                self.op_type = "WRITE"
                self.initiator_seg_type = "DRAM"
                self.target_seg_type = "DRAM"
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
            "backend": self.backend,
            "check_consistency": self.check_consistency,
            "device_list": self.device_list,
            "enable_pt": self.enable_pt,
            "etcd_endpoints": self.etcd_endpoints,
            "storage_enable_direct": self.storage_enable_direct,
            "gds_filepath": self.gds_filepath,
            "enable_vmm": self.enable_vmm,
            "initiator_seg_type": self.initiator_seg_type,
            "max_batch_size": self.max_batch_size,
            "max_block_size": self.max_block_size,
            "mode": self.mode,
            "num_initiator_dev": self.num_initiator_dev,
            "num_iter": self.num_iter,
            "num_target_dev": self.num_target_dev,
            "num_threads": self.num_threads,
            "op_type": self.op_type,
            "runtime_type": self.runtime_type,
            "scheme": self.scheme,
            "start_batch_size": self.start_batch_size,
            "start_block_size": self.start_block_size,
            "target_seg_type": self.target_seg_type,
            "total_buffer_size": self.total_buffer_size,
            "warmup_iter": self.warmup_iter,
            "worker_type": self.worker_type,
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
            "etcd_endpoints": "http://localhost:2379",
            "storage_enable_direct": False,
            "gds_filepath": "",
            "enable_vmm": False,
            "initiator_seg_type": "DRAM",
            "max_batch_size": 1,  # ios per gpu
            "max_block_size": 67108864,  # io size
            "mode": "SG",
            "num_initiator_dev": 1,
            "num_iter": 1000,
            "num_target_dev": 1,
            "num_threads": 1,
            "op_type": "WRITE",
            "runtime_type": "ETCD",
            "scheme": "pairwise",
            "start_batch_size": 1,
            "start_block_size": 4096,
            "target_seg_type": "DRAM",
            "total_buffer_size": 8589934592,
            "warmup_iter": 100,
            "worker_type": "nixl",
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
            # print(json.dumps(merged_params))
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
