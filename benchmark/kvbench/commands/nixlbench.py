from models.models import BaseModelArch
from config.model_config import ModelConfig

class NIXLBench:
    """
    NIXL Benchmarking utility for KV cache performance testing.
    
    This class provides a configurable interface for running benchmarks
    on NIXL with various parameters and configurations. It handles parameter
    validation, default values, and command generation.
    """

    def __init__(self,
                 model: BaseModelArch,
                 model_config: ModelConfig,
                 backend="UCX",
                 check_consistency=False,
                 device_list="all",
                 enable_pt=False,
                 etcd_endpoints="http://localhost:2379",
                 gds_enable_direct=False,
                 gds_filepath="",
                 initiator_seg_type="DRAM",
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
                 worker_type="nixl"):
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
            gds_enable_direct (bool, optional): Whether to enable GDS direct access. Defaults to False.
            gds_filepath (str, optional): Path for GDS file. Defaults to "".
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
        self.gds_enable_direct = gds_enable_direct
        self.gds_filepath = gds_filepath
        self.initiator_seg_type = initiator_seg_type
        self.max_batch_size = self.model_config.runtime.batch_size
        self.max_block_size = self.model_config.model.tp_size * self.model_config.runtime.isl
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

    def _override_defaults(self):
        """
        Set default values for parameters that were not explicitly provided.
        
        This method is called during initialization to ensure all required
        parameters have valid values before running benchmarks.
        """
        if self.max_batch_size is None:
            self.max_batch_size = 1
        if self.max_block_size is None:
            self.max_block_size = self.model_config.model.tp_size * self.model_config.runtime.isl
        if self.start_batch_size is None:
            self.start_batch_size = 1
        if self.start_block_size is None:
            self.start_block_size = 4096
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
            "gds_enable_direct": self.gds_enable_direct,
            "gds_filepath": self.gds_filepath,
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
            "worker_type": self.worker_type
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
            "gds_enable_direct": False,
            "gds_filepath": "",
            "initiator_seg_type": "DRAM",
            "max_batch_size": 1,
            "max_block_size": 67108864,
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
            "worker_type": "nixl"
        }

    def plan(self):
        """
        Generate the nixlbench command with appropriate parameters.
        
        This method builds a command string for the nixlbench tool,
        including only non-default parameters to keep the command concise.
        The generated command is printed to the console.
        """
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
                command_parts.append(f"--{name} {value}")
        
        command = " \\\n    ".join(command_parts)
        print(command)

    def profile(self):
        """
        Generate the nixlbench command with appropriate parameters.
        
        This method builds a command string for the nixlbench tool,
        including only non-default parameters to keep the command concise.
        The generated command is printed to the console.
        """
        import subprocess
        import os
        def nixl_bench_exists():
            return os.path.exists("nixlbench") and os.access("nixlbench", os.X_OK)

        if not nixl_bench_exists():
            print("nixlbench not found")
            return -1

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
        subprocess.run(command_parts, capture_output=True)