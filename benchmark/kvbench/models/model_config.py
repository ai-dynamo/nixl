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

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class StrategyConfig:
    """
    Configuration for model parallelism and quantization strategy.

    Specifies tensor parallelism, pipeline parallelism, and quantization settings
    for both model weights and KV cache.
    """

    tp_size: int = 1
    pp_size: int = 1
    model_quant_mode: str = "fp8"
    kvcache_quant_mode: str = "fp8"


@dataclass
class RuntimeConfig:
    """
    Configuration for model runtime parameters.

    Specifies batch size and sequence length parameters for inference.
    """

    num_requests: Optional[int] = None
    isl: Optional[int] = None  # input sequence length
    osl: Optional[int] = None  # output sequence length


@dataclass
class SystemConfig:
    """
    Configuration for system hardware and backend.

    Specifies the hardware platform and inference backend to use.
    """

    backend: str = "SGLANG"
    hardware: Optional[str] = None
    page_size: Optional[int] = None
    access_pattern: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None


@dataclass
class ModelConfig:
    """
    Comprehensive configuration for model deployment and benchmarking.

    Combines strategy, runtime, and system configurations into a single object.
    Provides methods for loading from and saving to YAML files.
    """

    model: StrategyConfig = field(default_factory=StrategyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """
        Load configuration from a single YAML file.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Returns:
            ModelConfig: A new ModelConfig instance loaded from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml_files(cls, yaml_paths: List[str]) -> "ModelConfig":
        """
        Load and merge configurations from multiple YAML files.

        Later files in the list override values from earlier files when there are conflicts.

        Args:
            yaml_paths (List[str]): List of paths to YAML configuration files.

        Returns:
            ModelConfig: A new ModelConfig instance with merged configurations.
        """
        config = cls()

        for path in yaml_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    config_dict = yaml.safe_load(f)
                config = config.update(config_dict)
            else:
                print(f"Warning: Config file not found: {path}")

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Create a ModelConfig instance from a dictionary.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration values.

        Returns:
            ModelConfig: A new ModelConfig instance populated with values from the dictionary.
        """
        config = cls()
        return config.update(config_dict)

    def update(self, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Update config with values from a dictionary.

        Creates a new ModelConfig instance with updated values rather than
        modifying the current instance.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration values.

        Returns:
            ModelConfig: A new ModelConfig instance with updated values.
        """
        result = deepcopy(self)

        for section_name, section_dict in config_dict.items():
            if section_name == "strategy" and isinstance(section_dict, dict):
                for key, value in section_dict.items():
                    if hasattr(result.model, key):
                        setattr(result.model, key, value)
            elif section_name == "runtime" and isinstance(section_dict, dict):
                for key, value in section_dict.items():
                    if hasattr(result.runtime, key):
                        setattr(result.runtime, key, value)
            elif section_name == "system" and isinstance(section_dict, dict):
                for key, value in section_dict.items():
                    if hasattr(result.system, key):
                        setattr(result.system, key, value)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the configuration.
        """
        return {
            "strategy": {
                "tp_size": self.model.tp_size,
                "pp_size": self.model.pp_size,
                "model_quant_mode": self.model.model_quant_mode,
                "kvcache_quant_mode": self.model.kvcache_quant_mode,
            },
            "runtime": {
                "num_requests": self.runtime.num_requests,
                "isl": self.runtime.isl,
                "osl": self.runtime.osl,
            },
            "system": {
                "hardware": self.system.hardware,
                "backend": self.system.backend,
                "page_size": self.system.page_size,
                "access_pattern": self.system.access_pattern,
                "source": self.system.source,
                "destination": self.system.destination,
            },
        }

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            yaml_path (str): Path where the YAML file should be saved.
        """
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
