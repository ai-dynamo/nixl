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
from models.model_config import ModelConfig
from models.models import BaseModelArch
from commands.args import add_common_args

class Command:
    """
    Command handler for the 'kvcache' subcommand.
    
    This command analyzes and displays key-value cache size information for the 
    specified model architecture and configuration, including per-token size,
    total size, and related metrics.
    """
    
    def __init__(self):
        """
        Initialize the kvcache command.
        
        Sets the command name and help text for the command-line interface.
        """
        self.name = "kvcache"
        self.help = "Display kvcache information"

    def add_arguments(self, subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add command-specific arguments to the argument parser.
        
        Args:
            subparser (argparse.ArgumentParser): The parser for this command.
            
        Returns:
            argparse.ArgumentParser: The updated argument parser with added arguments.
        """
        add_common_args(subparser)
        return subparser
    
    def execute(self, args: argparse.Namespace):
        """
        Execute the kvcache command with the provided arguments.
        
        Loads the model architecture and configuration from the specified files,
        calculates KV cache metrics (per-token size, total size, page size), and
        displays the information in a formatted output.
        
        Args:
            args (argparse.Namespace): Command-line arguments.
            
        Returns:
            int: -1 if required arguments are missing, otherwise None.
        """
        if not args.model or not args.model_config:
            print("Error: --model and --model_config are required")
            return -1
        
        # Load model architecture
        model = BaseModelArch.from_yaml(args.model)
        
        # Load model configuration
        model_config = ModelConfig.from_yaml(args.model_config)
        
        # Set model_config on the model instance using the new method
        model.set_model_config(model_config)

        from math import floor, log

        def format_bytes(size):
            power = 0 if size <= 0 else floor(log(size, 1024))
            return f"{round(size / 1024 ** power, 2)} {['B', 'KB', 'MB', 'GB', 'TB'][int(power)]}"

        # Get basic parameters
        kv_size_per_token = model.get_kv_size_per_token()
        batch_size = model.model_config.runtime.batch_size
        isl = model.model_config.runtime.isl
        osl = model.model_config.runtime.osl
        
        # Calculate metrics for ISL (Input Sequence)
        kv_size_per_page_isl = batch_size * kv_size_per_token
        kv_size_total_isl = kv_size_per_token * isl
        
        # Calculate metrics for OSL (Output Sequence) 
        kv_size_total_osl = kv_size_per_token * osl
        
        # Calculate combined size (ISL + OSL)
        kv_size_total_combined = kv_size_per_token * (isl + osl)
        
        # Determine labels and max width for formatting
        labels = [
            "KV Cache Size Per Token", 
            "Page Size (batch)",
            "Input Sequence Length (ISL)",
            "KV Cache Size For ISL",
            "Output Sequence Length (OSL)",
            "KV Cache Size For OSL",
            "Total Sequence Length (ISL+OSL)",
            "Total KV Cache Size (ISL+OSL)",
            "Batch Size"
        ]
        max_width = max(len(label) for label in labels)
        
        # Display metrics
        print("KV Cache Information:")
        print("-" * (max_width + 20))
        print(f"{'KV Cache Size Per Token':{max_width}}: {format_bytes(kv_size_per_token)}")
        print(f"{'Batch Size':{max_width}}: {batch_size}")
        print(f"{'Page Size (batch)':{max_width}}: {format_bytes(kv_size_per_page_isl)}")
        print("-" * (max_width + 20))
        print(f"{'Input Sequence Length (ISL)':{max_width}}: {isl}")
        print(f"{'KV Cache Size For ISL':{max_width}}: {format_bytes(kv_size_total_isl)}")
        print("-" * (max_width + 20))
        print(f"{'Output Sequence Length (OSL)':{max_width}}: {osl}")
        print(f"{'KV Cache Size For OSL':{max_width}}: {format_bytes(kv_size_total_osl)}")
        print("-" * (max_width + 20))
        print(f"{'Total Sequence Length (ISL+OSL)':{max_width}}: {isl + osl}")
        print(f"{'Total KV Cache Size (ISL+OSL)':{max_width}}: {format_bytes(kv_size_total_combined)}")