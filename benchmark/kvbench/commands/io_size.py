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
    Command handler for the 'iops' subcommand.
    
    This command calculates and displays the number of IOPs for a given model and configuration.
    """
        
    def __init__(self):
        """
        Initialize the iops command.
        
        Sets the command name and help text for the command-line interface.
        """
        self.name = "iosize"
        self.help = "Display io size information"

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
        Execute the iops command with the provided arguments.
        
        Loads the model architecture and configuration from the specified files,
        calculates the number of IOPs for a given model and configuration, and
        displays the information in a formatted output.
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
        io_size = model.get_io_size()
        print(f"IO Size per GPU: {io_size}")



