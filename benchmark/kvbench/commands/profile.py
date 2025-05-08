import argparse
from models.model_config import ModelConfig
from models.models import BaseModelArch
from commands.args import add_common_args, add_nixl_bench_args
from commands.nixlbench import NIXLBench

class Command:
    """
    Command handler for the 'plan' subcommand.
    
    This command displays the recommended configuration for nixlbench based on
    the provided model architecture and model configuration files.
    """
    
    def __init__(self):
        """
        Initialize the plan command.
        
        Sets the command name and help text for the command-line interface.
        """
        self.name = "profile"
        self.help = "Run nixlbench"

    def add_arguments(self, subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add command-specific arguments to the argument parser.
        
        Args:
            subparser (argparse.ArgumentParser): The parser for this command.
            
        Returns:
            argparse.ArgumentParser: The updated argument parser with added arguments.
        """
        add_common_args(subparser)
        add_nixl_bench_args(subparser)
        return subparser
    
    def execute(self, args: argparse.Namespace):
        """
        Execute the plan command with the provided arguments.
        
        Loads the model architecture and configuration from the specified files,
        creates a NIXLBench instance with the provided arguments, and generates
        a nixlbench command plan.
        
        Args:
            args (argparse.Namespace): Command-line arguments.
            
        Returns:
            int: -1 if required arguments are missing, otherwise None.
        """
        if not args.model or not args.model_config:
            print("Error: --model and --model_config are required")
            return -1
        
        if args.model:
            model = BaseModelArch.from_yaml(args.model)
        if args.model_config:
            model_config = ModelConfig.from_yaml(args.model_config)

        filtered_args = {k: v for k, v in args.__dict__.items() if k in NIXLBench.defaults()}
        nixl_bench = NIXLBench(model, model_config, **filtered_args)
        nixl_bench.profile()