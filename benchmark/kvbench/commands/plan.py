import argparse
from models.model_config import ModelConfig
from models.models import BaseModelArch
from commands.args import add_common_args, add_nixl_bench_args
from commands.nixlbench import NIXLBench

class Command:
    """
    Command handler for the 'plan' subcommand.
    
    This command displays the recommended configuration for nixlbench based on
    the provided model architecture and model configuration files, showing both
    ISL (Input Sequence Length) and OSL (Output Sequence Length) versions.
    """
    
    def __init__(self):
        """
        Initialize the plan command.
        
        Sets the command name and help text for the command-line interface.
        """
        self.name = "plan"
        self.help = "Display the recommended configuration for nixlbench"

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
        creates NIXLBench instances for both ISL and OSL configurations, and 
        generates nixlbench command plans for both sequence types.
        
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
        
        # Set model_config on the model instance
        model.set_model_config(model_config)
        
        filtered_args = {k: v for k, v in args.__dict__.items() if k in NIXLBench.defaults()}
        
        # Create a horizontal separator for better readability
        separator = "=" * 80
        # Create and display ISL nixlbench configuration
        isl_nixl_bench = NIXLBench(model, model_config, **filtered_args)
        osl_nixl_bench = NIXLBench(model, model_config, **filtered_args)

        isl_nixl_bench.set_io_size(model.get_io_size(model_config.runtime.isl))
        osl_nixl_bench.set_io_size(model.get_io_size(model_config.runtime.osl))      
        
        isl_nixl_bench.configure_scheme(direction="isl")
        osl_nixl_bench.configure_scheme(direction="osl")
    
        print(separator)
        print("NIXL BENCHMARK COMMAND FOR ISL (INPUT SEQUENCE)")
        print(f"ISL: {model_config.runtime.isl} tokens")
        print(separator)
        isl_nixl_bench.plan()
        print(separator)
        print("NIXL BENCHMARK COMMAND FOR OSL (OUTPUT SEQUENCE)")
        print(f"OSL: {model_config.runtime.osl} tokens")
        print(separator)
        osl_nixl_bench.plan()
        print("\nNOTE: Use the appropriate command based on whether you're benchmarking")
        print("      input sequence (prefill) or output sequence (generation) performance.")