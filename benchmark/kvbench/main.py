import sys
import commands 
import argparse 
def main():
    parser = argparse.ArgumentParser(description="KVBench")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    for command in commands.available_commands:
        subparser = subparsers.add_parser(command.name, help=command.help)
        command.add_arguments(subparser)

    args = parser.parse_args()

    if args.command:
        for command in commands.available_commands:
            if command.name == args.command:
                command.execute(args)
                break
    else:
        parser.print_help()

if __name__ == "__main__":
    main()