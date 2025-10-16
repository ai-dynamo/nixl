#! /usr/bin/env python3

import tomlkit
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--set-name", type=str, help="Set the project name")
parser.add_argument("file", type=str, help="The toml file to modify")
args = parser.parse_args()

with open(args.file) as f:
    doc = tomlkit.parse(f.read())

if args.set_name:
    doc["project"]["name"] = args.set_name

with open(args.file, "w") as f:
    f.write(tomlkit.dumps(doc))
