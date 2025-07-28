#!/bin/bash
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

# Check if a path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_python_directory>"
  echo "Example: $0 ./path/to/python/directory"
  exit 1
fi

DIR_PATH="$1"

# Validate that the provided path is a directory
if [ ! -d "$DIR_PATH" ]; then
  echo "Error: The provided path '$DIR_PATH' is not a valid directory."
  exit 1
fi

echo "Checking for BUILT-IN 'print()' calls in Python files within: $DIR_PATH"
echo "---------------------------------------------------------------------"

found_print=false

# Find all Python files and process them
while read -r py_file; do
  # Use grep to find 'print()' calls with line numbers, then filter out method calls.
  # First grep: finds all occurrences of 'print(' with word boundary.
  # Second grep: filters out lines where 'print(' is preceded by a dot and optional whitespace.
  MATCHES=$(grep -nE '\bprint\s*\(' "$py_file" | grep -vE '\.[[:space:]]*print\s*\(')

  if [ -n "$MATCHES" ]; then
    echo "Found built-in 'print()' in: $py_file"
    echo "${MATCHES//$'\n'/$'\n' Line }" # Indent and prepend "Line "
    echo # Add a blank line for readability
    found_print=true
  fi
done < <(find "$DIR_PATH" -name "*.py")

echo "---------------------------------------------------------------------"

if [ "$found_print" = true ]; then
  echo "One or more Python files in '$DIR_PATH' contain built-in 'print()' calls."
  exit 1
else
  echo "No built-in 'print()' calls found in any Python files within '$DIR_PATH'."
  exit 0
fi
