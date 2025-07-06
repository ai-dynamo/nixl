#!/bin/bash

# Check if a repository path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_repository>"
  echo "Example: $0 ~/my_cloned_repo"
  exit 1
fi

REPO_PATH="$1"

# Validate that the provided path is a directory
if [ ! -d "$REPO_PATH" ]; then
  echo "Error: The provided path '$REPO_PATH' is not a valid directory."
  exit 1
fi

echo "Checking for BUILT-IN 'print()' calls in Python files within: $REPO_PATH"
echo "---------------------------------------------------------------------"

FOUND_PRINT=false

# Find all Python files and process them
find "$REPO_PATH" -name "*.py" | while read -r py_file; do
  # Use grep to find 'print()' calls with line numbers, then filter out method calls.
  # First grep: finds all occurrences of 'print(' with word boundary.
  # Second grep: filters out lines where 'print(' is preceded by a dot and optional whitespace.
  MATCHES=$(grep -nE '\bprint\s*\(' "$py_file" | grep -vE '\.[[:space:]]*print\s*\(')

  if [ -n "$MATCHES" ]; then
    echo "Found built-in 'print()' in: $py_file"
    echo "$MATCHES" | sed 's/^/  Line /' # Indent and prepend "Line "
    echo # Add a blank line for readability
    FOUND_PRINT=true
  fi
done

echo "---------------------------------------------------------------------"

if [ "$FOUND_PRINT" = true ]; then
  echo "One or more Python files in '$REPO_PATH' contain built-in 'print()' calls."
else
  echo "No built-in 'print()' calls found in any Python files within '$REPO_PATH'."
fi
