#!/bin/bash

# Number of rows and columns (64x64 matrix)
N=64

# Output file
OUTPUT_FILE="test_matrix2_8N.txt"

# Clear the file if it exists
> $OUTPUT_FILE

# Generate the matrix
for ((i=0; i<N; i++)); do
    row=""
    for ((j=0; j<N; j++)); do
        if [[ $j -le $i ]]; then
            # On or below diagonal - put 0
            row+="0 "
        else
            # Above diagonal - put 128M
            row+="128M "
        fi
    done
    # Remove trailing space and add newline
    echo "${row%?}" >> $OUTPUT_FILE
done
