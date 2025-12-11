#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run all NIXL EP performance tests and generate a report
# Usage: ./run_all_perf.sh [NUM_PROCESSES] [OUTPUT_DIR]

set -e

NUM_PROCESSES=${1:-8}
OUTPUT_DIR=${2:-/tmp/nixl_perf_results}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "NIXL EP Performance Test Suite"
echo "=============================================="
echo "Processes: $NUM_PROCESSES"
echo "Output dir: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")"

# Run control plane tests
echo ""
echo ">>> Running Control Plane Tests..."
python3 test_control_plane.py \
    --num-processes=$NUM_PROCESSES \
    --experts-per-rank=2,4,8,16,32 \
    --test=cycle \
    --timeout=600 \
    --output="$OUTPUT_DIR/ctrl_plane_$TIMESTAMP.json" \
    2>&1 | tee "$OUTPUT_DIR/ctrl_plane_$TIMESTAMP.log"

# Run data plane tests
echo ""
echo ">>> Running Data Plane Tests..."
python3 test_data_plane.py \
    --num-processes=$NUM_PROCESSES \
    --test=all \
    --tokens=512,2048 \
    --hidden=4096 \
    --experts-per-rank=8 \
    --timeout=300 \
    --output="$OUTPUT_DIR/data_plane_$TIMESTAMP.json" \
    2>&1 | tee "$OUTPUT_DIR/data_plane_$TIMESTAMP.log"

# Generate combined report
echo ""
echo ">>> Generating Report..."

# Merge JSON files
python3 -c "
import json
import sys
from datetime import datetime

ctrl_file = '$OUTPUT_DIR/ctrl_plane_$TIMESTAMP.json'
data_file = '$OUTPUT_DIR/data_plane_$TIMESTAMP.json'
output_file = '$OUTPUT_DIR/combined_$TIMESTAMP.json'

combined = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'num_processes': $NUM_PROCESSES,
    },
    'results': {}
}

try:
    with open(ctrl_file) as f:
        ctrl = json.load(f)
    combined['results'].update(ctrl.get('results', {}))
    combined['config'].update(ctrl.get('config', {}))
except Exception as e:
    print(f'Warning: Could not load control plane results: {e}')

try:
    with open(data_file) as f:
        data = json.load(f)
    combined['results'].update(data.get('results', {}))
except Exception as e:
    print(f'Warning: Could not load data plane results: {e}')

with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)

print(f'Combined results saved to: {output_file}')
"

# Generate markdown report
python3 report_generator.py \
    --input="$OUTPUT_DIR/combined_$TIMESTAMP.json" \
    --output="$OUTPUT_DIR/report_$TIMESTAMP.md"

echo ""
echo "=============================================="
echo "Performance Test Suite Complete!"
echo "=============================================="
echo "Results: $OUTPUT_DIR/"
echo "  - ctrl_plane_$TIMESTAMP.json"
echo "  - data_plane_$TIMESTAMP.json"
echo "  - combined_$TIMESTAMP.json"
echo "  - report_$TIMESTAMP.md"
echo "=============================================="

