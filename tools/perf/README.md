# CTPerftest - Custom Traffic Performance Test Implementation

## Overview
This PR introduces a new performance testing capability that measure the performance of customizable traffic patterns, where each rank sends a custom message size to every other rank (can be 0). This is similar to [ucc alltoallv perftest](https://github.com/openucx/ucc/pull/973). 

Such a pattern is defined using a transfer matrix, i.e a matrix where cell [i.j] defines the size of the message sent by rank i to rank j. 

This PR implements two tests:
- CT Perftest: Benchmark the performance of one traffic pattern, the pattern is ran in multiple iterations and then metrics are reported. It is useful to optimize specific patterns
- Sequential CT Perftest: Benchmark the performance of a continuum of traffic patterns one after the other, before running each pattern, all the ranks involved in the pattern do a barrier, then optionally sleep a given amount of time, and then run the pattern and measure the execution time.

## Implementation Details
- `custom_traffic_perftest.py` implements CT Perftest and the TrafficPattern dataclass
- `sequential_custom_traffic_perftest.py` implements Sequential CT Perftest
- Utilizes common utilities from `utils.py` and `dist_utils.py` for distributed testing support
- Command-line interface extensions in `cli.py` for test configuration
- `common.py` defines an abstraction for NixlBuffers
- `inference_workload_matgen.py` - Small script to generate transfer matrices of KV transfers, with very basic computation simulation, the output of this script will be a lot of matrices and a `metadata.yaml` file that can be directly fed to Sequential CT Benchmark.

## Reports
The metrics reported differ for CT and Sequential-CT perftests. 

CT Perftest reports total latency (the time elapsed between the first rank started until the last rank finished), the average time per iteration, the total size sent over the network and the average bandwidth by rank.

Sequential CT Perftest reports the total latency per matrix execution, along with their latency when ran isolated, which can be used to evaluate how good the network react to congestion. The report is both saved as a JSON file (controlled by the `--json-output-path` parameter) and displayed in stdout as:

```
  TP size (GB)    Latency (ms)    Isolated Latency (ms)    Num Senders
--------------  --------------  -----------------------  -------------
         4.945          35.047                   35.421              4
         3.230          21.152                   21.800              4
         1.104           8.222                    8.280              4
         ...            ...                         ...             ...
         0.129           2.147                    2.386              4
```


## Usage
Tests can be defined using YAML configuration files. 

CT Perftest define a single traffic pattern, as well as number of iterations and warmup iterations:
```yaml
iters: 50
warmup_iters: 10
traffic_pattern:
  matrix_file: "/path/to/matrix.txt"
  shards: 1
  mem_type: "cuda"
  xfer_op: "WRITE"
```

Sequential CT Perftest configuration defines a sequence of traffic patterns:

```yaml
traffic_patterns:
- matrix_file: /swgwork/eshukrun/nixl/tools/perf/run/llama-405b/prefill_tp_4_decode_tp_8/matrices/matrix_0.txt
  metadata:
    isl: 38328
  sleep_before_launch_sec: 168.480753057792
- matrix_file: /swgwork/eshukrun/nixl/tools/perf/run/llama-405b/prefill_tp_4_decode_tp_8/matrices/matrix_1.txt
  metadata:
    isl: 25034
  sleep_before_launch_sec: 71.875102179328
```
`traffic_patterns` can contain multiple elements that run sequentially. See `TrafficPattern` in `common.py` for default values.

- **matrix_file**: The file containing the matrix, the matrix cells should be separated by whitespaces and contain either a number of bytes as integer or use a standard unit like K, M and G. 
- **shards**: Number of chunks the buffer has to be sharded into.
- **mem_type**: For now support only cuda, but it should follow nixl memory types
- **xfer_op**:  xfer operation, can be READ or WRITE
- **sleep_before_launch_sec**: number of seconds to sleep before running this traffic pattern, can be used for example to simulate computation. 

Example of a matrix file:
```
0 0 1M 1M
0 0 1M 1M
0 0 0 5K
0 0 0 5K
```

## Usage
Run the test using the CLI:
```bash
# Sequential CT benchmark
python tools/perf/cli.py sequential-ct-perftest ./matrices/metadata.yaml --verify-buffers --json-output-path ./results.json

# CT benchmark
python tools/perf/cli.py  ct-perftest path/to/config.yaml
```

## Next Steps
- [ ] Support more memory types

## Known Issues
- The nixl xfer preparation currently takes a lot of time (the `_prepare_tp()` method).
