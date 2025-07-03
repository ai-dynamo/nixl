# CTPerftest - Custom Traffic Performance Test Implementation

This refers to the commands `ct-perftest` and `sequential-ct-perftest`


## Overview
Benchmark tool to measure the performance of customizable traffic patterns, where each rank sends a custom message size to every other rank (can be 0).

The purpose of such a benchmark is to simulate asymmetric traffic flow, like the KV caches transfers flow in a disaggregated inference system.

Such a pattern is defined using a transfer matrix, i.e a matrix where cell [i.j] defines the size of the message sent by rank i to rank j.

The following benchmarks are available:
- Sequential CT Perftest: Benchmark the performance of a continuum of traffic patterns one after the other, before running each pattern, all the ranks involved in the pattern do a barrier, then optionally sleep a given amount of time, and then run the pattern and measure the execution time.
- CT Perftest: Benchmark the performance of one traffic pattern, the pattern is ran in multiple iterations and then metrics are reported. It is useful to optimize specific patterns


## Implementation Details
- `custom_traffic_perftest.py` implements CT Perftest and the TrafficPattern dataclass
- `sequential_custom_traffic_perftest.py` implements Sequential CT Perftest
- Utilizes common utilities from `runtime`, mostly for collective communication
- `common.py` defines an abstraction for NixlBuffers


# Sequential CT Perftest

### Reports
Sequential CT Perftest reports the total latency per matrix execution, along with their latency when ran isolated, which can be used to evaluate how good the network react to congestion. The report is both saved as a JSON file (controlled by the `--json-output-path` parameter) and displayed in stdout as:

```
  Transfer size (GB)    Latency (ms)    Isolated Latency (ms)    Num Senders
--------------------  --------------  -----------------------  -------------
         4.945               35.047                   35.421              4
         3.230               21.152                   21.800              4
         1.104                8.222                    8.280              4
         ...                 ...                         ...             ...
         0.129                2.147                    2.386              4
```


### Usage
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
  sleep_before_launch_sec: 16.480753057792
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

### Usage
Install requirements
```bash
pip install -r requirements.txt
```

Optionally, generate matrices using the `benchmark/tools/inference_workload_matgen.py` script
```bash
python benchmark/tools/inference_workload_matgen.py generate \
    --num-user-requests 10 \
    --num-prefill-nodes 16 \
    --num-decode-nodes 16 \
    --prefill-tp 8 \
    --prefill-pp 1 \
    --prefill-cp 1 \
    --decode-tp 8 \
    --decode-pp 1 \
    --decode-cp 1 \
    --results-dir $PWD/matrices \
    --model llama-405b
```

Run the test using the CLI:
```bash
python main.py sequential-ct-perftest ./matrices/metadata.yaml --verify-buffers --json-output-path ./results.json
# Or with slurm
srun <params> python main.py sequential-ct-perftest ./matrices/metadata.yaml --verify-buffers --json-output-path ./results.json
```

### Next Steps
- [ ] Support more memory types

### Known Issues
- The nixl xfer preparation currently takes a lot of time (the `_prepare_tp()` method).


# Sequential CT Perftest

### Reports
CT Perftest reports total latency (the time elapsed between the first rank started until the last rank finished), the average time per iteration, the total size sent over the network and the average bandwidth by rank.


### Usage
Tests can be defined using YAML configuration files.


The configuration define a single traffic pattern, as well as number of iterations and warmup iterations:
```yaml
iters: 50
warmup_iters: 10
traffic_pattern:
  matrix_file: "/path/to/matrix.txt"
  shards: 1
  mem_type: "cuda"
  xfer_op: "WRITE"
```

Then run it with

```bash
python tools/perf/main.py  ct-perftest path/to/config.yaml
```