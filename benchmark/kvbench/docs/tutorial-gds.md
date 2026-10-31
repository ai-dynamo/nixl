# NIXL KVCache GDS Profiling Tutorial

KVBench profiles transfers between GPU memory and a local file using NIXLBench's
GDS or GDS_MT backend. It uses model metadata to size the transfers; model weights
are not loaded.

## Prerequisites

- NIXL with the GDS/GDS_MT plugins and a CUDA-enabled `nixlbench` on `PATH`
- The KVBench Python dependencies installed (see the KVBench README)
- A CUDA GPU, a working cuFile installation, and a writable storage directory

Run the commands below from `benchmark/kvbench`. A single-worker storage run
uses the null runtime by default and does not require an ETCD server.

## GPU-to-file WRITE

Use an empty directory dedicated to this run:

```bash
write_dir=$(mktemp -d /tmp/kvbench-gds-write.XXXXXX)
python main.py profile \
  --model ./examples/model_llama_3_1_8b.yaml \
  --model_config ./examples/block-tp1-pp1.yaml \
  --backend GDS \
  --source gpu \
  --isl 16 --page_size 16 --num_requests 1 \
  --num_iter 16 --warmup_iter 4 \
  --check_consistency \
  --filepath "$write_dir"
```

This writes from a VRAM buffer with one 1 MiB transfer per batch. Check that the
NIXLBench configuration reports `Initiator seg type: VRAM`, `Op type: WRITE`,
and consistency checking enabled. The result table reports bandwidth and timing.

## File-to-GPU READ

READ initializes a missing input file itself. Use a separate empty directory:

```bash
read_dir=$(mktemp -d /tmp/kvbench-gds-read.XXXXXX)
python main.py profile \
  --model ./examples/model_llama_3_1_8b.yaml \
  --model_config ./examples/block-tp1-pp1.yaml \
  --backend GDS \
  --source file \
  --isl 16 --page_size 16 --num_requests 1 \
  --num_iter 16 --warmup_iter 4 \
  --check_consistency \
  --filepath "$read_dir"
```

Check that the configuration reports `Initiator seg type: VRAM` and
`Op type: READ`. Replace `--backend GDS` with `--backend GDS_MT` to exercise the
per-request engine. Use a new directory for each run.

The consistency check expects the benchmark's own fill pattern. Reusing a file
from a WRITE run for a consistency-checked READ can fail because WRITE changes
that pattern. These examples test the two directions independently, rather than
a persistent-data round trip. Remove the generated directories after inspecting
the results.

## Configuration semantics

For GDS and GDS_MT, `--source gpu` selects WRITE and `--source file` selects READ.
Both use `initiator_seg_type=VRAM`. These high-level choices override explicit
`--op_type`, `--initiator_seg_type`, and `--target_seg_type` settings; use
`nixlbench` directly to benchmark a DRAM buffer with GDS.

The remote file descriptor is selected by the native storage backend.
`target_seg_type=DRAM` is a parseable placeholder, not the file's memory type.
The GDS configurator does not use `--destination`.

This corrects earlier KVBench behavior: file-source runs could use DRAM, while
GPU-source runs selected an invalid FILE memory parameter. Results from those
older configurations are not comparable to GPU-buffer profiling.

Block size, batch size, and total buffer size are derived from the model/config
and override the corresponding low-level flags. To preview the effective values,
replace `profile` with `plan --format json`. For the example above, increasing
`--isl` to 64 produces four 1 MiB transfers per batch.

## cuFile mode and troubleshooting

cuFile may use native GDS or compatibility mode depending on GPU, filesystem,
and driver support. On consumer GPUs such as GeForce, use compatibility mode.
A compatibility-mode run verifies GPU-buffer/file functionality; its bandwidth
is not evidence of native storage-to-GPU DMA.

To explicitly force compatibility mode, use a per-process cuFile JSON config
with `properties.allow_compat_mode` and `properties.force_compat_mode` set to
`true`, and point `CUFILE_ENV_PATH_JSON` at it. Check the cuFile logs for the
selected mode. See the [cuFile configuration documentation](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html).

- A failed `nixlbench` run makes `kvbench profile` exit with a nonzero status;
  inspect the streamed native diagnostics.
- For file creation errors, verify the directory exists and is writable.
- For cuFile errors, verify the plugin, CUDA/cuFile library paths and mode.
- For consistency errors, retry with an empty directory and inspect the printed
  operation, VRAM segment and workload sizes.
