# NIXLBench scenarios

Scenario commands are available when NIXLBench is configured with
`-Dbuild_raw_cli=true` and built with CLI11.

## Allocate-once storage

`scenario allocate-once` models a fixed file-backed dataset with changing
block-aligned transfer offsets. NIXLBench discovers compatible installed
plugins from their advertised memory types: the plugin must support `FILE_SEG`
and at least one of `DRAM_SEG` or `VRAM_SEG`. Adding another compatible storage
plugin therefore does not require a NIXLBench code change.

```bash
# Create the directory used by the managed-file examples
mkdir -p /tmp/nixlbench-data

# Discover the compatible plugins installed on this system
nixlbench scenario allocate-once --help

# Inspect the resolved plan without opening backing files or creating transfer resources
nixlbench scenario allocate-once posix \
  --path /tmp/nixlbench-data \
  --file-size 64GB \
  --block-size 64KB \
  --batch-size 16 \
  --threads 4 \
  --dry-run

# Reuse two registered files across four threads and changing random offsets
nixlbench scenario allocate-once gds \
  --path /tmp/nixlbench-data \
  --file-registration-mode path \
  --file-size 64GB \
  --block-size 64KB \
  --batch-size 16 \
  --threads 4 \
  --num-files 2 \
  --iterations 1000 \
  --offset-mode random \
  --seed 42
```

Scenario options may appear before or after the plugin subcommand. Automatic
memory selection prefers `VRAM_SEG` when the selected plugin advertises it and
otherwise uses `DRAM_SEG`; an explicit `--initiator-memory` request fails
instead of falling back.

NIXLBench-managed files use deterministic names under `--path`. Missing or
wrong-sized managed files are initialized in bounded chunks and retained after
the run; exact-sized files are reused unless `--check-consistency` requests a
known initial byte pattern. With `--filenames`, every file must already exist
and NIXLBench never creates, resizes, or deletes it. The default
`--file-registration-mode descriptor` makes NIXLBench open and pin each file
descriptor. The optional `path` mode passes the path to the selected `FILE_SEG`
backend, which owns open and close; any required managed-file initialization is
also issued through that backend before benchmark timing begins.

The scenario owns the open/register-once policy through an allocate-once worker
strategy built on the common NIXL worker facilities. Generic scenario dispatch
does not contain allocate-once branches. The common transfer loop invokes a
scenario-owned lifecycle object before creating each request and after releasing
it, so later scenarios can acquire and release per-request resources without
copying that loop. Common options, file options, plugin selection, metadata
parameters, resolved-plan fields, and execution adaptation are owned by the shared
scenario framework. A new scenario supplies only its distinct options,
validation, plan details, resource policy, and worker strategy, then adds one
entry to the scenario registry.

Each thread is assigned to a file round-robin and receives a disjoint file
partition. Every iteration creates and releases a transfer request. The
scenario-owned `--offset-mode` is `random` by default and samples unique
block-aligned locations inside the thread partition; `sequential` walks and
wraps that partition. The shared benchmark loop expands each thread's retained
working buffer into `--batch-size` block-sized descriptors before the scenario
assigns one offset to each descriptor. This is intentionally distinct from the
legacy `--randomize_location_mode=blockaligned` behavior, which only shuffles
the otherwise sequential IOVs in a batch. A nonzero `--seed` makes random
selection reproducible; zero or an omitted seed resolves to a generated nonzero
seed shown in the plan. Transfer working memory is therefore
`threads * batch-size * block-size`, independent of the file size. The shared
worker reports request preparation, post, transfer latency, and throughput.
`--check-consistency` is available for managed files and validates the last
completed transfer per thread after the timed interval.

Plugin initialization parameters remain opaque:
`--plugin-param KEY VALUE` accepts only keys advertised by the selected plugin
and forwards the value unchanged.
