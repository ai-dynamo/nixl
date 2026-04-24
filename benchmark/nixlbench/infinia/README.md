# INFINIA Backend for nixlbench

This directory contains all INFINIA-specific configuration files and documentation for using the INFINIA backend with nixlbench.

## Contents

| File | Description |
|------|-------------|
| `README.md` | This file - overview and configuration guide |
| `infinia_example.conf` | Example INFINIA plugin configuration |
| `nixlbench_infinia.toml` | Complete nixlbench configuration example |

## Configuration Method

The INFINIA plugin uses a **simple key=value config file** passed via the `--infinia_config_file` parameter.
This keeps INFINIA-specific settings separate from nixlbench settings.

**No external dependencies required** - the plugin uses a built-in parser for the simple config format.

## Quick Start

### Step 1: Create INFINIA plugin config file

Create `infinia.conf`:

```ini
# INFINIA configuration
cluster=my_cluster
tenant=my_tenant
dataset=my_dataset
sthreads=8
num_buffers=512
num_ring_entries=512
```

### Step 2: Create nixlbench config file

Create `nixlbench.toml`:

```toml
backend = "INFINIA"
infinia_config_file = "infinia/infinia.conf"
initiator_seg_type = "DRAM"
target_seg_type = "DRAM"
total_buffer_size = 67108864
num_iter = 16
```

### Step 3: Run nixlbench

```bash
# From the nixlbench build directory
./nixlbench --config_file nixlbench.toml

# Or use the provided example
./nixlbench --config_file ../infinia/nixlbench_infinia.toml
```

## Alternative: Command-Line Only

```bash
# From the nixlbench build directory
./nixlbench \
  --backend INFINIA \
  --infinia_config_file ../infinia/infinia_example.conf \
  --initiator_seg_type DRAM \
  --target_seg_type DRAM \
  --num_iter 16
```

## Config File Format

The INFINIA config file uses a simple **key=value** format:

- One parameter per line: `key=value`
- Comments start with `#`
- Blank lines are ignored
- Whitespace around `=` is trimmed
- No sections or brackets needed

**Example:**
```ini
# Connection settings
cluster=production
tenant=team_a
dataset=benchmark_data

# Performance tuning
sthreads=4

# BatchTask configuration
max_retries=3
```

## INFINIA Configuration Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `cluster` | Infinia cluster name | `"production"` |
| `tenant` | Tenant name | `"team_a"` |
| `dataset` | Dataset name | `"benchmark_data"` |

### Optional Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `subtenant` | Subtenant | `"red"` | `"sub1"` |
| `sthreads` | Number of service threads (limited by CPU cores) | `8` | `16` |
| `num_buffers` | Pre-allocated deferred operation buffers for async ops | `512` | `1024` |
| `num_ring_entries` | Depth of the asynchronous I/O ring buffer | `512` | `1024` |
| `coremasks` | CPU affinity: hex ("0x0F"), list ("[0-3,8]"), or empty disables | `""` | `"0-7"` |
| `max_retries` | BatchTask retry limit | `3` | `5` |

## Example Configurations

### Small Transfers
```ini
cluster=test
tenant=dev
dataset=small_test
sthreads=4
num_buffers=128
num_ring_entries=128
max_retries=3
```

### Large Transfers
```ini
cluster=prod
tenant=production
dataset=large_test
sthreads=16
num_buffers=256
num_ring_entries=256
coremasks=
max_retries=5
```

### GPU Direct
```ini
cluster=gpu_cluster
tenant=ml_team
dataset=gpu_data
sthreads=8
num_buffers=512
num_ring_entries=512
coremasks=0-15
max_retries=3
```

## How It Works

1. **nixlbench** reads `--infinia_config_file` parameter
2. Passes the config file path to the INFINIA plugin via `backend_params["config_file"]`
3. **INFINIA plugin** parses the simple key=value file in its constructor using a built-in parser
4. No external dependencies (TOML, YAML, JSON libraries) required

This keeps INFINIA configuration completely separate from nixlbench configuration.

## See Also

- Example configs in `benchmark/nixlbench/infinia/`
- INFINIA plugin README: `src/plugins/infinia/README.md`
- Main nixlbench README: `benchmark/nixlbench/README.md`

