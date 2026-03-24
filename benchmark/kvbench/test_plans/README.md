# KVBench Test Plans

This directory contains SLURM sbatch scripts, YAML configs, and test
artifacts organized by cluster and test phase.

**Note**: These files are not tracked in git. They are cluster-specific
and contain hardcoded paths. Use them as templates for new clusters.

---

## Directory Structure

### `scripts/`
Early test scripts developed on the GAIA cluster (DGX H100). Basic RDMA,
storage, and scaling tests. These were the first functional tests used to
validate the kvbench framework.

Key files:
- `test_simple_2rank.sbatch` - Minimal 2-rank RDMA validation
- `test_simple_8rank.sbatch` - 8-rank single-node RDMA
- `test_storage_8rank.sbatch` - 8-rank storage with POSIX
- `test_storage_8rank_gds.sbatch` - 8-rank storage with GDS
- `comprehensive_test.sbatch` - Full test suite (all backends, sizes)

### `isr1_all_test/`
Full test suite for the ISR1 cluster. Includes RDMA, storage, and
combined tests. Uses enroot containers.

Key files:
- `run_storage.sbatch` - Storage-only test
- `run_rdma_storage.sbatch` - Combined RDMA + storage
- `run_12nodes.sbatch` - 12-node scaling test
- `run_2nodes_8gpu.sbatch` - 2-node 8-GPU test
- `run_roce_rail.sbatch` - RoCE rail-optimized test

Configs:
- `config.yaml` - Basic RDMA config
- `config_storage_only.yaml` - Storage-only config
- `config_rdma_storage.yaml` - Combined config

### `isr1_pre_test/` (Primary - Most Recent)
ISR1-PRE cluster tests with VAST storage. This is where the main
benchmarking work was done.

#### Storage-Only (KV Cache Simulation)
- `run_8nodes_storage_tp8_264tp_vast.sbatch` - 8 nodes, TP=8, 264 patterns, GDS
- `run_8nodes_storage_tp4_264tp_vast.sbatch` - 8 nodes, TP=4, 264 patterns, GDS
- `run_8nodes_storage_tp8_512tp_vast.sbatch` - 8 nodes, TP=8, 512 patterns, GDS
- `run_8nodes_storage_tp4_512tp_vast.sbatch` - 8 nodes, TP=4, 512 patterns, GDS
- `run_12nodes_tp8_264tp_vast.sbatch` - 12 nodes, TP=8, 264 patterns
- `run_12nodes_tp4_264tp_vast.sbatch` - 12 nodes, TP=4, 264 patterns

#### POSIX Backend Variants
- `run_storage_posix_small.sbatch` - Small I/O sizes
- `run_storage_posix_large.sbatch` - Large I/O (64M-1G) with O_DIRECT
- `run_storage_posix_large_buffered.sbatch` - Large I/O without O_DIRECT
- `run_storage_posix_optimized.sbatch` - Optimized (block splitting + uring)
- `run_storage_posix_1node_test.sbatch` - Quick 1-node validation

#### GDS Backend
- `run_storage_gds.sbatch` - GDS on VAST
- `run_1node_gds_test.sbatch` - 1-node GDS validation
- `run_1node_gds_minimal.sbatch` - Minimal GDS debug test
- `run_4nodes_decode_gds.sbatch` - 4-node decode simulation

#### Combined and RDMA
- `run_rdma.sbatch` - RDMA-only baseline
- `run_rdma_storage.sbatch` - Combined RDMA + storage
- `run_12nodes_rdma_storage.sbatch` - 12-node combined
- `run_12nodes_decode_posix.sbatch` - 12-node decode with POSIX

#### Saturation and Baseline
- `run_sat_v2.sbatch` - Parameterized saturation test
- `run_8nodes_fio_simple.sbatch` - FIO baseline
- `run_8nodes_fio_read_vast.sbatch` - FIO read benchmark
- `run_8nodes_fio_isolated_agg_vast.sbatch` - FIO isolated test

#### Supporting Files
- `bind_8gpus.sh` - GPU-NIC affinity mapping (ISR1-PRE specific)
- `config_rdma.yaml` - RDMA transfer config
- `config_storage.yaml` - Basic storage config
- `config_storage_large.yaml` - Large I/O storage config
- `config_rdma_storage.yaml` - Combined config

### `spcx_test/`
Storage I/O optimization investigation scripts. Used during the
performance analysis that brought kvbench from ~10 GB/s to ~45 GB/s.

Key files:
- `run_nixlbench_baseline.sbatch` - nixlbench reference (48 GB/s)
- `run_mini_bench.sbatch` - Storage microbenchmark
- `run_diagnose.sbatch` - I/O diagnostics
- `run_vast_scaling.sbatch` - VAST scaling test
- `run_all_saturation.sbatch` - Saturation sweep

### `notification_test/`
Tests for the NIXL notification/signaling mechanism.

### `pre09_to_pre10/`
Migration tests when moving between cluster configurations.

### `isr1_rdma/`
Focused RDMA tests on ISR1 cluster.

---

## Adapting for a New Cluster

When setting up on a new cluster, modify these in the sbatch files:

1. **SLURM partition/reservation**: `#SBATCH --partition=...`
2. **Node list**: `#SBATCH --nodelist=...`
3. **Container path**: Update the `.sqsh` container path
4. **Storage path**: Change `/mnt/vast/...` to your storage mount
5. **GPU-NIC binding**: Create a new `bind_Ngpus.sh` for your topology
6. **UCX settings**: Adjust `UCX_TLS`, `UCX_IB_GID_INDEX` for your fabric

---

## Documentation Files

| File | Content |
|------|---------|
| `RESULTS_SUMMARY.md` | GAIA cluster performance results (Dec 2025) |
| `TEST_LOG.md` | Detailed test execution log with all findings |
| `STORAGE_SUMMARY.md` | Storage backend test summary |
| `GPUDIRECT_RDMA_TECHNICAL.md` | GPU Direct RDMA technical investigation |
| `storage_perf_testplan.md` | Original test plan for storage performance |
