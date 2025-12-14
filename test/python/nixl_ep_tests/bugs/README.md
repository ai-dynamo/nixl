# Bug Reproduction Test Suite

This directory contains tests that reproduce known bugs in NIXL EP and related libraries (NIXL, UCX).
These tests help developers reproduce, debug, and verify fixes for known issues.

---

## Quick Reference

| Bug ID | Component | Severity | Status | Test File |
|--------|-----------|----------|--------|-----------|
| BUG-01 | NIXL C++ | 🔴 Critical | Open | `test_bug_01_segfault.py` |
| BUG-02 | UCX | 🔴 Critical | Open | `test_bug_02_rcache.py` |
| BUG-03 | UCX | 🟡 Low | Open | `test_bug_03_gdr_copy.py` |
| BUG-04 | NIXL | 🟡 Low | Open | `test_bug_04_invalidate.py` |
| BUG-05 | NIXL EP/UCX/etcd | 🟠 Medium | Open | `test_bug_05_connect_cuda_error.py` |

---

## Running Bug Tests

### Run All Bug Tests
```bash
cd /workspace/nixl/test/python/nixl_ep_tests
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh
python3 -m pytest bugs/ -v --timeout=300
```

### Run Individual Tests
```bash
# BUG-01: Segfault (single-process, will crash if bug present)
python3 bugs/test_bug_01_segfault.py

# BUG-02: UCX rcache (use 16 experts for ~30% crash rate)
python3 bugs/test_bug_02_rcache.py --num-processes=8 --experts=16 --runs=3 --timeout=300

# BUG-03: gdr_copy warning (run back-to-back to trigger)
for i in {1..3}; do
  python3 bugs/test_bug_03_gdr_copy.py --num-processes=8
done 2>&1 | tee /tmp/bug03.log
echo "gdr_copy warnings: $(grep -c 'UCX.*WARN.*rcache gdr_copy' /tmp/bug03.log)"

# BUG-04: invalidateRemoteMD (55+ warnings typical)
python3 bugs/test_bug_04_invalidate.py --num-processes=8 2>&1 | tee /tmp/bug04.log
echo "invalidateRemoteMD warnings: $(grep -c 'invalidateRemoteMD.*NIXL_ERR_NOT_FOUND' /tmp/bug04.log)"
```

---

## Bug Details

### BUG-01: NIXL Segfault on Repeated Buffer Creation

**Severity**: 🔴 Critical
**Component**: NIXL library (C++)
**File**: `test_bug_01_segfault.py`
**Status**: Open - needs NIXL team fix

#### Description
Creating a second `nixl_ep.Buffer` instance after calling `destroy()` on the first one causes a segmentation fault. This prevents running multiple warmup/measurement rounds in performance tests.

#### Reproduction Steps
1. Create a `nixl_ep.Buffer` with `explicitly_destroy=True`
2. Call `destroy()` on the buffer
3. Create a new `nixl_ep.Buffer` in the same process
4. **Result**: Segfault occurs during second buffer initialization

#### Error Output
```
Round 1: Creating buffer...
Round 1: Buffer created, destroying...
Round 2: Creating buffer...
Segmentation fault (core dumped)
```

#### Root Cause Analysis
The NIXL C++ library appears to have global state that is not properly reset after `destroy()`.

**Tested**: Bug affects ALL nvlink backends (`nixl`, `ipc`, `none`) - confirmed Dec 2025.
This rules out backend-specific causes and points to core NIXL library issue.

**Investigated** (Dec 2025): We attempted fixes that DID NOT work:
1. Adding `deregisterMem()` for all registered buffers before `cudaFree()`
2. Resetting `nixl_agent_info` (unique_ptr to NIXL agent)
3. Resetting `nixl_ctx` (unique_ptr to EP context)

The issue is deeper than memory registration - likely UCX/CUDA internal state.

Possible remaining causes:
- UCX context global state not fully cleaned up
- CUDA driver-level state persisting
- Static/global variables in UCX plugins
- etcd client connection state

#### Workaround
Use `--warmup=0 --rounds=1` in performance tests to ensure only one buffer is created per process.

#### Impact
- Cannot run proper warmup rounds in benchmarks
- Each test requires spawning a fresh process
- Performance measurements may be less accurate due to lack of warmup

---

### BUG-02: UCX rcache Assertion Failure

**Severity**: 🔴 Critical
**Component**: UCX library
**File**: `test_bug_02_rcache.py`
**Status**: Open - needs UCX team investigation

#### Description
The UCX library's region cache (rcache) occasionally fails an assertion during buffer cleanup, causing the process to abort. This appears to be a reference counting issue in UCX's memory registration cache.

#### Reproduction Steps
1. Create buffer with 16 experts per rank
2. Connect all 8 ranks
3. Disconnect all ranks
4. Reconnect all ranks
5. Call `destroy()`
6. **Result**: ~30% chance of assertion failure

#### Error Output
```
[pool0-00290:2841279:0:2841279] rcache.c:477 Assertion `region->refcount > 0' failed
```

#### Root Cause Analysis
The UCX rcache maintains reference counts for registered memory regions. During cleanup:
- Multiple regions may be deregistered simultaneously
- Race condition between deregistration and reference count decrement
- 16 experts creates more memory regions than 8 or 32, possibly hitting a specific code path

#### Interesting Finding
The bug is **most reproducible with exactly 16 experts**:
- 2, 4, 8 experts: Rarely fails
- 16 experts: ~30% failure rate
- 32 experts: Occasionally fails

This suggests a specific memory layout or region count triggers the bug.

#### Workaround
- Skip 16-expert tests
- Retry failed tests (usually succeeds on retry)
- Use `--experts=2,4,8,32` to skip 16

#### Impact
- Intermittent test failures
- CI/CD flakiness
- Cannot reliably benchmark 16-expert configurations

---

### BUG-03: UCX rcache gdr_copy Warning

**Severity**: 🟡 Low (cosmetic)
**Component**: UCX library
**File**: `test_bug_03_gdr_copy.py`
**Status**: Open - needs investigation

#### Description
UCX emits a warning about memory regions remaining in the LRU list after operations complete. This indicates leftover memory registration entries but does not affect functionality.

#### Reproduction Steps
1. Run any full cycle test (init → connect → disconnect → destroy)
2. Check stderr for warning message

#### Error Output
```
[1765125844.886116] [pool0-00290:2842561:0] rcache.c:1392 UCX WARN rcache gdr_copy: 1 regions remained on lru list, first region: 0xad4e380
```

#### Root Cause Analysis
GDR (GPU Direct RDMA) copy operations register memory regions for efficient GPU-to-NIC transfers. These warnings indicate:
- Memory regions were not explicitly deregistered
- UCX's lazy cleanup left entries in the LRU cache
- Entries will be cleaned up eventually but weren't at warning time

#### Observations
- Appears in ~30% of test runs
- More common with higher expert counts
- Does not correlate with test failures
- Numbers vary (1-4 regions typically)
- **Key finding**: More likely when running tests back-to-back (cumulative UCX cache state)

#### Workaround
None needed - this is a cosmetic warning that doesn't affect correctness.

#### Impact
- Clutters test output
- May indicate minor memory inefficiency
- No functional impact

---

### BUG-04: NIXL invalidateRemoteMD Warning

**Severity**: 🟡 Low (cosmetic)
**Component**: NIXL library
**File**: `test_bug_04_invalidate.py`
**Status**: Open - NIXL/NIXL_EP issue

#### Description
When multiple ranks call `destroy()` simultaneously, NIXL emits warnings about failing to invalidate remote metadata. This is a race condition in the distributed cleanup protocol.

#### Reproduction Steps
1. Create buffers on 8 ranks
2. Connect all ranks to each other
3. All ranks call `destroy()` simultaneously (no barrier)
4. **Result**: Multiple ranks emit warnings

#### Error Output
```
nixl_agent.cpp:1700] invalidateRemoteMD: error invalidating remote metadata for agent '5' with status NIXL_ERR_NOT_FOUND
nixl_agent.cpp:1700] invalidateRemoteMD: error invalidating remote metadata for agent '3' with status NIXL_ERR_NOT_FOUND
```

#### Root Cause Analysis
When rank A tries to invalidate metadata on rank B:
1. Rank A sends invalidation request
2. Meanwhile, rank B has already destroyed its metadata
3. Rank A receives `NIXL_ERR_NOT_FOUND`
4. Warning is logged

This is expected behavior in simultaneous shutdown - the warning is benign.

#### Observations
- More warnings with more ranks (8 ranks = more warnings than 2)
- `disconnect_ranks()` before `destroy()` increases warnings
- `reconnect` before `destroy()` reduces warnings slightly
- Warnings don't affect correctness - operations still complete
- **Typical count**: 55+ warnings with 8 ranks (basic scenario)

#### Mitigation
Adding a `reconnect` phase before `destroy()` can reduce (but not eliminate) these warnings:
```python
buffer.disconnect_ranks(other_ranks)
buffer.connect_ranks(other_ranks)  # Reconnect
buffer.destroy()  # Fewer warnings
```

#### Workaround
Ignore these warnings in test output validation.

#### Impact
- Clutters test output
- May confuse users into thinking something is wrong
- No functional impact

---

### BUG-05: Intermittent CUDA Error During connect_ranks()

**Severity**: 🟠 Medium
**Component**: NIXL EP / UCX / etcd (unclear)
**File**: `test_bug_05_connect_cuda_error.py`
**Status**: Open - needs investigation

#### Description
Intermittent "CUDA error: invalid argument" failures during `connect_ranks()` when running tests back-to-back. Different ranks fail each time; re-running usually succeeds.

#### Reproduction Steps
1. Run `test_connect_ranks` multiple times in sequence (without etcd reset)
2. Error appears in ~10-20% of runs
3. Different ranks fail each time (not deterministic)

#### Error Output
```
RuntimeError: Failed: CUDA error ../examples/device/ep/csrc/nixl_ep.cpp:861 'invalid argument'
```

Often preceded by:
```
[1765442432.425215] rcache.c:1392 UCX WARN rcache gdr_copy: 2 regions remained on lru list
```

#### Root Cause Analysis
The exact cause is unclear. Possible factors:
1. UCX rcache state pollution from previous runs
2. etcd metadata exchange timing issues
3. GPU context state not fully cleaned up

#### Workaround
- Reset etcd before test runs: `source ./scripts/reset_etcd.sh`
- Add retry logic to tests
- Add delay between test runs

#### Impact
- CI/CD flakiness (~10-20% failure rate)
- Requires test retries
- Not a correctness bug (retry succeeds)

---

## Test Environment

These bugs were discovered and documented on:
- **Cluster**: DFW cluster
- **Nodes**: pool0-00290 (and similar)
- **GPUs**: 8x H100 per node (NVLink NV18)
- **NIXL version**: Current main branch
- **UCX version**: System default in container
- **Date**: December 2025

---

## For Developers

### Adding a New Bug Test

1. Create `test_bug_XX_description.py`
2. Add detailed docstring with:
   - Bug description
   - Reproduction steps
   - Expected behavior before/after fix
3. Add entry to `README.md`
4. Add to `__init__.py`

### Verifying a Fix

```bash
# Run the specific bug test
python3 bugs/test_bug_XX_description.py --num-processes=8

# For intermittent bugs, run multiple times
for i in {1..10}; do
    python3 bugs/test_bug_XX_description.py --num-processes=8 || echo "FAILED on run $i"
done
```

### CI/CD Integration

Bug tests can be marked as expected failures until fixed:
```python
@pytest.mark.xfail(reason="BUG-01: Segfault on repeated buffer creation")
def test_repeated_buffer_creation():
    ...
```

---

## Contact

For questions about these bug reproductions:
- **Test author**: AI assistant / NIXL EP test team
- **NIXL issues**: File in NIXL repository
- **UCX issues**: File in OpenUCX repository

