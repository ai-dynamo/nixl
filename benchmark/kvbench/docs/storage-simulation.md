# Storage Simulation for KV Cache Offloading

Simulate storage I/O patterns in multi-rank distributed workloads.

## Table of Contents

- [YAML Format](#yaml-format)
- [Use Cases](#use-cases)
- [CLI Options](#cli-options)
- [File Layout](#file-layout)
- [Storage Backend API](#storage-backend-api)

---

## YAML Format

Storage requirements use array format where index = rank:

```yaml
traffic_patterns:
  # RDMA + Storage
  - matrix_file: matrices/matrix_0.txt
    mem_type: cpu
    sleep_before_launch_sec: 0.01
    storage:
      read:  [1572864, 1572864, 0, 0]  # Ranks 0,1 read 1.5MB
      write: [524288, 524288, 0, 0]    # Ranks 0,1 write 0.5MB

  # Storage only (no RDMA)
  - mem_type: cpu
    storage:
      read:  [1M, 1M, 1M, 1M, 1M, 1M, 1M, 1M]
      write: [1M, 1M, 1M, 1M, 1M, 1M, 1M, 1M]

  # RDMA only (no storage)
  - matrix_file: matrices/matrix_1.txt
    mem_type: cuda
```

### Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `matrix_file` or `matrix` | No | RDMA transfer matrix (omit for storage-only) |
| `mem_type` | Yes | Memory type: cuda, cpu, vram, dram |
| `sleep_before_launch_sec` | No | Compute simulation time (default: 0) |
| `storage.read` | No | Array of read sizes per rank (index = rank) |
| `storage.write` | No | Array of write sizes per rank (index = rank) |

---

## Use Cases

### 1. Simulate LLM Inference I/O (75% Cache Hit)

```yaml
traffic_patterns:
  - matrix_file: matrix.txt
    mem_type: cpu
    sleep_before_launch_sec: 0.005  # Reduced compute
    storage:
      read:  [1572864, 1572864]     # 75% - read from cache
      write: [524288, 524288]       # 25% - write new KV
```

### 2. Combined Storage + RDMA

```yaml
traffic_patterns:
  - matrix_file: matrix.txt
    mem_type: cuda
    storage:
      read:  [1M, 1M, 1M, 1M]
      write: [256K, 256K, 256K, 256K]
```

### 3. Storage-Only Benchmark (FIO-like)

Test storage without RDMA overhead:

```yaml
traffic_patterns:
  - mem_type: cpu
    storage:
      read:  [1G, 1G, 1G, 1G, 1G, 1G, 1G, 1G]  # 8 ranks, 1GB each
      write: [0, 0, 0, 0, 0, 0, 0, 0]
```

### 4. Write-Only (Cold Start)

```yaml
traffic_patterns:
  - mem_type: cpu
    storage:
      write: [2M, 2M, 2M, 2M]  # No read, full write
```

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--storage-backend` | POSIX | Backend: POSIX, GDS, GDS_MT |
| `--storage-path` | `<config_dir>/storage` | Base path for storage files |
| `--storage-direct-io` | Auto | Enable O_DIRECT (auto for GDS) |

---

## File Layout

```
<storage_path>/
├── tp_0/                # Traffic pattern 0
│   ├── rank_0.bin       # [0, read_size) = READ, [read_size, end) = WRITE
│   └── rank_1.bin
├── tp_1/                # Traffic pattern 1
│   └── ...
```

---

## Storage Backend API

```python
class StorageBackend(ABC):
    @abstractmethod
    def prepare(self, tp_idx: int, rank: int, read_size: int, write_size: int) -> StorageHandle:
        """Create file and pre-fill read region."""

    @abstractmethod
    def get_read_handle(self, handle: StorageHandle, buffer: Any) -> Any:
        """Get NIXL handle for reading."""

    @abstractmethod
    def get_write_handle(self, handle: StorageHandle, buffer: Any) -> Any:
        """Get NIXL handle for writing."""
```

### Implementations

| Backend | Description |
|---------|-------------|
| `FilesystemBackend` | POSIX or GDS file I/O |
| [Future] `RedisBackend` | Redis GET/SET |
| [Future] `S3Backend` | S3 GetObject/PutObject |
