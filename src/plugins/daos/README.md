# NIXL DAOS backend

This plugin transfers data between host memory and files in a DAOS POSIX
container by calling `libdfs` directly. It does not require dfuse, interception,
or `LD_PRELOAD`.

The backend follows NIXL object-storage semantics:

- local descriptors use `DRAM_SEG`;
- storage descriptors use `OBJ_SEG`;
- `OBJ_SEG.metaInfo` is the path relative to the root of the DFS container;
- `OBJ_SEG.addr` is the byte offset in that file;
- an empty `metaInfo` uses the decimal `devId` as the path, matching the NIXL S3
  backend's key fallback.

## Dependencies and build

Install the DAOS client development package so `daos_fs.h`,
`daos_obj_class.h`, `libdfs.so`, and `libdaos.so` are available, then configure
NIXL with the plugin enabled:

```bash
meson setup build -Denable_plugins=DAOS
meson compile -C build
```

If DAOS is installed under a nonstandard prefix, set `PKG_CONFIG_PATH`,
`CPLUS_INCLUDE_PATH`, and `LIBRARY_PATH` as appropriate before running Meson.
The resulting dynamic plugin is `libplugin_DAOS.so`.

## Backend parameters

| Parameter | Default | Description |
|---|---:|---|
| `pool` | none | Required DAOS pool label or UUID |
| `container` | none | Required POSIX container label or UUID |
| `system` | empty | Optional DAOS system name; empty uses the default |
| `read_only` | `false` | Connect with `O_RDONLY` and reject writes |
| `create_container` | `false` | Add `O_CREAT` to `dfs_connect` |
| `chunk_size` | `0` | Chunk size used when a write creates a file; zero uses the DFS default |
| `object_class` | empty | Symbolic DAOS object class for new files, for example `RP_2G1` or `EC_4P2G1` |
| `object_class_hint` | empty | Topology-aware DFS class hint for new files, for example `file:single` or `file:max` |
| `oclass_id` | `0` | Legacy numeric object-class ID for new files; zero uses the DFS default |
| `num_event_queues` | `1` | Number of EQ lanes; each creates a DAOS network context and owner thread |
| `max_inflight_per_queue` | `1024` | Maximum queued plus active operations admitted to each lane |
| `submission_batch_size` | `32` | Maximum submissions processed before a lane polls for completions |
| `completion_batch_size` | `128` | Maximum completed events returned by one EQ poll |
| `progress_poll_timeout_us` | `1000` | Maximum blocking interval for each event-queue progress poll |
| `progress_cpu_affinity` | empty | Linux CPU IDs, one per lane, for example `4,5`; empty disables affinity |

Affinity is applied when the backend is created. Recreate the backend with a
different list when changing CPU placement between experimental runs.

Set at most one of `object_class`, `object_class_hint`, and a nonzero
`oclass_id`. Symbolic classes are resolved with `daos_oclass_name2id`; hints are
resolved after connecting with `dfs_suggest_oclass`, allowing DAOS to choose a
class for the container topology and redundancy factor. These settings and
`chunk_size` apply only when a file is created and do not relayout an existing
file. Regular DFS files remain `DAOS_OT_ARRAY_BYTE` objects; libdfs selects that
object type internally.

Parent directories must exist. Writes create the final file if it is absent and
support arbitrary offsets. Reads must return the full requested range; a short
read completes with `NIXL_ERR_BACKEND`.

## C++ usage sketch

```cpp
nixl_b_params_t params = {
    {"pool", "my-pool"},
    {"container", "kv-cache"},
    {"object_class", "EC_4P2G1"},
    {"chunk_size", "4194304"},
    {"num_event_queues", "2"},
    {"max_inflight_per_queue", "1024"},
    {"progress_cpu_affinity", "4,5"},
};

nixlBackendH *backend = nullptr;
auto status = agent.createBackend("DAOS", params, backend);

// Register an OBJ_SEG descriptor with metaInfo="models/layer-0/k.bin".
// The descriptor's addr is the file offset and len is the registered range.
```

## Initial implementation limits

- Host memory only; there is no direct GPU-memory path yet.
- `prepXfer` performs the blocking path lookup/open and retains each DFS object.
  `postXfer` distributes operations round-robin to bounded submission queues.
- Each lane owns one DAOS event queue, its network context, and a fixed progress
  thread. The thread alternates bounded submission batches with completion
  polling, preventing sustained submission load from starving DAOS progress.
- Every in-flight operation retains its event, I/O vector, scatter/gather list,
  read byte count, DFS object, completion callback, and caller buffer address
  until the event is polled and finalized.
- DAOS event abort does not currently cancel internal operations. `releaseReqH`
  therefore returns `NIXL_ERR_NOT_ALLOWED` until every event completes.
- If a lane's `max_inflight_per_queue` is exhausted, that descriptor completes
  with `NIXL_ERR_BACKEND`; already admitted descriptors are drained normally.
- CPU affinity is experimental and non-fatal: an invalid OS affinity operation
  is logged and that lane continues unbound. Configure exactly one CPU ID per
  event queue. The parameter is accepted but cannot bind threads on non-Linux
  platforms.
- Directory creation and file deletion are intentionally outside the transfer
  API in this first version.
