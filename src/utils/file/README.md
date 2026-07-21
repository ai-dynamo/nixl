# File Utils, QueryMem API, and Path-Mode Registration

This directory contains shared C++ utilities for NIXL file-aware backends:

- `file_utils.{h,cpp}`: `nixl::queryFileInfo()` / `queryFileInfoList()` helpers
  for the QueryMem API (file existence + stat).
- `file_path_mode.{h,cpp}`: `nixl::parsePathMeta()` parser and `nixlFilePathMD`
  owned-fd RAII base for path-mode FILE_SEG registration; see
  [Path-Mode File Registration](#path-mode-file-registration).

All file-aware plugins (POSIX, HF3FS, CUDA_GDS, GDS_MT) link
`file_utils_interface` and consume both sets of helpers.

## QueryMem API Implementation through queryFileInfoList

The QueryMem API has been implemented for these file backends:

- **POSIX Backend** (`src/plugins/posix/`)
- **HF3FS Backend** (`src/plugins/hf3fs/`)
- **GDS MT Backend** (`src/plugins/gds_mt/`)
- **CUDA GDS Backend** (`src/plugins/cuda_gds/`)

The backend extracts the filenames from the input descriptors (`nixl_reg_dlist_t`) and passes them to queryFileInfoList.
Then queryFileInfoList returns a vector of `nixl_query_resp_t` structures containing:
   - `accessible`: Boolean indicating if file exists
   - `info`: Additional file information (size, mode, mtime) if file exists

### Usage Example:

```cpp
// Create registration descriptor list with filenames in metaInfo
nixl_reg_dlist_t descs(FILE_SEG, false);
descs.addDesc(nixlBlobDesc(0, 0, 0, "/path/to/file1.txt"));
descs.addDesc(nixlBlobDesc(0, 0, 0, "/path/to/file2.txt"));
descs.addDesc(nixlBlobDesc(0, 0, 0, "/path/to/file3.txt"));

// Query file status using the plugin's queryMem method
std::vector<nixl_query_resp_t> resp;
nixl_status_t status = plugin->queryMem(descs, resp);

// Check results
for (const auto& result : resp) {
    if (result.accessible) {
        std::cout << "File exists, size: " << result.info["size"] << std::endl;
    } else {
        std::cout << "File does not exist" << std::endl;
    }
}
```

## File Utils Functions

### `queryFileInfo`
- **Purpose**: Query file information for a single file
- **Parameters**:
  - `filename`: The filename to query
  - `resp`: Output response structure
- **Returns**: NIXL_SUCCESS on success, error code otherwise

### `queryFileInfoList`
- **Purpose**: Query file information for multiple files
- **Parameters**:
  - `filenames`: Vector of filenames to query
  - `resp`: Output response vector
- **Returns**: NIXL_SUCCESS on success, error code otherwise

## Building

The file utils are built as a shared library (`libfile_utils.so`) and linked with all file backends. The build system has been updated to include the file utils dependency in all relevant backend meson.build files.


## Testing

Test files are provided:
- `test/unit/utils/file/test_file_utils.cpp`: Tests the file utils functions
- `test/python/test_query_mem.py`: Python tests for QueryMem API functionality

## Dependencies

- Standard C++ libraries
- POSIX system calls (`stat`, `open`, `close`)
- NIXL common library for logging

## Path-Mode File Registration

Path-mode lets a caller declare a `FILE_SEG` descriptor by path in
`nixlBlobDesc::metaInfo` instead of pre-opening an fd; the backend
opens in `registerMem` and closes in `deregisterMem`. Motivation:
collapse N Python `os.open()` GIL crossings into one.

A `metaInfo` string is parsed as path-mode iff it matches:

```text
metaInfo := <modes>:<path>     # path-mode
          | <anything else>    # fd-in-devId mode
modes    := <access>[,<flag>]*
access   := "ro"               # O_RDONLY
          | "rw"               # O_RDWR
flag     := "direct"           # | O_DIRECT
          | "sync"             # | O_SYNC
          | "noatime"          # | O_NOATIME
          | "create"           # | O_CREAT (mode 0644)
```

Examples: `ro:/var/cache/x.bin`, `rw,direct:/var/cache/x.bin`,
`rw,create:/var/cache/x.bin`. Unknown/missing tokens yield `nullopt`
(fail-loud); the design is strictly additive: any non-matching
`metaInfo` falls through to caller-owned fd in `devId`.

Backends consume the shared helpers `nixl::parsePathMeta()` +
`nixlFilePathMD` from `file_path_mode.{h,cpp}`. POSIX uses
`nixlFilePathMD` directly; HF3FS / CUDA_GDS / GDS_MT extend their
existing per-descriptor MD struct with `owned` (and close the fd in
`deregisterMem` after the backend-specific teardown). The GDS per-fd
caches key on the *opened* fd, so two path-mode registrations of the
same path yield two cuFile handles (no path-level dedup).
