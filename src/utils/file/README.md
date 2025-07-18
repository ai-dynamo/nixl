# File Utils and QueryMem API Implementation

This directory contains the implementation of file utilities for NIXL file backends and the QueryMem API.

## Overview

The implementation provides:

1. **QueryMem API**: Implementation of the QueryMem API for file backends that uses `nixl_reg_dlist_t` with filenames in `metaInfo` and uses `stat` to check file existence.

2. **File Utils (`file_utils.h` and `file_utils.cpp`)**: Utility functions for file operations including `prefixedOpen` and file query functions.

## QueryMem API Implementation

The QueryMem API has been implemented for all file backends:

- **POSIX Backend** (`src/plugins/posix/`)
- **HF3FS Backend** (`src/plugins/hf3fs/`)
- **GDS MT Backend** (`src/plugins/gds_mt/`)
- **CUDA GDS Backend** (`src/plugins/cuda_gds/`)

### How it works:

1. **Input**: Takes a `nixl_reg_dlist_t` containing file descriptors with filenames in the `metaInfo` field
2. **Processing**:
   - Uses `extractMetadata()` method from `nixlDescList` class to extract filenames from descriptors
   - Calls `queryFileInfoList()` to check file existence using `stat`
   - Strips any prefixes (RO:, RW:, WR:) before checking file existence
3. **Output**: Returns a vector of `nixl_query_resp_t` structures containing:
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
  - `filename`: The filename to query (can be prefixed)
  - `resp`: Output response structure
- **Returns**: NIXL_SUCCESS on success, error code otherwise
- **Prefix Support**: Handles prefixed filenames by stripping prefixes before checking file existence

### `queryFileInfoList`
- **Purpose**: Query file information for multiple files
- **Parameters**:
  - `filenames`: Vector of filenames to query (can be prefixed)
  - `resp`: Output response vector
- **Returns**: NIXL_SUCCESS on success, error code otherwise
- **Prefix Support**: Handles prefixed filenames by stripping prefixes before checking file existence

### `prefixedOpen`
- **Purpose**: Opens a file with prefix handling
- **Parameters**:
  - `prefix`: The prefix containing the file path (e.g., "RO:/path/to/file", "RW:/path/to/file")
  - `fd`: Pointer to file descriptor (output for open)
  - `was_opened`: Output parameter indicating if a file was opened (true) or if devId was used as fd (false)
- **Returns**: 0 on success, -1 on failure
- **Prefix Support**:
  - `RO:` - Uses O_RDONLY (read-only) with mode 0444
  - `RW:` - Uses O_RDWR (read-write) with mode 0644
  - `WR:` - Uses O_WRONLY (write-only) with mode 0222
  - No prefix - Returns 0 without opening a file

### Note on prefixedOpen:

The `prefixedOpen` function is available in `file_utils` but is **not currently used by the plugins**. The plugins currently use `devId` directly as the file descriptor. The `prefixedOpen` function is kept available for future use when prefixed filename support is needed in the plugins.

### Prefix Examples:

```cpp
// The prefix parameter contains the entire string including the file path
// The prefix type is parsed from the beginning of the string

// File opening prefixes:
"RO:/path/to/file.txt"  // Read-only prefix
"RW:/path/to/file.txt"  // Read-write prefix
"WR:/path/to/file.txt"  // Write-only prefix
"/path/to/file.txt"     // No prefix

// For prefixedOpen, the prefix determines the open flags and mode:
int fd1 = -1, fd2 = -1, fd3 = -1;
bool was_opened1, was_opened2, was_opened3;
prefixedOpen("RO:/path/to/file.txt", &fd1, &was_opened1); // Uses O_RDONLY and mode 0444
prefixedOpen("RW:/path/to/file.txt", &fd2, &was_opened2); // Uses O_RDWR and mode 0644
prefixedOpen("WR:/path/to/file.txt", &fd3, &was_opened3); // Uses O_WRONLY and mode 0222
```

## Architecture

The current architecture separates concerns:

1. **Descriptor Operations**: The `nixlDescList` class provides `extractMetadata()` method to extract metadata from descriptors
2. **File Operations**: The `file_utils` provides generic file query functions (`queryFileInfo`, `queryFileInfoList`) and `prefixedOpen` for future use
3. **Plugin Integration**: Each plugin directly uses `extractMetadata()` and `queryFileInfoList()` without intermediate layers
4. **File Descriptor Management**: Plugins currently use `devId` directly as file descriptor, with `prefixedOpen` available for future prefixed filename support

This approach eliminates the need for the `file_query_helper` layer and provides better separation of concerns.

## Building

The file utils are built as a shared library (`libfile_utils.so`) and linked with all file backends. The build system has been updated to include the file utils dependency in all relevant backend meson.build files.



## Testing

Test files are provided:
- `test/unit/utils/file/test_file_utils.cpp`: Tests the file utils functions including prefix handling
- `test/python/test_prefixed_open.py`: Python tests for prefixedOpen functionality
- `test/python/test_query_mem.py`: Python tests for QueryMem API functionality

## Dependencies

- Standard C++ libraries
- POSIX system calls (`stat`, `open`, `close`)
- NIXL common library for logging
