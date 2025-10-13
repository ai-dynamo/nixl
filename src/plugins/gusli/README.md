# NIXL Gusli Plugin

This plugin utilizes `gusli_clnt.so` as an I/O backend for NIXL. gusli client communicates with Gusli server (different process on local machine) and via shared memory sends the content of the io.
Gusli server is integrated into user space block device storage provider, like SPDK, NVMeshUM etc.
IO completely bypasses the kernel.
Gusli client can work without Server, accessing local block devices/files, but this is inefficient as it uses standard kernel API's

## Usage Guide
1. Build and install [Gusli](https://github.com/nvidia/gusli).
2. Do it via: git clone git clone https://github.com/nvidia/gusli.git
3. cd gusli; `make all BUILD_RELEASE=1 BUILD_FOR_UNITEST=0 VERBOSE=1 ALLOW_USE_URING=0`
4. Ensure that libraries: `libgusli_clnt.so`, are installed under `/usr/lib/`.
5. Ensure that headers are installed under `/usr/include/gusli_*.hpp`.
6. Build NIXL. [!IMPORTANT] You must build gusli before building NIXL
7. Once the Gusli Backend is built, you can use it in your data transfer task by specifying the backend name as "GUSLI".
8. See example in nixl_gusli_test.cpp file. In short:

```cpp
nixlAgent agent("your_client_name", nixlAgentConfig(true));
nixl_b_params_t params = gen_gusli_plugin_params(agent);	// Insert list of your block devices here, grep this function to see how it is used
nixlBackendH* gusli_ptr = nullptr;		// Backend gusli plugin (typically dont need to access this pointer)
nixl_status_t status = agent.createBackend("GUSLI", params, n_backend);
...
```

## GUSLI config file

A GUSLI configuration file is a simple text file with the following structure:

1.  **Comment Header (Optional):** Lines beginning with `#` are comments.
2.  **Version Directive:** The file must contain a version line (e.g., `version=1`).
3.  **Block Device Entries:** Each subsequent line defines a single block device (`bdev`).

### Basic Layout

```ini
# Config file for gusli client lib
version=1
# bdevs: id type how direct path security_cookie
# --- Block Device Entries Below ---
[bdev_entry_1]
[bdev_entry_2]
...
```

## Block Device (`bdev`) Entry Format

Each `bdev` entry is a single line of space-separated values.

### Syntax

`id type how direct path security_cookie`

### Fields

| Field             | Description                                                                                             | Example                  |
| :---------------- | :------------------------------------------------------------------------------------------------------ | :----------------------- |
| `id`              | A unique 16-byte string identifier for the device.                                                      | `11`                     |
| `type`            | A single character representing the device type. See Device Types below.                                | `F`                      |
| `how`             | A single character representing the access mode. See Access Modes below.                                | `W`                      |
| `direct`          | A flag for I/O mode: `D` for Direct I/O (bypasses page cache) or `N` for Normal I/O.                    | `D`                      |
| `path`            | The path or address to the device.                                                                      | `./store0.bin`           |
| `security_cookie` | A 16-byte UTF-8 string for server authentication. Can be `none` if not used.                            | `none`                   |

---

## Field Details

### Device ID (`id`)

The `id` is a string (up to 15 characters + null terminator) used to uniquely identify a volume within GUSLI. Simple integers represented as strings (e.g., `11`) are common.

### Device Types (`type`)

The `type` field determines the backend for the block device.

| Char | Description                                                              |
| :--- | :----------------------------------------------------------------------- |
| `F`  | A local file used as a block device (e.g., `./store0.bin`).              |
| `K`  | A standard kernel block device (e.g., `/dev/nvme0n1`, `/dev/zero`).      |
| `N`  | A remote block device accessed via a networked GUSLI server.             |
| `x`  | A dummy device that always fails I/O operations (for testing errors).    |
| `s`  | A dummy device that never completes I/O (for testing timeouts).          |

### Access Modes (`how`)

The `how` field defines the access permissions for the device.

| Char | Description                 |
| :--- | :-------------------------- |
| `W`  | Shared Read/Write access.     |
| `R`  | Read-Only access.           |
| `X`  | Exclusive Read/Write access.  |

### Path

The `path` string's meaning depends on the device `type`:
*   **For `F` and `K` types:** A standard filesystem path (e.g., `/mnt/data/my_file.img`, `/dev/sdc`).
*   **For `N` type:** A remote server address, prefixed with the protocol: `t` for TCP or `u` for UDP (e.g., `t127.0.0.1`, `u10.0.0.5`).

---

## Example Configuration File

Here is an example of a `gusli.conf` file demonstrating various device types and all primary access modes (`R`, `W`, `X`).

```ini
# ============================================
# GUSLI Client Configuration Example
# ============================================
version=1

# Fields: id type how direct path security_cookie
# ----------------------------------------------------

# ID 11: A local file for general read/write tests.
# Access Mode: W (Shared Read/Write)
11 F W D ./scratch.bin none

# ID 15: A source data file that should not be modified.
# Access Mode: R (Read-Only)
15 F R D ./weights.bin none

# ID 20: A physical NVMe drive.
# Access Mode: X (Exclusive Read/Write)
20 K X D /dev/nvme0n1 none

# ID 35: A remote GUSLI server.
# Access Mode: W (Shared Read/Write)
35 N W D t127.0.0.1 my_secret_key

# ID 98: A dummy device for testing error handling.
98 x W D dummy_path none

# ID 99: A dummy device for testing timeouts.
99 s W D dummy_path none
```

## Running gusli unit test
1. build NIXL in a directory. Example: be it `/root/NNN`
2. `clear; ninja -C /root/NNN install`
3. Run gusli unit-test via: `clear; /root/NNN/test/unit/plugins/gusli/nixl_gusli_test; GUSLI show`
4. Run in unit-test framework via `rm /root/NNN/meson-logs/testlog.txt; meson test gusli_plugin_test -C /root/NNN; cat /root/NNN/meson-logs/testlog.txt`


## Known Issues
1. The `Notif[ication]` and `ProgTh[read]` features are not supported.
