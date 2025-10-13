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

## Sample config

The config file can be generated using the GUSLI API:

```cpp
gusli::client_config_file conf(1 /*Version*/);
using gsc = gusli::bdev_config_params;
conf.bdev_add(gsc(__stringify (UUID_LOCAL_FILE_0), gsc::bdev_type::DEV_FS_FILE,    "./store0.bin", "sec=0x03", 0, gsc::connect_how::SHARED_RW));
conf.bdev_add(gsc(__stringify (UUID_K_DEV_ZERO_1), gsc::bdev_type::DEV_BLK_KERNEL, "/dev/zero",    "sec=0x71", 0, gsc::connect_how::EXCLUSIVE_RW));
conf.bdev_add(gsc(__stringify (UUID_NVME_DISK__0), gsc::bdev_type::DEV_BLK_KERNEL, "/dev/nvme0n1", "sec=0x07", 1, gsc::connect_how::EXCLUSIVE_RW));
params["config_file"] = conf.get();
```

See [gusli_client_api.hpp](https://github.com/NVIDIA/GUSLI/blob/main/gusli_client_api.hpp) for more details.

## Running gusli unit test
1. build NIXL in a directory. Example: be it `/root/NNN`
2. `clear; ninja -C /root/NNN install`
3. Run gusli unit-test via: `clear; /root/NNN/test/unit/plugins/gusli/nixl_gusli_test; GUSLI show`
4. Run in unit-test framework via `rm /root/NNN/meson-logs/testlog.txt; meson test gusli_plugin_test -C /root/NNN; cat /root/NNN/meson-logs/testlog.txt`


## Known Issues
1. The `Notif[ication]` and `ProgTh[read]` features are not supported.
