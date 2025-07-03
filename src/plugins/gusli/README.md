# NIXL Gusli Plugin

This plugin utilizes `gusli_clnt.so` as an I/O backend for NIXL. gusli client communicates with Gusli server (different process on local machine) and via shared memory sends the content of the io.
Gusli server is integrated into user space block device storage provider, like SPDK, NVMeshUM etc.
IO completely bypasses the kernel.
Gusli client can work without Server, accessing local block devices/files, but this is inefficient as it uses standard kernel API's

## Usage Guide
1. Build and install [Gusli](https://gitlab-master.nvidia.com/excelero/gusli).
2. Do it via: git clone ssh://git@gitlab-master.nvidia.com:12051/excelero/gusli.git
3. cd gusli; make clean all BUILD_RELEASE=1 BUILD_FOR_UNITEST=0
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

## Known Issues
1. The `Notif[ication]` and `ProgTh[read]` features are not supported.
