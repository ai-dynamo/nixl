# NIXL HF3FS Plugin

This plugin utilizes `hf3fs_usrbio.so` as an I/O backend for NIXL.

## Usage
1. Build and install [3FS](https://github.com/deepseek-ai/3FS/).
2. Ensure that 3FS libraries,`hf3fs_usrbio.so` and `libhf3fs_api_shared.so`, are installed under `/usr/lib/`.
3. Ensure that 3FS headers are installed under `/usr/include/hf3fs`.
4. Build NIXL.
5. Once the HF3FS Backend is built, you can use it in your data transfer task by specifying the backend name as "HF3FS":

```cpp
nixl_status_t ret1;
std::string ret_s1;
nixlAgentConfig cfg(true);
nixl_b_params_t init1;
nixl_mem_list_t mems1;
nixlBackendH      *hf3fs;
nixlAgent A1(agent1, cfg);
ret1 = A1.getPluginParams("HF3FS", mems1, init1);
assert (ret1 == NIXL_SUCCESS);
ret1 = A1.createBackend("HF3FS", init1, hf3fs);
...
```

### Backend parameters
Paramaters accepted by the HF3FS plugin during createBackend()
- mem_config: Indicate if the plugin should create a shared memory on the user-provided memory ["dram", "dram_zc", "auto"] (default: "auto")
	- dram_zc: always create shared memory and return failure if shared memory creation fails.
	- dram: Never create shared memory
	- auto: the plugin will try to create a shared memory and fallback to non shared memory if fails.

## Performance tuning
To get the best performance, please provide a memory that is page-aligned with sized the multiple of page size to `nixlAgent->registerMem()`.

This plugin will try to shared the user provided memory direcrly with the 3FS backend process using mmap() to reduce data copy.
mmap() requires the memory to be page-aligned, and size has to be multiple of page size.

If the user-provided memory could not be shared, the plugin will allocate it's own memory to shared with 3FS backend process and copy the data between user-provided memory and the shared memory.
