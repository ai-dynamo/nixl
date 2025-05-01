Mooncake transfer engine is a high-performance, zero-copy data transfer library. To achieve better performance in NIXL, we have designed an new backend based on Mooncake transfer engine. 

# Usage
1. Build the install [Mooncake](https://github.com/kvcache-ai/Mooncake) manually:
```
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j
sudo make install # compulsory
```
2. Build NIXL with option `disable_mooncake_backend` set as `false`.
3. When Mooncake Backend is built, you can use it in you data transfer task by specifying the backend name as "Mooncake":
```cpp
    nixl_status_t ret1;
    std::string ret_s1;
    nixlAgentConfig cfg(true);
    nixl_b_params_t init1;
    nixl_mem_list_t mems1;
    nixlAgent A1(agent1, cfg);
    ret1 = A1.getPluginParams("Mooncake", mems1, init1);
    assert (ret1 == NIXL_SUCCESS);
    nixlBackendH* ucx1, *ucx2;
    ret1 = A1.createBackend("Mooncake", init1, ucx1);
    ...
```
