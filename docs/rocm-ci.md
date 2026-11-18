# ROCm Based NIXL CI Setup and Build

The AMD/ROCm build and CI path is maintained separately from the NVIDIA one.
ROCm images come from [`.ci/dockerfiles/Dockerfile.rocm`](../.ci/dockerfiles/Dockerfile.rocm)
and [`.gitlab/build-rocm.sh`](../.gitlab/build-rocm.sh); the NVIDIA
`Dockerfile.base` / `.gitlab/build.sh` flow is not used here and needs no
ROCm-specific changes.

## Build Process

The ROCm image is built directly on a public ROCm base and is self-contained —
it provides HIP, hipFile (`libhipfile.so`), the `ais-stats` CLI, and a UCX built
with `--with-rocm`. No CUDA stack is involved.

```bash
DOCKER_BUILDKIT=1 docker build -f .ci/dockerfiles/Dockerfile.rocm \
    --build-arg ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full \
    -t nixl-rocm .
```

The default base is ROCm 7.14. ROCm 10.1 is selected through build args only.
The torch layer installs `torch==${TORCH_VERSION}+rocm<X.Y.Z>`, where `X.Y.Z` is
read from the base image (`/opt/rocm/core/.info/version`), so `TORCH_VERSION`
and `TORCH_INDEX_URL` must name a torch that AMD publishes for that ROCm
release. ROCm 10 wheels are not on the default index:

```bash
DOCKER_BUILDKIT=1 docker build -f .ci/dockerfiles/Dockerfile.rocm \
    --build-arg ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:10.1.0-full \
    --build-arg TORCH_VERSION=2.14.0 \
    --build-arg TORCH_INDEX_URL=https://stable.repo.amd.com/rocm/whl-next/ \
    -t nixl-rocm:10.1 .
```

| Build arg | Default (ROCm 7.14) | ROCm 10.1 |
| --- | --- | --- |
| `ROCM_BASE_IMAGE` | `rocm/dev-ubuntu-24.04:7.14.0-full` | `rocm/dev-ubuntu-24.04:10.1.0-full` |
| `TORCH_VERSION` | `2.11.0` | `2.14.0` |
| `TORCH_INDEX_URL` | `https://repo.amd.com/rocm/whl-multi-arch/` | `https://stable.repo.amd.com/rocm/whl-next/` |

The image is x86_64 only.

`Dockerfile.rocm` is a BuildKit-parallel refactor of the sequential
`build-rocm.sh` flow: each source dependency builds in its own stage so the DAG
scheduler runs them concurrently. `build-rocm.sh` remains the reference for
configuring NIXL/nixlbench (`-Dnixlbench_gpu=rocm -Drocm_path=...`); run it
inside the image with `PRE_INSTALLED_ENV=1 PRE_INSTALLED_UCX_ENV=1`.

### GPU memory support in UCX and libfabric

Both transports are built against the ROCm prefix `${ROCM_PATH}/core`
(`ROCM_PATH` is set by the ROCm base image; the build fails early if it is
empty):

- UCX is configured with `--with-rocm`, and the build fails if no UCX ROCm
  module was produced.
- libfabric is configured with `--with-rocr --enable-rocr-dlopen`, which
  compiles in the ROCr memory interface (`FI_HMEM_ROCR`) used by the NIXL
  libfabric plugin. libfabric does not link the ROCm runtime; it loads
  `libhsa-runtime64.so` with `dlopen()` when it initializes. The image puts
  `${ROCM_PATH}/core/lib` on `LD_LIBRARY_PATH` so that it is found.

### Python package naming

`build-rocm.sh` does not pass `-Dwheel_variant`. With `-Drocm_path` set and no
CUDA present, meson derives the variant from the ROCm install as
`rocm<major>_<minor>`, so the Python bindings are installed as
`nixl_rocm7_14` on a ROCm 7.14 image and `nixl_rocm10_1` on a ROCm 10.1 image
(previously `nixl_rocm`). To force a name, pass `-Dwheel_variant=<variant>` in
the script's third argument (`EXTRA_BUILD_ARGS`).

The build also produces upstream's pure-Python `nixl` meta wheel (the `nixl`
dispatcher), exactly as a CUDA build does, and the script copies it to
`dist/nixl-<version>-py3-none-any.whl`. It depends on the backend of the ROCm
version it was built against (`nixl-rocm7.14` or `nixl-rocm10.1`) and is for
local use only; it is not the `rocm-nixl` distribution.

This image is a CI and development environment. The `rocm-nixl` release wheel
is built by a separate flow: `contrib/Dockerfile.manylinux` with
`GPU_STACK=rocm` builds one intermediate backend wheel per ROCm version
(`nixl_rocm<X>_<Y>-*.whl`, no meta wheel), and `contrib/wheel_merge_rocm.py`
merges them into a single `rocm_nixl-*.whl`; see
[`contrib/README.md`](../contrib/README.md).

### Building behind a TLS-intercepting proxy

Both `Dockerfile.rocm` and `contrib/Dockerfile.manylinux` accept the proxy's
root CA certificate as an optional BuildKit secret named `proxy_ca`. When it is
given, the certificate is added to the image's system trust store before the
first HTTPS fetch that follows the base package install; without it the step
does nothing.

```bash
DOCKER_BUILDKIT=1 docker build -f .ci/dockerfiles/Dockerfile.rocm \
    --secret id=proxy_ca,src=/path/to/proxy-root-ca.pem \
    -t nixl-rocm .
```

For the wheel build, `contrib/build-container.sh --proxy-ca-cert FILE` forwards
the same secret.

## AIS on AMD GPU (docker + nixlbench)

For an interactive AIS run on VRAM with an NVMe-backed directory mounted
into the container, hipFile stats (`HIPFILE_STATS_LEVEL`), and `ais-stats`
wrapping `nixlbench`, use
[`benchmark/nixlbench/contrib/run-ais-amd-gpu-test.sh`](../benchmark/nixlbench/contrib/run-ais-amd-gpu-test.sh).

On the host, prefer mounting a directory on a filesystem that sits on a stable
NVMe identity path under `/dev/disk/by-id/`. Inside the container, set
`FILEPATH` to an existing, writable **directory** under the mount (nixlbench
creates `nixlbench_ais_test_file_*` files inside it; a single pre-created
`.bin` file path will fail with “Not a directory”). The script runs
**single-process** AIS (null runtime; no etcd) which matches nixlbench’s
storage-backend layout.

If the container reports `No such file or directory` for
`run-ais-amd-gpu-test.sh`, bind-mount it from the host — the image does not
ship the repo (see the `-v` line in the example below).

The script does not create `FILEPATH`. If the image user cannot write your
host-owned bind mount, run `mkdir -p` on the host for the path you use as
`FILEPATH` and make it writable, or add `--user "$(id -u):$(id -g)"` to
`docker run` (plus `--group-add` for `video` / `render` if `/dev/dri` needs
those groups). From repo root, bind-mounting the contrib script (omit that `-v`
if the image already contains it):

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --security-opt seccomp=unconfined \
  -v /mnt/nvme-nixlbench:/mnt/nvme-nixlbench:rw \
  -v "$PWD/benchmark/nixlbench/contrib/run-ais-amd-gpu-test.sh:/workspace/nixl/benchmark/nixlbench/contrib/run-ais-amd-gpu-test.sh:ro" \
  -e FILEPATH=/mnt/nvme-nixlbench/nixlbench-ais \
  -e HIPFILE_STATS_LEVEL=1 \
  nixl-rocm:latest \
  bash /workspace/nixl/benchmark/nixlbench/contrib/run-ais-amd-gpu-test.sh
```

The contrib script sets `HOME=/tmp` when the image `HOME` is not executable for
the current UID (typical with `docker --user` while `ENV HOME` still points at
`/home/svc-nixl`), so NIXL does not resolve `$HOME/.nixl.cfg` under that path.
Alternatively set `NIXL_CONFIG_FILE` to a readable file or pass `-e HOME=/tmp`
before the benchmark runs.

If nixlbench fails with `Permission denied` while **creating** files under
`FILEPATH`, the directory exists but is not writable for the container UID
(same UID mismatch as above). Grant write access, for example on the host
`sudo chown -R <image-uid>:<image-gid> /mnt/.../nixlbench-ais` (check with
`docker run --rm <image> id`), or use the `docker run`
example block above with `--user "$(id -u):$(id -g)"`.

A successful run of the `docker run` command above looks like this (numbers
will differ):

```
HIPFILE_STATS_LEVEL=1
Running: /opt/nixl/bin/nixlbench --backend AIS --filepath /mnt/nvme-nixlbench/nixlbench-ais --initiator_seg_type VRAM --target_seg_type VRAM --gds_mt_num_threads 4 --total_buffer_size 67108864 --start_block_size 65536 --max_block_size 65536 --start_batch_size 1 --max_batch_size 1 --num_iter 200 --warmup_iter 40 --op_type WRITE --check_consistency
WARNING: Adjusting num_iter to 208 to allow equal distribution to 1 threads
WARNING: Adjusting warmup_iter to 48 to allow equal distribution to 1 threads
Using null runtime for storage backend without ETCD
AIS backend
AIS thread_count: 4
Waiting for all processes to start... (expecting 1 total: 1 initiators and 1 targets)
All processes are ready to proceed
Creating file: /mnt/nvme-nixlbench/nixlbench-ais/nixlbench_ais_test_file_initiator_0
nixlbench: SG mode — set NEURON_RT_NUM_CORES=1
****************************************************************************************************************************************************************
NIXLBench Configuration
****************************************************************************************************************************************************************
Runtime (--runtime_type=[ETCD,ASIO])                        : ETCD
ETCD Endpoint                                               : disabled (storage backend)
Worker type (--worker_type=[nixl,nvshmem])                  : nixl
Backend (--backend=[UCX,GDS,GDS_MT,AIS,POSIX,Mooncake,HF3FS,OBJ,REDIS,GUSLI,AZURE_BLOB]): AIS
Enable pt (--enable_pt=[0,1])                               : 0
Progress threads (--progress_threads=N)                     : 0
Device list (--device_list=dev1,dev2,...)                   : all
Enable VMM (--enable_vmm=[0,1])                             : 0
VMM GPUDirect RDMA capable (--vmm_gdr_capable=[0,1])        : 1
VMM locality domain (--use_localized=[-1,0,1])              : -1
Recreate xfer each iteration (--recreate_xfer=[0,1])        : 0
Re-register memory each iteration (--reregister_mem=[0,1])  : 0
Prepared xfer (prep+make) (--prepared_xfer=[0,1])           : 0
Pipeline depth (--pipeline_depth=N)                         : 1
Use hugepages (--use_hugepages=[0,1])                       : 0
GDS_MT / AIS thread pool (--gds_mt_num_threads=N)           : 4
filepath (--filepath=path)                                  : /mnt/nvme-nixlbench/nixlbench-ais
filenames (--filenames=filename1,filename2,...)             :
Number of files (--num_files=N)                             : 1
Storage enable direct (--storage_enable_direct=[0,1])       : 0
Randomize location mode (--randomize_location_mode=[none, blockaligned, bytealigned]): none
Initiator seg type (--initiator_seg_type=[DRAM,VRAM])       : VRAM
Target seg type (--target_seg_type=[DRAM,VRAM])             : VRAM
Scheme (--scheme=[pairwise,manytoone,onetomany,tp])         : pairwise
Mode (--mode=[SG,MG])                                       : SG
Op type (--op_type=[READ,WRITE])                            : WRITE
Check consistency (--check_consistency=[0,1])               : 1
Total buffer size (--total_buffer_size=N)                   : 67108864
Num initiator dev (--num_initiator_dev=N)                   : 1
Num target dev (--num_target_dev=N)                         : 1
Start block size (--start_block_size=N)                     : 65536
Max block size (--max_block_size=N)                         : 65536
Start batch size (--start_batch_size=N)                     : 1
Max batch size (--max_batch_size=N)                         : 1
Num iter (--num_iter=N)                                     : 208
Warmup iter (--warmup_iter=N)                               : 48
Large block iter factor (--large_blk_iter_ftr=N)            : 16
Num threads (--num_threads=N)                               : 1
Use Device API (--use_device_api=[0,1])                     : 0
----------------------------------------------------------------------------------------------------------------------------------------------------------------

Block Size (B)      Batch Size     B/W (GB/Sec)   Avg Lat. (us)  Avg Prep (us)  P99 Prep (us)  Avg Post (us)  P99 Post (us)  Avg Tx (us)    P99 Tx (us)
----------------------------------------------------------------------------------------------------------------------------------------------------------------
65536               1              1.530252       42.8           19.0           19.0           5.2            6.0            37.5           43.0
AIS-STATS Version: 1
HipFile Stats Level: 1
File Handle Registrations: 1
Buffer Registrations: 1
Fastpath Rejections: 0

Total Fastpath Read Size (B): 0
Average Fastpath Read Bandwidth (GiB/s): 0
Average Fastpath Read Latency (us): 0
Total Fastpath Read Errors: 0
Total Fastpath Read Unaligned: 0

Total Fastpath Write Size (B): 16777216
Average Fastpath Write Bandwidth (GiB/s): 1.56172
Average Fastpath Write Latency (us): 39.082
Total Fastpath Write Errors: 0
Total Fastpath Write Unaligned: 0

Total Fallback Read Size (B): 0
Average Fallback Read Bandwidth (GiB/s): 0
Average Fallback Read Latency (us): 0
Total Fallback Read Errors: 0
Total Fallback Read Unaligned: 0

Total Fallback Write Size (B): 0
Average Fallback Write Bandwidth (GiB/s): 0
Average Fallback Write Latency (us): 0
Total Fallback Write Errors: 0
Total Fallback Write Unaligned: 0

GPU 0:
IO Size Histogram
IO Size (KiB)               Fastpath Read Size (B)           Fastpath Write Size (B)            Fallback Read Size (B)           Fallback Write Size (B)
0-4                                              0                                 0                                 0                                 0
4-8                                              0                                 0                                 0                                 0
8-16                                             0                                 0                                 0                                 0
16-32                                            0                                 0                                 0                                 0
32-64                                            0                                 0                                 0                                 0
64-128                                           0                          16777216                                 0                                 0
128-256                                          0                                 0                                 0                                 0
256-512                                          0                                 0                                 0                                 0
512-1024                                         0                                 0                                 0                                 0
1024-2048                                        0                                 0                                 0                                 0
2048-4096                                        0                                 0                                 0                                 0
4096-8192                                        0                                 0                                 0                                 0
8192-16384                                       0                                 0                                 0                                 0
16384-32768                                      0                                 0                                 0                                 0
32768-65536                                      0                                 0                                 0                                 0
65536-...                                        0                                 0                                 0                                 0
IO Bandwidth Histogram
IO Size (KiB)      Fastpath Read Bandwidth (GiB/s)  Fastpath Write Bandwidth (GiB/s)   Fallback Read Bandwidth (GiB/s)  Fallback Write Bandwidth (GiB/s)
0-4                                              0                                 0                                 0                                 0
4-8                                              0                                 0                                 0                                 0
8-16                                             0                                 0                                 0                                 0
16-32                                            0                                 0                                 0                                 0
32-64                                            0                                 0                                 0                                 0
64-128                                           0                           1.56172                                 0                                 0
128-256                                          0                                 0                                 0                                 0
256-512                                          0                                 0                                 0                                 0
512-1024                                         0                                 0                                 0                                 0
1024-2048                                        0                                 0                                 0                                 0
2048-4096                                        0                                 0                                 0                                 0
4096-8192                                        0                                 0                                 0                                 0
8192-16384                                       0                                 0                                 0                                 0
16384-32768                                      0                                 0                                 0                                 0
32768-65536                                      0                                 0                                 0                                 0
65536-...                                        0                                 0                                 0                                 0
IO Latency Histogram
IO Size (KiB)           Fastpath Read Latency (us)       Fastpath Write Latency (us)        Fallback Read Latency (us)       Fallback Write Latency (us)
0-4                                              0                                 0                                 0                                 0
4-8                                              0                                 0                                 0                                 0
8-16                                             0                                 0                                 0                                 0
16-32                                            0                                 0                                 0                                 0
32-64                                            0                                 0                                 0                                 0
64-128                                           0                            39.082                                 0                                 0
128-256                                          0                                 0                                 0                                 0
256-512                                          0                                 0                                 0                                 0
512-1024                                         0                                 0                                 0                                 0
1024-2048                                        0                                 0                                 0                                 0
2048-4096                                        0                                 0                                 0                                 0
4096-8192                                        0                                 0                                 0                                 0
8192-16384                                       0                                 0                                 0                                 0
16384-32768                                      0                                 0                                 0                                 0
32768-65536                                      0                                 0                                 0                                 0
65536-...                                        0                                 0                                 0                                 0
IO Errors Histogram
IO Size (KiB)            Fastpath Read Error Count        Fastpath Write Error Count         Fallback Read Error Count        Fallback Write Error Count
0-4                                              0                                 0                                 0                                 0
4-8                                              0                                 0                                 0                                 0
8-16                                             0                                 0                                 0                                 0
16-32                                            0                                 0                                 0                                 0
32-64                                            0                                 0                                 0                                 0
64-128                                           0                                 0                                 0                                 0
128-256                                          0                                 0                                 0                                 0
256-512                                          0                                 0                                 0                                 0
512-1024                                         0                                 0                                 0                                 0
1024-2048                                        0                                 0                                 0                                 0
2048-4096                                        0                                 0                                 0                                 0
4096-8192                                        0                                 0                                 0                                 0
8192-16384                                       0                                 0                                 0                                 0
16384-32768                                      0                                 0                                 0                                 0
32768-65536                                      0                                 0                                 0                                 0
65536-...                                        0                                 0                                 0                                 0
Fastpath Read Unaligned Count: 0
Fastpath Write Unaligned Count: 0
Fallback Read Unaligned Count: 0
Fallback Write Unaligned Count: 0
```
