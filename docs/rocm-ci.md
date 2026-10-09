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

`Dockerfile.rocm` is a BuildKit-parallel refactor of the sequential
`build-rocm.sh` flow: each source dependency builds in its own stage so the DAG
scheduler runs them concurrently. `build-rocm.sh` remains the reference for the
sequential build and for configuring NIXL/nixlbench
(`-Dnixlbench_gpu=rocm -Drocm_path=...`).

## AIS_MT on AMD GPU (docker + nixlbench)

For an interactive AIS_MT run on VRAM with an NVMe-backed directory mounted
into the container, hipFile stats (`HIPFILE_STATS_LEVEL`), and `ais-stats`
wrapping `nixlbench`, use
[`benchmark/nixlbench/contrib/run-ais-mt-amd-gpu-test.sh`](../benchmark/nixlbench/contrib/run-ais-mt-amd-gpu-test.sh).

On the host, prefer mounting a directory on a filesystem that sits on a stable
NVMe identity path under `/dev/disk/by-id/`. Inside the container, set
`FILEPATH` to an existing, writable **directory** under the mount (nixlbench
creates `nixlbench_ais_mt_test_file_*` files inside it; a single pre-created
`.bin` file path will fail with “Not a directory”). The script runs
**single-process** AIS_MT (null runtime; no etcd) which matches nixlbench’s
storage-backend layout.

If the container reports `No such file or directory` for
`run-ais-mt-amd-gpu-test.sh`, bind-mount it from the host — the image does not
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
  -v "$PWD/benchmark/nixlbench/contrib/run-ais-mt-amd-gpu-test.sh:/workspace/nixl/benchmark/nixlbench/contrib/run-ais-mt-amd-gpu-test.sh:ro" \
  -e FILEPATH=/mnt/nvme-nixlbench/nixlbench-ais-mt \
  -e HIPFILE_STATS_LEVEL=1 \
  nixl-rocm:latest \
  bash /workspace/nixl/benchmark/nixlbench/contrib/run-ais-mt-amd-gpu-test.sh
```

The contrib script sets `HOME=/tmp` when the image `HOME` is not executable for
the current UID (typical with `docker --user` while `ENV HOME` still points at
`/home/svc-nixl`), so NIXL does not resolve `$HOME/.nixl.cfg` under that path.
Alternatively set `NIXL_CONFIG_FILE` to a readable file or pass `-e HOME=/tmp`
before the benchmark runs.

If nixlbench fails with `Permission denied` while **creating** files under
`FILEPATH`, the directory exists but is not writable for the container UID
(same UID mismatch as above). Grant write access, for example on the host
`sudo chown -R <image-uid>:<image-gid> /mnt/.../nixlbench-ais-mt` (check with
`docker run --rm <image> id`), or use the `docker run`
example block above with `--user "$(id -u):$(id -g)"`.

A successful run of the `docker run` command above looks like this (numbers
and log paths will differ):

```
HIPFILE_STATS_LEVEL=1 (ais-stats + nixlbench log: /tmp/ais-mt-ais-stats.uYulz6.log)
Running: /opt/nixl/bin/nixlbench --backend AIS_MT --filepath /mnt/nvme-nixlbench/nixlbench-ais-mt --initiator_seg_type VRAM --target_seg_type VRAM --gds_mt_num_threads 4 --total_buffer_size 67108864 --start_block_size 65536 --max_block_size 65536 --start_batch_size 1 --max_batch_size 1 --num_iter 200 --warmup_iter 40 --op_type WRITE --check_consistency
WARNING: Adjusting num_iter to 208 to allow equal distribution to 1 threads
WARNING: Adjusting warmup_iter to 48 to allow equal distribution to 1 threads
Using null runtime for storage backend without ETCD
AIS_MT backend
AIS_MT thread_count: 4
Waiting for all processes to start... (expecting 1 total: 1 initiators and 1 targets)
All processes are ready to proceed
Creating file: /mnt/nvme-nixlbench/nixlbench-ais-mt/nixlbench_ais_mt_test_file_initiator_0
****************************************************************************************************************************************************************
NIXLBench Configuration
****************************************************************************************************************************************************************
Runtime (--runtime_type=[ETCD,ASIO])                        : ETCD
ETCD Endpoint                                               : disabled (storage backend)
Worker type (--worker_type=[nixl,nvshmem])                  : nixl
Backend (--backend=[UCX,GDS,GDS_MT,AIS_MT,POSIX,Mooncake,HF3FS,OBJ,AZURE_BLOB]): AIS_MT
Enable pt (--enable_pt=[0,1])                               : 0
Progress threads (--progress_threads=N)                     : 0
Device list (--device_list=dev1,dev2,...)                   : all
Enable VMM (--enable_vmm=[0,1])                             : 0
Recreate xfer each iteration (--recreate_xfer=[0,1])        : 0
Re-register memory each iteration (--reregister_mem=[0,1])  : 0
Prepared xfer (prep+make) (--prepared_xfer=[0,1])           : 0
Pipeline depth (--pipeline_depth=N)                         : 1
Use hugepages (--use_hugepages=[0,1])                       : 0
GDS_MT / AIS_MT thread pool (--gds_mt_num_threads=N)        : 4
filepath (--filepath=path)                                  : /mnt/nvme-nixlbench/nixlbench-ais-mt
filenames (--filenames=filename1,filename2,...)             :
Number of files (--num_files=N)                             : 1
Storage enable direct (--storage_enable_direct=[0,1])       : 0
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
----------------------------------------------------------------------------------------------------------------------------------------------------------------

Block Size (B)      Batch Size     B/W (GB/Sec)   Avg Lat. (us)  Avg Prep (us)  P99 Prep (us)  Avg Post (us)  P99 Post (us)  Avg Tx (us)    P99 Tx (us)
----------------------------------------------------------------------------------------------------------------------------------------------------------------
65536               1              1.236753       53.0           16.0           16.0           4.0            6.0            48.9           64.0
AIS-STATS Version: 1
HipFile Stats Level: 1
File Handle Registrations: 1
Buffer Registrations: 1
Fastpath Rejections: 0

Total Fastpath Read Size (B): 0
Average Fastpath Read Bandwidth (GiB/s): 0
Average Fastpath Read Latency (us): 0
Total Fastpath Read Errors: 0

Total Fastpath Write Size (B): 16777216
Average Fastpath Write Bandwidth (GiB/s): 1.14452
Average Fastpath Write Latency (us): 53.3281
Total Fastpath Write Errors: 0

Total Fallback Read Size (B): 0
Average Fallback Read Bandwidth (GiB/s): 0
Average Fallback Read Latency (us): 0
Total Fallback Read Errors: 0

Total Fallback Write Size (B): 0
Average Fallback Write Bandwidth (GiB/s): 0
Average Fallback Write Latency (us): 0
Total Fallback Write Errors: 0

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
64-128                                           0                           1.14452                                 0                                 0
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
64-128                                           0                           53.3281                                 0                                 0
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
OK: ais-stats / hipFile output mentions fastpath (see /tmp/ais-mt-ais-stats.uYulz6.log).
```
