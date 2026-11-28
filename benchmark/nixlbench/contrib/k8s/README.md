# nixlbench on Kubernetes

Runs nixlbench in a pod on one node of a Kubernetes cluster. The pod sleeps, and
you run nixlbench in it with `kubectl exec`, so one pod serves any number of
storage and memory benchmarks on that node.

## Prerequisites

- A nixlbench image in a registry the cluster can pull from. Build it with
  `contrib/build.sh` and push it; there is no published image.
- The NVIDIA device plugin, so `nvidia.com/gpu` can be requested. nixlbench
  links libcuda and needs the driver libraries even for DRAM-only runs.
- For storage benchmarks, the device mounted on the node, for example an NVMe
  drive at `/mnt/nvme0`.
- For GDS, the pod mounts the host's `/run/udev` read-only; cuFile reads the
  block device attributes from it and fails to register files without it.

## Deploy

Edit the three values marked `EDIT` in `nixlbench-pod.yaml`: the node name, the
image, and the host directory to benchmark. Then:

```bash
kubectl apply -f nixlbench-pod.yaml
kubectl wait --for=condition=Ready pod/nixlbench --timeout=600s
```

The host directory must be a directory on the device you want to measure. The
manifest uses `type: Directory`, so a wrong path fails at pod creation instead
of silently benchmarking the root disk.

## Run

Storage, one process, no etcd needed:

```bash
kubectl exec -it nixlbench -- nixlbench --backend POSIX --posix_api_type URING \
    --filepath /mnt/storage --storage_enable_direct

kubectl exec -it nixlbench -- nixlbench --backend GDS --initiator_seg_type VRAM \
    --filepath /mnt/storage --storage_enable_direct
```

Memory to memory with UCX needs two nixlbench processes and an etcd they both
reach. Both run in the same pod; etcd is in the image:

```bash
kubectl exec -it nixlbench -- bash
etcd --data-dir /tmp/etcd --listen-client-urls http://127.0.0.1:2379 \
    --advertise-client-urls http://127.0.0.1:2379 &
nixlbench --backend UCX --initiator_seg_type DRAM --target_seg_type VRAM \
    --etcd_endpoints http://127.0.0.1:2379 &
nixlbench --backend UCX --initiator_seg_type DRAM --target_seg_type VRAM \
    --etcd_endpoints http://127.0.0.1:2379
```

The first process to register with etcd becomes the initiator. Before another
run, clear the keys with `etcdctl del xferbench --prefix`, or pass a different
`--benchmark_group` to each run.

## Notes

- `seccompProfile: Unconfined` is only needed for `--posix_api_type URING`.
  Switch to `RuntimeDefault` for other backends.
- Requests are below limits on purpose. With a static CPU manager, a
  Guaranteed pod gets an exclusive cpuset, which gets in the way of placing
  the benchmark threads yourself.
- Multi-node runs need an etcd Service reachable from both pods and one pod per
  node; this manifest covers a single node.

## Clean up

```bash
kubectl delete pod nixlbench
```
