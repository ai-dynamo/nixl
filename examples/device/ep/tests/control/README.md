# Control Plane Latency Test Suite

Measures latency of a full NIXL EP Buffer control plane cycle:
**init → connect → disconnect → reconnect → destroy**.

Each iteration times every step individually and reports both per-rank and
cross-rank averages (plus the sum of per-op averages as `total`).

## Single-node (8 GPUs):

```bash
cd nixl/examples/device/ep
python3 tests/control/control.py --num-processes 8
python3 tests/control/control.py --num-processes 8 --warmup 1 --iters 5
```

## Multi-node Setup:

**node 0**:
```bash
cd nixl/examples/device/ep
python3 tests/control/control.py --num-processes 8 --num-ranks 16
```

**node 1**:
```bash
cd nixl/examples/device/ep
python3 tests/control/control.py --num-processes 8 --num-ranks 16 --tcp-server $SERVER_IP
```

## Measured steps

| Step | What is measured |
|------|-----------------|
| `init` | `Buffer()` + `update_memory_buffers()` |
| `connect` | `connect_ranks()` |
| `disconnect` | `disconnect_ranks()` |
| `reconnect` | `connect_ranks()` after a prior disconnect |
| `destroy` | `buffer.destroy()` |
| `total` | Sum of all per-step averages (i.e. the whole cycle) |
