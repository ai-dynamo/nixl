# Control Plane Latency Test Suite

Measures latency of NIXL EP Buffer control plane operations:
**init**, **connect**, **disconnect**, **reconnect**, **destroy**, and a **full cycle**.

## Single-node (8 GPUs):

```bash
cd nixl/examples/device/ep
python3 tests/control/control.py --num-processes 8
python3 tests/control/control.py --num-processes 8 --mode connect
python3 tests/control/control.py --num-processes 8 --mode connect --warmup 1 --iters 5
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
python3 tests/control/control.py --num-processes 8 --num-ranks 16 --tcp-server $MASTER_IP
```

## Modes

| Mode | What is measured |
|------|-----------------|
| `cycle` | Full cycle: init → connect → disconnect → reconnect → destroy |
| `init` | `Buffer()` + `update_memory_buffers()` |
| `connect` | `connect_ranks()` |
| `disconnect` | `disconnect_ranks()` |
| `reconnect` | `connect_ranks()` after a prior disconnect |
| `destroy` | `buffer.destroy()` |
