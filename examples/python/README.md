# Python Examples

This directory contains Python examples for NIXL demonstrating multi-peer transfers with tensors.

## Python examples overview

- `basic_two_peers.py`: Simple two-peer example showing registration, metadata exchange, a single READ, notification, verification, and teardown.
- `expanded_two_peers.py`: Comprehensive example with parallel READs (prepped), parallel WRITEs (initialized), reposting READs, notification patterns, and minimal layout sharing.
- `nixl_gds_example.py`: GPU Direct Storage integration—buffer-to-file write and read-back verification.
- `partial_md_example.py`: Shows handling of partial metadata updates and retrying initialize_xfer.
- `query_mem_example.py`: Demonstrates querying backend memory/storage details for registered regions.
- `telemetry_reader.py`: Example for fetching and validating transfer telemetry.
- `remote_storage_example/`: Client-server peer-to-peer storage pipeline with supporting utilities and diagrams.

## basic_two_peers.py

Two processes (target and initiator) cooperate to perform the following:
- Registration and metadata exchange
- A single READ transfer from target to initiator
- Notification from initiator to target (e.g., `Done_reading`) via `get_new_notifs`
- Data verification and teardown

Run:
Start target:
```bash
python3 examples/python/basic_two_peers.py --mode target --ip 127.0.0.1 --port 5555
```
Then start initiator:
```bash
python3 examples/python/basic_two_peers.py --mode initiator --ip 127.0.0.1 --port 5555
```
Note: `127.0.0.1` refers to localhost for same-machine testing. Replace it with the actual IP addresses of two machines to run across hosts.
Options:
- `--use_cuda True` to use GPU tensors if supported by the backend and the system.



## expanded_two_peers.py

This extends `basic_two_peers.py` with additional use cases:
- Parallel READs using prep_xfer_dlist + make_prepped_xfer (known descriptor blocks)
- Parallel WRITEs using initialize_xfer (offsets chosen at transfer time)
- Reposting the READs (reading updated data again without re-prepping)
- Check transfers' status on the initiator side and wait for their completion on the target via notifications (exact match and starting tag examples)
- Final validation and teardown

Note: You can use more fine-grained try/except blocks.
Run:
Start target:
```bash
python3 examples/python/expanded_two_peers.py --mode target --ip 127.0.0.1 --port 5555 --backend UCX
```
Then start initiator:
```bash
python3 examples/python/expanded_two_peers.py --mode initiator --ip 127.0.0.1 --port 5555 --backend UCX
```
Options:
- `--backend` selects the backend (default: UCX). Note that the example is targetting network backends not storage.
- `--use_cuda True` to use GPU tensors if supported by the backend and the system.
