---
title: Remote Storage Transfer
description: A client/server storage transfer system using NIXL with POSIX and GDS backends for local and remote storage operations.
---

This example demonstrates a high-performance storage transfer system built on NIXL that supports both local and remote storage operations. It uses POSIX and GDS (GPUDirect Storage) backends with UCX-based networking. Source: [`examples/python/remote_storage_example/`](https://github.com/ai-dynamo/nixl/tree/main/examples/python/remote_storage_example)

## Features

- **Flexible Storage Backends**: GDS for high-performance, POSIX fallback, automatic selection
- **Transfer Modes**: Local and remote memory-to-storage, bidirectional READ/WRITE, batch processing
- **Network Communication**: UCX-based data transfer, metadata exchange, async notifications

## Overview

The system operates in two modes. The **server** waits for requests from clients to READ/WRITE from its storage to a remote node. The **client** initiates transfers and performs both local and remote operations with storage servers.

The four phases below -- initialization, metadata exchange, remote write, and remote read -- cover the complete remote storage transfer lifecycle. Each phase includes a sequence diagram followed by a code walkthrough.

<Info>
In production, initialization and metadata exchange happen once at startup. Only the transfer phases (write/read) repeat per request.
</Info>

### Initialization

<div className="diagram-light">
<Frame caption="Phase 1: Initialization">
<img src="../figures/remote-storage/nixl_remote_storage_01_init_light.svg" alt="Initialization sequence diagram showing client and server agent creation, backend registration, and memory registration" style={{maxWidth: '78%'}} />
</Frame>
</div>
<div className="diagram-dark">
<Frame caption="Phase 1: Initialization">
<img src="../figures/remote-storage/nixl_remote_storage_01_init_dark.svg" alt="Initialization sequence diagram showing client and server agent creation, backend registration, and memory registration" style={{maxWidth: '78%'}} />
</Frame>
</div>

Both nodes create a NIXL agent, register storage backends (GDS_MT preferred, POSIX fallback), and register a UCX backend for network transfers. The client registers VRAM segments (GPU memory) and file descriptors. The server registers DRAM segments and file descriptors.

```python title="Python"
# Both client and server
my_agent = nixl_agent(agent_name, agent_config)

# Register storage + network backends
my_agent.create_backend("GDS_MT")   # or "POSIX" as fallback
my_agent.create_backend("UCX")

# Client: VRAM + FILE, Server: DRAM + FILE
nixl_mem_reg_descs  = my_agent.register_memory(mem_list, "VRAM")  # or "DRAM"
nixl_file_reg_descs = my_agent.register_memory(file_list, "FILE")
```

### Metadata Exchange

<div className="diagram-light">
<Frame caption="Phase 2: Metadata Exchange">
<img src="../figures/remote-storage/nixl_remote_storage_02_metadata_light.svg" alt="Metadata exchange sequence diagram showing client publishing metadata and fetching server metadata" style={{maxWidth: '78%'}} />
</Frame>
</div>
<div className="diagram-dark">
<Frame caption="Phase 2: Metadata Exchange">
<img src="../figures/remote-storage/nixl_remote_storage_02_metadata_dark.svg" alt="Metadata exchange sequence diagram showing client publishing metadata and fetching server metadata" style={{maxWidth: '78%'}} />
</Frame>
</div>

The client reads a list of storage servers from a file and connects to each one. For each server, the client publishes its own metadata and fetches the server's metadata, then polls until the exchange completes.

```python title="Python"
# For each server in agents_file ("<agent_name> <ip> <port>" per line)
my_agent.send_local_metadata(server_ip, server_port)
my_agent.fetch_remote_metadata(server_name, server_ip, server_port)

# Poll until metadata is available
while my_agent.check_remote_metadata(server_name) is False:
    time.sleep(1.0)
```

### Remote Write Request

<div className="diagram-light">
<Frame caption="Phase 3: Remote Write Request">
<img src="../figures/remote-storage/nixl_remote_storage_03_write_light.svg" alt="Remote write sequence diagram showing notification, pipelined network read and storage write, and completion" style={{maxWidth: '78%'}} />
</Frame>
</div>
<div className="diagram-dark">
<Frame caption="Phase 3: Remote Write Request">
<img src="../figures/remote-storage/nixl_remote_storage_03_write_dark.svg" alt="Remote write sequence diagram showing notification, pipelined network read and storage write, and completion" style={{maxWidth: '78%'}} />
</Frame>
</div>

The client serializes its VRAM descriptors and sends a `WRTE` notification to the server. The server deserializes the descriptors and executes a pipelined loop: **network read** (UCX read from client VRAM into server DRAM) followed by **storage write** (GDS/POSIX write from DRAM to local file). These two operations overlap across iterations for throughput. On completion, the server sends a `COMPLETE` notification back.

```python title="Python"
# Client: send write request
descs_str = my_agent.get_serialized_descs(my_mem_descs)
my_agent.send_notif(server_name, b"WRTE" + iterations_str + descs_str)

# Client: wait for completion
while not my_agent.check_remote_xfer_done(server_name, b"COMPLETE"):
    continue
```

```python title="Python"
# Server: pipelined write handler (network read → storage write)
sent_descs = my_agent.deserialize_descs(received_data)

# Each iteration: read from client network, then write to local storage
# Operations are pipelined across iterations using thread pool
handle = my_agent.initialize_xfer("READ", dram_descs, client_vram_descs, client_name)
my_agent.transfer(handle)  # network read

handle = my_agent.initialize_xfer("WRITE", dram_descs, file_descs, self_name)
my_agent.transfer(handle)  # storage write

my_agent.send_notif(client_name, b"COMPLETE")
```

### Remote Read Request

<div className="diagram-light">
<Frame caption="Phase 4: Remote Read Request">
<img src="../figures/remote-storage/nixl_remote_storage_04_read_light.svg" alt="Remote read sequence diagram showing notification, pipelined storage read and network write, and completion" style={{maxWidth: '78%'}} />
</Frame>
</div>
<div className="diagram-dark">
<Frame caption="Phase 4: Remote Read Request">
<img src="../figures/remote-storage/nixl_remote_storage_04_read_dark.svg" alt="Remote read sequence diagram showing notification, pipelined storage read and network write, and completion" style={{maxWidth: '78%'}} />
</Frame>
</div>

The client sends a `READ` notification to the server. The server executes the reverse pipeline: **storage read** (GDS/POSIX read from local file into server DRAM) followed by **network write** (UCX write from server DRAM to client VRAM). Again, operations overlap across iterations. On completion, the server notifies the client.

```python title="Python"
# Client: send read request
descs_str = my_agent.get_serialized_descs(my_mem_descs)
my_agent.send_notif(server_name, b"READ" + iterations_str + descs_str)

# Client: wait for completion
while not my_agent.check_remote_xfer_done(server_name, b"COMPLETE"):
    continue
```

```python title="Python"
# Server: pipelined read handler (storage read → network write)
sent_descs = my_agent.deserialize_descs(received_data)

# Each iteration: read from local storage, then write to client network
# Operations are pipelined across iterations using thread pool
handle = my_agent.initialize_xfer("READ", dram_descs, file_descs, self_name)
my_agent.transfer(handle)  # storage read

handle = my_agent.initialize_xfer("WRITE", dram_descs, client_vram_descs, client_name)
my_agent.transfer(handle)  # network write

my_agent.send_notif(client_name, b"COMPLETE")
```

### Pipelining

To improve throughput, the server pipelines storage and network operations across iterations. While one iteration's network transfer is in flight, the next iteration's storage operation begins concurrently using a thread pool.

#### Read Pipeline (Storage Read → Network Write)

<div className="diagram-light">
<Frame caption="Read Pipeline: storage read overlaps with previous network write">
<img src="../figures/remote-storage/nixl_remote_storage_pipeline_read_light.svg" alt="Read pipeline sequence diagram showing overlapping storage reads and network writes across iterations" style={{maxWidth: '78%'}} />
</Frame>
</div>
<div className="diagram-dark">
<Frame caption="Read Pipeline: storage read overlaps with previous network write">
<img src="../figures/remote-storage/nixl_remote_storage_pipeline_read_dark.svg" alt="Read pipeline sequence diagram showing overlapping storage reads and network writes across iterations" style={{maxWidth: '78%'}} />
</Frame>
</div>

#### Write Pipeline (Network Read → Storage Write)

<div className="diagram-light">
<Frame caption="Write Pipeline: network read overlaps with previous storage write">
<img src="../figures/remote-storage/nixl_remote_storage_pipeline_write_light.svg" alt="Write pipeline sequence diagram showing overlapping network reads and storage writes across iterations" style={{maxWidth: '78%'}} />
</Frame>
</div>
<div className="diagram-dark">
<Frame caption="Write Pipeline: network read overlaps with previous storage write">
<img src="../figures/remote-storage/nixl_remote_storage_pipeline_write_dark.svg" alt="Write pipeline sequence diagram showing overlapping network reads and storage writes across iterations" style={{maxWidth: '78%'}} />
</Frame>
</div>

The pipeline is implemented using a `ThreadPoolExecutor` with two workers -- one for storage and one for network. The first and last iterations are special-cased (no overlap), while middle iterations submit both operations concurrently:

```python title="Python"
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    # Middle iterations: storage and network run in parallel
    executor.submit(execute_transfer, agent, dram, file, self, "READ")   # storage
    executor.submit(execute_transfer, agent, dram, vram, client, "WRITE") # network
```

## Usage

### Running as Client

```bash
python nixl_p2p_storage_example.py --role client \
                      --agents_file <file_path> \
                      --fileprefix <path_prefix> \
                      --name <agent_name> \
                      [--buf_size <size>] \
                      [--batch_size <count>]
```

The `--agents_file` is a list of storage servers the client connects to. The file should have agents separated by line, with `<agent_name> <ip_address> <port>` on each line.

The `--fileprefix` specifies a path to run local storage transfers on. The `--name` sets the Transfer Agent name for this client.

### Running as Server

```bash
python nixl_p2p_storage_example.py --role server \
                      --fileprefix <path_prefix> \
                      --name <agent_name> \
                      [--buf_size <size>] \
                      [--batch_size <count>]
```

Server names must match what is listed in the client agents file. The `buf_size` and `batch_size` must match between client and server.

## Requirements

- Python 3.6+
- NIXL library with plug-ins: GDS (optional), POSIX, UCX

## Performance Tips

- For optimal GDS performance, use the GDS_MT backend with default concurrency
- Check that your GDS setup is running true GPU-direct IO (not compatibility mode)
- For network tuning, set `UCX_MAX_RMA_RAILS=1` for VRAM-to-DRAM transfers (may need higher for larger messages)

<Tip>
For GDS configuration details, see [Environment Variables](../resources/environment-variables#gds-gpudirect-storage). For backend-specific documentation, see [NIXL Backends](/docs/user-guide/backend-selection).
</Tip>
