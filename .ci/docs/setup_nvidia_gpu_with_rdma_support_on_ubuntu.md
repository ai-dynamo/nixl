
# GPU test environment Setup

To set up an NVIDIA GPU server with RDMA support on Ubuntu, including components like MPS, `nvidia-peermem`, and GDR Copy, follow these steps. The process involves driver installation, RDMA module setup, and configuration for GPU-direct communication.

### 1. **Prerequisites and System Configuration**

- **Disable Secure Boot**: Access BIOS/UEFI during boot and disable Secure Boot to avoid driver installation issues[^1_3].
- **Install Ubuntu Server**


### 2. **Set Up RDMA Components**

- **Install MLNX_OFED**:
Download Mellanox OFED from the [official site](https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed) and install:

```bash
wget https://www.mellanox.com/downloads/DOCA/DOCA_v3.0.0/host/doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb
sudo dpkg -i doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb
sudo apt-get -y install doca-<all|ofed>
```


### 3. **Install NVIDIA Drivers**
- For GPUDirect we need to install the open kernel package

```bash
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt install nvidia-kernel-open-<version>  # e.g., nvidia-kernel-open-575
sudo reboot
```

If running on nvlink hosts like DGX we should also install fabric manager
```bash
sudo apt install nvidia-fabricmanager-<version> # should be same as kernel version nvidia-fabricmanager-575
```

Verify with `nvidia-smi`. Driver compatibility is critical for RDMA support[^1_1][^1_3].


This is required for GPUDirect RDMA[^1_1][^1_4].
- **Configure `nvidia-peermem` (replaces `nv_peer_mem`)**:
Modern NVIDIA drivers (≥ v465) include `nvidia-peermem`. Verify loading:

```bash
lsmod | grep nvidia_peermem
```

If missing, install via:

```bash
sudo apt install nvidia-peermem
sudo modprobe nvidia-peermem
```

Older systems may require deprecated `nv_peer_mem` from [GitHub](https://github.com/gpudirect/nv_peer_memory)[^1_1][^1_2][^1_4].
- **Install GDR Copy**:

```bash
git clone https://github.com/NVIDIA/gdrcopy
cd gdrcopy
make CUDA=/usr/local/cuda all install
sudo ./insmod.sh  # Load kernel module
```

Verify with `lsmod | grep gdrdrv`[^1_1][^1_5].

### ⚙️ 4. **Install GDS Components**

- **Install `nvidia-fs-dkms`** (the GDS kernel module):

```bash
sudo apt install nvidia-fs-dkms
```

This builds the `nvidia_fs` kernel module for your system.
- **Load the module**:

```bash
sudo modprobe nvidia_fs
```

- **Verify**:

```bash
lsmod | grep nvidia_fs  # Should show loaded module
```

### 📦 5. **Enable GDS in CUDA**

GDS is bundled with CUDA ≥11.4 but requires explicit enabling:

```bash
sudo echo "options nvidia NVreg_EnableGpuDirectStorage=1" > /etc/modprobe.d/nvidia-gds.conf

# Configure PeerMappingOverride for GPU-Initiated RDMA support
sudo echo "options nvidia NVreg_RegistryDwords=\"PeerMappingOverride=1;\"" > /etc/modprobe.d/nvidia.conf

sudo update-initramfs -u
sudo reboot
```

The `PeerMappingOverride=1` option is required for proper GPU peer-to-peer communication in RDMA environments.

### 6. **Enable Kernel Modules at Boot**

Add required modules to `/etc/modules`:

```bash
echo -e "nvidia\nnvidia_peermem\nnvidia_fs" | sudo tee -a /etc/modules
```

- **nvidia**: Core driver
- **nvidia_peermem**: RDMA support
- **nvidia_fs**: GPUDirect Storage


### 7. **Enable Multi-Process Service (MPS)**

MPS allows GPU sharing between processes. Configure after driver installation:

```bash
sudo nvidia-smi -i <GPU_ID> -c EXCLUSIVE_PROCESS  # Set compute mode
sudo nvidia-cuda-mps-control -d  # Start MPS daemon
```

Monitor with `nvidia-smi`. MPS reduces context-switch overhead for RDMA workloads[^1_1][^1_5].

### 8. **Docker Configuration (Optional)**

For containerized GPU workloads:

```bash
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access in containers using `docker run --gpus all nvcr.io/nvidia/pytorch:25.02-py3 nvidia-smi`[^1_3].

### 9. **Validation and Troubleshooting**

- **Check RDMA Status**:

```bash
ibv_devinfo  # Verify InfiniBand device
ib_write_bw  # Test RDMA bandwidth
```

- **GPU-NIC Alignment**:
Ensure GPU and NIC share an I/O root complex (use `lspci -tv`). Misalignment degrades RDMA performance[^1_2][^1_4].
- **Module Conflicts**:
Unload conflicting modules (e.g., `nv_peer_mem` if using `nvidia-peermem`)[^1_1][^1_4].


### Key Notes

- **Deprecation Warning**: `nv_peer_mem` is deprecated since CUDA 11.5; prefer `nvidia-peermem`[^1_1][^1_4].
- **Hardware Requirements**: NVIDIA ConnectX-3+ NICs and compatible GPUs[^1_4][^1_5].
- **Performance**: For optimal GPUDirect RDMA, use GPUs and NICs on the same PCIe root complex[^1_2][^1_5].

This setup enables direct GPU-NIC communication, bypassing host memory for low-latency RDMA transfers. Cite NVIDIA documentation for version-specific nuances[^1_1][^1_4][^1_5].


[^1_1]: https://docs.nvidia.com/networking/display/GPUDirectRDMAv18/Installing+GPUDirect+RDMA

[^1_2]: https://github.com/gpudirect/nv_peer_memory

[^1_3]: https://gist.github.com/kengz/a106e03a782cfaec339433daf8965d76

[^1_4]: https://docs.nvidia.com/cuda/gpudirect-rdma/

[^1_5]: http://hidl.cse.ohio-state.edu/userguide/gdr/

[^1_6]: https://github.com/openucx/ucx/blob/master/docs/source/faq.md

[^1_7]: https://conference.eresearch.edu.au/wp-content/uploads/2019/09/2019-eResearch_149_Testing-GPUDirect.pdf

[^1_8]: https://forums.developer.nvidia.com/t/internode-nvshmme-and-ib-problem/286552

[^1_9]: https://docs.nvidia.com/doca/archive/doca-v1.5.3/installation-guide-for-linux/index.html

[^1_10]: https://network.nvidia.com/pdf/prod_software/Ubuntu_20_04_Inbox_Driver_User_Manual.pdf

