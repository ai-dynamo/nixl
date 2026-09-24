/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dc_descriptor_provider.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "common/config_traits.h"
#include "common/nixl_log.h"
#include "common/str_util.h"

namespace {

constexpr int dc_port = 1;

/// Count the space-separated entries in a bonding "slaves" line (the contents
/// of /sys/class/net/<bond>/bonding/slaves). Returns the slave count, or 0 for an
/// empty line. Callers clamp to [1, dc_max_lag_ports].
int
countBondSlaves(const char *slaves_line) {
    if (slaves_line == nullptr) {
        return 0;
    }
    return static_cast<int>(nixl::str::splitStripped(slaves_line, ' ').size());
}

/// Canonicalize a sysfs symlink to its /sys/devices/... target. Empty on failure.
std::string
canonicalPath(const std::string &path) {
    char resolved[PATH_MAX];
    if (realpath(path.c_str(), resolved) == nullptr) {
        return "";
    }
    return std::string(resolved);
}

/// Read a single integer from a sysfs file; returns fallback on any failure.
int
readIntFile(const std::string &path, int fallback = -1) {
    std::ifstream f(path);
    int v = fallback;
    if (f && (f >> v)) {
        return v;
    }
    return fallback;
}

/// Number of shared leading '/'-separated components between two paths. Used as
/// a PCIe-proximity score: a longer shared prefix means a closer shared bridge.
size_t
commonPathPrefix(const std::string &a, const std::string &b) {
    if (a.empty() || b.empty()) {
        return 0;
    }
    std::stringstream sa(a), sb(b);
    std::string ca, cb;
    size_t n = 0;
    while (std::getline(sa, ca, '/') && std::getline(sb, cb, '/')) {
        if (ca != cb) {
            break;
        }
        n++;
    }
    return n;
}

/// The CUDA driver API calls the provider needs, resolved at run time from
/// libcuda.so.1 so the connector has no build-time CUDA dependency. Any process
/// holding VRAM has the driver loaded already. When the library or a symbol is
/// missing, ok is false and GPU/NIC affinity degrades to every NIC being affine.
struct cudaDriverApi {
    using cu_init_fn_t = int (*)(unsigned int);
    using cu_device_get_fn_t = int (*)(int *, int);
    using cu_device_get_pci_bus_id_fn_t = int (*)(char *, int, int);

    cu_init_fn_t init = nullptr;
    cu_device_get_fn_t deviceGet = nullptr;
    cu_device_get_pci_bus_id_fn_t deviceGetPciBusId = nullptr;
    bool ok = false;
};

const cudaDriverApi &
cudaDriver() {
    static const cudaDriverApi api = [] {
        cudaDriverApi a;
        // Never closed: the driver stays loaded for the life of the process.
        void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            NIXL_INFO << "ibverbs_dc: libcuda.so.1 not loadable (" << dlerror()
                      << "); no GPU/NIC affinity";
            return a;
        }
        a.init = reinterpret_cast<cudaDriverApi::cu_init_fn_t>(dlsym(lib, "cuInit"));
        a.deviceGet =
            reinterpret_cast<cudaDriverApi::cu_device_get_fn_t>(dlsym(lib, "cuDeviceGet"));
        a.deviceGetPciBusId = reinterpret_cast<cudaDriverApi::cu_device_get_pci_bus_id_fn_t>(
            dlsym(lib, "cuDeviceGetPCIBusId"));
        a.ok = a.init && a.deviceGet && a.deviceGetPciBusId && a.init(0) == 0;
        if (!a.ok) {
            NIXL_WARN << "ibverbs_dc: CUDA driver API unavailable; no GPU/NIC affinity";
        }
        return a;
    }();
    return api;
}

/// PCI bus id ("0000:1b:00.0", lower-case) for a CUDA device. Empty on error.
/// Stable identity for the physical GPU under CUDA_VISIBLE_DEVICES, which
/// renumbers the visible ordinals (the driver API numbers them like the runtime);
/// maps directly to `nvidia-smi topo`.
std::string
gpuBusId(int dev_id) {
    const cudaDriverApi &cu = cudaDriver();
    int dev = 0;
    char busid[32] = {};
    if (!cu.ok || cu.deviceGet(&dev, dev_id) != 0 ||
        cu.deviceGetPciBusId(busid, sizeof(busid), dev) != 0) {
        return "";
    }
    // The driver yields upper-case "0000:1B:00.0"; sysfs uses lower-case.
    for (char *p = busid; *p; ++p) {
        *p = static_cast<char>(std::tolower(static_cast<unsigned char>(*p)));
    }
    return busid;
}

/// Canonical /sys PCIe path for a CUDA device, via its PCI bus id. Empty on error.
std::string
gpuPciPath(int dev_id) {
    const std::string busid = gpuBusId(dev_id);
    if (busid.empty()) {
        return "";
    }
    return canonicalPath(std::string("/sys/bus/pci/devices/") + busid);
}

/// Map a runtime-visible GPU ordinal back to the id the user set: the entry at
/// position dev_id in CUDA_VISIBLE_DEVICES (which renumbers visible devices to
/// 0..N). Falls back to dev_id when the variable is unset or the entry is not a
/// plain integer (e.g. a UUID or MIG handle).
int
visibleToUserGpu(int dev_id) {
    const char *cvd = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!cvd || dev_id < 0) {
        return dev_id;
    }
    const std::vector<std::string> ids = nixl::str::splitStripped(cvd);
    if (static_cast<size_t>(dev_id) >= ids.size()) {
        return dev_id;
    }
    try {
        const int id = nixl::config::configTraits<int>::convert(ids[dev_id]);
        return id >= 0 ? id : dev_id;
    }
    catch (const std::runtime_error &) {
        return dev_id; // not an integer: a UUID or a MIG handle
    }
}

/// RoCE GID type via sysfs: 2 (RoCEv2), 1 (RoCEv1), 0 (IB), -1 on error.
int
gidTypeSysfs(const char *dev_name, int gid_index) {
    char path[256];
    snprintf(path,
             sizeof(path),
             "/sys/class/infiniband/%s/ports/1/gid_attrs/types/%d",
             dev_name,
             gid_index);
    FILE *f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    char buf[32];
    int got = (fgets(buf, sizeof(buf), f) != nullptr);
    fclose(f);
    if (!got) {
        return -1;
    }
    if (strstr(buf, "RoCE v2")) {
        return 2;
    }
    if (strstr(buf, "v1")) {
        return 1;
    }
    return 0;
}

/// True if the GID is an IPv4-mapped address (::ffff:a.b.c.d), the routable
/// RoCEv2 GID the server expects, as opposed to a link-local fe80:: GID.
bool
isIpv4MappedGid(const uint8_t raw[16]) {
    static const uint8_t prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    return memcmp(raw, prefix, 12) == 0;
}

/**
 * Open the RDMA device for a NIC specifier, which is either an IPv4 address or an
 * RDMA device name (e.g. "mlx5_1"):
 *  - IPv4: match the device/GID whose IPv4-mapped GID (::ffff:<addr>) equals it.
 *  - device name: open that device and select its best RoCEv2 GID.
 * Among candidates, prefer the highest GID type (RoCEv2 > RoCEv1 > IB).
 */
ibv_context *
openDeviceForNic(const std::string &nic, int *out_gid_index) {
    *out_gid_index = 0;

    int num_devices = 0;
    ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        NIXL_ERROR << "ibverbs_dc: no RDMA devices found";
        if (dev_list) {
            ibv_free_device_list(dev_list);
        }
        return nullptr;
    }

    in_addr v4{};
    const bool is_ip = (inet_aton(nic.c_str(), &v4) != 0);
    uint8_t target_gid[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0, 0, 0, 0};
    if (is_ip) {
        memcpy(target_gid + 12, &v4, 4);
    }

    // Below every score: gidTypeSysfs() returns -1 when sysfs cannot be read.
    int best_dev = -1, best_gidx = 0, best_score = std::numeric_limits<int>::min();
    for (int d = 0; d < num_devices; d++) {
        const char *dev_name = ibv_get_device_name(dev_list[d]);
        // For a device-name specifier, only consider that device.
        if (!is_ip && nic != dev_name) {
            continue;
        }
        ibv_context *ctx = ibv_open_device(dev_list[d]);
        if (!ctx) {
            continue;
        }
        ibv_port_attr port_attr{};
        if (ibv_query_port(ctx, dc_port, &port_attr) != 0) {
            ibv_close_device(ctx);
            continue;
        }
        for (int g = 0; g < port_attr.gid_tbl_len; g++) {
            ibv_gid gid;
            if (ibv_query_gid(ctx, dc_port, g, &gid) != 0) {
                continue;
            }
            // IPv4 specifier: the GID must match. Device-name specifier: take the
            // device's best GID (skip the all-zero/unset entries).
            if (is_ip) {
                if (memcmp(gid.raw, target_gid, 16) != 0) {
                    continue;
                }
            } else {
                ibv_gid zero{};
                if (memcmp(gid.raw, zero.raw, 16) == 0) {
                    continue;
                }
            }
            // Prefer RoCEv2, then the routable IPv4-mapped GID over a link-local
            // fe80:: one (both are RoCEv2; the server can only route the former).
            int gtype = gidTypeSysfs(dev_name, g);
            int score = gtype * 2 + (isIpv4MappedGid(gid.raw) ? 1 : 0);
            if (score > best_score) {
                best_dev = d;
                best_gidx = g;
                best_score = score;
            }
        }
        ibv_close_device(ctx);
    }

    ibv_context *result = nullptr;
    if (best_dev >= 0) {
        result = ibv_open_device(dev_list[best_dev]);
        if (result) {
            *out_gid_index = best_gidx;
            NIXL_DEBUG << "ibverbs_dc: NIC " << nic << " -> "
                       << ibv_get_device_name(dev_list[best_dev]) << " gid_index=" << best_gidx
                       << " (score " << best_score << ")";
        }
    } else {
        NIXL_ERROR << "ibverbs_dc: no RDMA device found for NIC '" << nic
                   << "' (expected an IPv4 address or device name like mlx5_1)";
    }

    ibv_free_device_list(dev_list);
    return result;
}

} // namespace

int
dcDescriptorProvider::numLagPorts(const char *dev_name) {
    // /sys/class/infiniband/<dev>/device/net/<iface>
    char net_dir[256];
    snprintf(net_dir, sizeof(net_dir), "/sys/class/infiniband/%s/device/net", dev_name);
    DIR *d = opendir(net_dir);
    if (!d) {
        return 1;
    }
    std::string iface;
    for (dirent *ent = readdir(d); ent != nullptr; ent = readdir(d)) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        iface = ent->d_name;
        break;
    }
    closedir(d);
    if (iface.empty()) {
        return 1;
    }

    // /sys/class/net/<iface>/master -> bond
    char master_link[256];
    snprintf(master_link, sizeof(master_link), "/sys/class/net/%s/master", iface.c_str());
    char master_target[256];
    ssize_t len = readlink(master_link, master_target, sizeof(master_target) - 1);
    if (len < 0) {
        return 1; // not a bond slave
    }
    master_target[len] = '\0';
    char *bond = strrchr(master_target, '/');
    bond = bond ? bond + 1 : master_target;

    // /sys/class/net/<bond>/bonding/slaves
    char slaves_path[512];
    snprintf(slaves_path, sizeof(slaves_path), "/sys/class/net/%s/bonding/slaves", bond);
    FILE *f = fopen(slaves_path, "r");
    if (!f) {
        return 1;
    }
    char buf[256];
    int got = (fgets(buf, sizeof(buf), f) != nullptr);
    fclose(f);
    if (!got) {
        return 1;
    }

    int n = countBondSlaves(buf);
    if (n < 1 || n > dc_max_lag_ports) {
        NIXL_WARN << "ibverbs_dc: bond '" << bond << "' has " << n << " slave(s), out of range [1,"
                  << dc_max_lag_ports << "], using 1";
        return 1;
    }
    NIXL_DEBUG << "ibverbs_dc: bond '" << bond << "' has " << n << " slave(s) (iface " << iface
               << ")";
    return n;
}

bool
dcDescriptorProvider::setupNic(const std::string &nic_spec, uint64_t dc_key, nicCtx &nic) {
    nic.ctx = openDeviceForNic(nic_spec, &nic.gidIndex);
    if (!nic.ctx) {
        return false;
    }

    nic.devName = ibv_get_device_name(nic.ctx->device);
    nic.numLagPorts = numLagPorts(nic.devName.c_str());
    NIXL_DEBUG << "ibverbs_dc: NIC " << nic_spec << " LAG ports: " << nic.numLagPorts;

    // PCIe topology for GPU/NIC affinity. Advisory only: a failure here degrades
    // to no affinity, never blocks.
    const std::string ib_dev = "/sys/class/infiniband/" + nic.devName + "/device";
    nic.pciPath = canonicalPath(ib_dev);
    nic.numaNode = readIntFile(ib_dev + "/numa_node");
    NIXL_DEBUG << "ibverbs_dc: NIC " << nic_spec << " (" << nic.devName << ") pci_path='"
               << nic.pciPath << "' numa_node=" << nic.numaNode;

    ibv_device_attr dev_attr{};
    if (ibv_query_device(nic.ctx, &dev_attr) == 0) {
        nic.maxMrSize = dev_attr.max_mr_size;
    } else {
        NIXL_WARN << "ibverbs_dc: ibv_query_device failed for " << nic_spec;
    }

    nic.pd = ibv_alloc_pd(nic.ctx);
    if (!nic.pd) {
        NIXL_ERROR << "ibverbs_dc: ibv_alloc_pd failed for " << nic_spec;
        return false;
    }

    nic.cq = ibv_create_cq(nic.ctx, 2 * nic.numLagPorts, nullptr, nullptr, 0);
    if (!nic.cq) {
        NIXL_ERROR << "ibverbs_dc: ibv_create_cq failed for " << nic_spec;
        return false;
    }

    ibv_srq_init_attr srq_attr{};
    srq_attr.attr.max_wr = 1;
    srq_attr.attr.max_sge = 1;
    nic.srq = ibv_create_srq(nic.pd, &srq_attr);
    if (!nic.srq) {
        NIXL_ERROR << "ibverbs_dc: ibv_create_srq failed for " << nic_spec;
        return false;
    }

    ibv_port_attr port_attr{};
    if (ibv_query_port(nic.ctx, dc_port, &port_attr)) {
        NIXL_ERROR << "ibverbs_dc: ibv_query_port failed for " << nic_spec;
        return false;
    }
    nic.lid = port_attr.lid; // 0 on RoCEv2
    // 4096 is the largest RDMA MTU; a 1500-byte Ethernet MTU makes the port's 1024.
    const ibv_mtu path_mtu =
        (port_attr.active_mtu < IBV_MTU_4096) ? port_attr.active_mtu : IBV_MTU_4096;

    for (int p = 0; p < nic.numLagPorts; p++) {
        ibv_qp_init_attr_ex qp_attr{};
        qp_attr.qp_type = IBV_QPT_DRIVER;
        qp_attr.send_cq = nic.cq;
        qp_attr.recv_cq = nic.cq;
        qp_attr.srq = nic.srq;
        qp_attr.pd = nic.pd;
        qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD;
        qp_attr.cap.max_send_wr = 1;
        qp_attr.cap.max_recv_wr = 1;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;

        mlx5dv_qp_init_attr mlx5_attr{};
        mlx5_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
        mlx5_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
        mlx5_attr.dc_init_attr.dct_access_key = dc_key;

        nic.dctQps[p] = mlx5dv_create_qp(nic.ctx, &qp_attr, &mlx5_attr);
        if (!nic.dctQps[p]) {
            NIXL_ERROR << "ibverbs_dc: mlx5dv_create_qp (DCT) failed for " << nic_spec
                       << " lag_port " << (p + 1);
            return false;
        }

        ibv_qp_attr mod{};
        mod.qp_state = IBV_QPS_INIT;
        mod.port_num = dc_port;
        mod.pkey_index = 0;
        mod.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(nic.dctQps[p],
                          &mod,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            NIXL_ERROR << "ibverbs_dc: DCT QP -> INIT failed for " << nic_spec;
            return false;
        }

        memset(&mod, 0, sizeof(mod));
        mod.qp_state = IBV_QPS_RTR;
        mod.path_mtu = path_mtu;
        mod.ah_attr.sl = sl_;
        mod.ah_attr.port_num = dc_port;
        mod.ah_attr.is_global = 1;
        mod.ah_attr.grh.sgid_index = nic.gidIndex;
        mod.ah_attr.grh.hop_limit = 64;
        mod.ah_attr.grh.traffic_class = trafficClass_;
        mod.min_rnr_timer = 12;
        if (ibv_modify_qp(nic.dctQps[p],
                          &mod,
                          IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_MIN_RNR_TIMER)) {
            NIXL_ERROR << "ibverbs_dc: DCT QP -> RTR failed for " << nic_spec;
            return false;
        }

        nic.dctns[p] = nic.dctQps[p]->qp_num;
        NIXL_DEBUG << "ibverbs_dc: NIC " << nic_spec << " DCT QP[" << p
                   << "] dctn=" << nic.dctns[p];
    }

    ibv_gid gid;
    if (ibv_query_gid(nic.ctx, dc_port, nic.gidIndex, &gid)) {
        NIXL_ERROR << "ibverbs_dc: ibv_query_gid failed for " << nic_spec;
        return false;
    }
    memcpy(nic.gid, gid.raw, 16);
    return true;
}

dcDescriptorProvider::dcDescriptorProvider(const std::vector<std::string> &nics,
                                           uint64_t dc_key,
                                           uint8_t sl,
                                           uint8_t traffic_class)
    : sl_(sl),
      trafficClass_(traffic_class) {
    if (nics.empty()) {
        NIXL_ERROR << "ibverbs_dc: no NICs provided";
        return;
    }
    // A NIC that cannot be set up is left out; the others still carry traffic.
    for (const std::string &spec : nics) {
        nicCtx nic;
        if (!setupNic(spec, dc_key, nic)) {
            NIXL_ERROR << "ibverbs_dc: failed to set up NIC " << spec << "; continuing without it";
            releaseNic(nic);
            continue;
        }
        nics_.push_back(std::move(nic));
    }
    if (nics_.empty()) {
        NIXL_ERROR << "ibverbs_dc: none of the " << nics.size() << " NIC(s) could be set up";
        return;
    }
    nicIssued_.assign(nics_.size(), 0);
    allNics_.resize(nics_.size());
    for (size_t i = 0; i < nics_.size(); i++) {
        allNics_[i] = static_cast<int>(i);
    }
    connected_ = true;
    NIXL_INFO << "ibverbs_dc: DC transport ready across " << nics_.size() << " NIC(s), key=0x"
              << std::hex << std::noshowbase << dc_key << std::dec << ", RoCE sl=" << (int)sl_
              << " traffic_class=" << (int)trafficClass_ << " (DSCP " << (trafficClass_ >> 2)
              << ")";
}

dcDescriptorProvider::~dcDescriptorProvider() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto &kv : buffers_) {
            for (auto &rail : kv.second.rails) {
                if (rail.mr) {
                    ibv_dereg_mr(rail.mr);
                }
            }
        }
        buffers_.clear();
    }
    for (auto &nic : nics_) {
        releaseNic(nic);
    }
    nics_.clear();
}

void
dcDescriptorProvider::releaseNic(nicCtx &nic) {
    for (int p = 0; p < dc_max_lag_ports; p++) {
        if (nic.dctQps[p]) {
            ibv_destroy_qp(nic.dctQps[p]);
            nic.dctQps[p] = nullptr;
        }
    }
    if (nic.srq) {
        ibv_destroy_srq(nic.srq);
        nic.srq = nullptr;
    }
    if (nic.cq) {
        ibv_destroy_cq(nic.cq);
        nic.cq = nullptr;
    }
    if (nic.pd) {
        ibv_dealloc_pd(nic.pd);
        nic.pd = nullptr;
    }
    if (nic.ctx) {
        ibv_close_device(nic.ctx);
        nic.ctx = nullptr;
    }
}

bool
dcDescriptorProvider::isConnected() const {
    return connected_;
}

const std::vector<int> &
dcDescriptorProvider::affineNicsFor(int dev_id) {
    auto it = gpuAffineNics_.find(dev_id);
    if (it != gpuAffineNics_.end()) {
        return it->second;
    }

    const std::string gpu_path = gpuPciPath(dev_id);
    const int gpu_numa = gpu_path.empty() ? -1 : readIntFile(gpu_path + "/numa_node");

    // Candidate rails: same-NUMA NICs (avoids the cross-socket SYS path). If NUMA
    // is unknown, every NIC is a candidate.
    std::vector<int> candidates;
    if (gpu_numa >= 0) {
        for (size_t i = 0; i < nics_.size(); ++i) {
            if (nics_[i].numaNode == gpu_numa) {
                candidates.push_back(static_cast<int>(i));
            }
        }
    }
    if (candidates.empty()) {
        candidates = allNics_;
    }

    // Narrow to PCIe switch-local NICs, but ONLY when the locality is real. A NIC
    // is switch-local to the GPU when the GPU's PCIe path reaches into the NIC's
    // switch, i.e. the GPU shares MORE of the path with that NIC than the NIC's
    // same-switch siblings share with each other. On hardware where the RDMA NICs
    // sit on a shared bridge equidistant (NODE) from every GPU in the node, no NIC
    // is switch-local, so all same-NUMA rails stay in the set. On hardware where
    // each GPU has a dedicated (PXB/PIX) NIC, that one rail wins.
    std::vector<int> affine;
    if (!gpu_path.empty() && candidates.size() > 1) {
        int best = candidates.front();
        size_t best_prefix = commonPathPrefix(gpu_path, nics_[best].pciPath);
        for (int idx : candidates) {
            size_t p = commonPathPrefix(gpu_path, nics_[idx].pciPath);
            if (p > best_prefix) {
                best_prefix = p;
                best = idx;
            }
        }
        // How deep the best NIC's switch neighborhood extends among the candidates.
        size_t sibling_prefix = 0;
        for (int idx : candidates) {
            if (idx == best) {
                continue;
            }
            sibling_prefix =
                std::max(sibling_prefix, commonPathPrefix(nics_[best].pciPath, nics_[idx].pciPath));
        }
        // GPU reaches past where the siblings diverge -> genuinely switch-local;
        // keep every candidate at that deepest tier and drop the rest.
        if (best_prefix > sibling_prefix) {
            for (int idx : candidates) {
                if (commonPathPrefix(gpu_path, nics_[idx].pciPath) == best_prefix) {
                    affine.push_back(idx);
                }
            }
        }
    }

    // Not switch-local (equidistant NODE-level rails) or NUMA-only: use them all.
    if (affine.empty()) {
        affine = candidates;
    }

    if (gpu_path.empty() && dev_id >= 0) {
        NIXL_WARN << "ibverbs_dc: no PCIe path for GPU " << visibleToUserGpu(dev_id)
                  << "; every NIC treated as affine";
    } else if (gpu_numa < 0 && dev_id >= 0) {
        NIXL_WARN << "ibverbs_dc: GPU " << visibleToUserGpu(dev_id)
                  << " reports no NUMA node; every NIC treated as affine";
    }

    auto res = gpuAffineNics_.emplace(dev_id, std::move(affine));
    return res.first->second;
}

size_t
dcDescriptorProvider::pickRail(const buffer &buf) {
    // VRAM uses only the rails affine to its GPU. Host memory has no affinity, and
    // a buffer none of whose affine NICs took the registration uses every rail.
    const std::vector<int> *affine = nullptr;
    if (buf.devId >= 0) {
        affine = &affineNicsFor(buf.devId);
    }
    auto is_affine = [&](int nic_idx) {
        return std::find(affine->begin(), affine->end(), nic_idx) != affine->end();
    };
    const bool strict =
        affine && std::any_of(buf.rails.begin(), buf.rails.end(), [&](const railMr &r) {
            return is_affine(r.nicIdx);
        });

    size_t best = buf.rails.size();
    for (size_t i = 0; i < buf.rails.size(); ++i) {
        const int nic_idx = buf.rails[i].nicIdx;
        if (strict && !is_affine(nic_idx)) {
            continue;
        }
        if (best == buf.rails.size() || nicIssued_[nic_idx] < nicIssued_[buf.rails[best].nicIdx]) {
            best = i;
        }
    }
    return best;
}

ibv_mr *
dcDescriptorProvider::registerRail(void *ptr, size_t size, int nic_idx) {
    const nicCtx &nic = nics_[nic_idx];
    if (nic.maxMrSize != 0 && size > nic.maxMrSize) {
        NIXL_ERROR << "ibverbs_dc: " << size << " bytes exceeds the largest MR " << nic.devName
                   << " accepts (" << nic.maxMrSize << ")";
        return nullptr;
    }
    // Relaxed ordering lets the NIC pipeline the PCIe writes into GPU BAR instead
    // of serializing them. Without it, GET writes into VRAM collapse under
    // multi-GPU concentration on a single dual-port card: the card cannot drain
    // the strict-ordered writes fast enough, asserts PFC pause, and throughput
    // falls below even a single GPU. It is an optional access flag: drivers that
    // don't support it silently ignore it.
    unsigned int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING;
    ibv_mr *mr = ibv_reg_mr(nic.pd, ptr, size, access);
    if (!mr) {
        NIXL_ERROR << "ibverbs_dc: ibv_reg_mr failed on " << nic.devName << " (ptr=" << ptr
                   << ", size=" << size << ")";
    }
    return mr;
}

nixl_status_t
dcDescriptorProvider::registerMemory(void *ptr, size_t size, int dev_id) {
    if (!connected_) {
        return NIXL_ERR_BACKEND;
    }

    std::lock_guard<std::mutex> lk(mu_);

    // The same buffer may be registered more than once (e.g. as two descriptors):
    // those registrations share its MRs, and each deregistration releases one.
    auto existing = buffers_.find(reinterpret_cast<uintptr_t>(ptr));
    if (existing != buffers_.end()) {
        if (existing->second.len != size) {
            NIXL_ERROR << "ibverbs_dc: " << ptr << " is already registered with "
                       << existing->second.len << " bytes, not " << size;
            return NIXL_ERR_INVALID_PARAM;
        }
        existing->second.refs++;
        return NIXL_SUCCESS;
    }

    // One MR per rail covering the whole buffer: an rkey covers any sub-range, so
    // the rail is chosen per request in makeDescriptor, not here.
    buffer buf;
    buf.len = size;
    buf.devId = dev_id;
    for (int nic_idx : allNics_) {
        if (ibv_mr *mr = registerRail(ptr, size, nic_idx)) {
            buf.rails.push_back({mr, nic_idx});
        }
    }
    if (buf.rails.empty()) {
        NIXL_ERROR << "ibverbs_dc: no NIC accepted the registration of " << size << " bytes at "
                   << ptr;
        return NIXL_ERR_BACKEND;
    }
    if (buf.rails.size() < nics_.size()) {
        NIXL_WARN << "ibverbs_dc: " << ptr << " registered on " << buf.rails.size() << " of "
                  << nics_.size() << " rails; the rest stay unreachable for this buffer";
    }

    NIXL_DEBUG << "ibverbs_dc: registered 0x" << std::hex << reinterpret_cast<uintptr_t>(ptr)
               << std::dec << " (" << size << " bytes, dev_id=" << dev_id << ") on "
               << buf.rails.size() << " rail(s)";
    buffers_[reinterpret_cast<uintptr_t>(ptr)] = std::move(buf);
    return NIXL_SUCCESS;
}

nixl_status_t
dcDescriptorProvider::deregisterMemory(void *ptr) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = buffers_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == buffers_.end()) {
        NIXL_ERROR << "ibverbs_dc: deregisterMemory: ptr " << ptr << " not registered";
        return NIXL_ERR_NOT_FOUND;
    }
    if (--it->second.refs > 0) {
        return NIXL_SUCCESS;
    }
    for (auto &rail : it->second.rails) {
        if (rail.mr) {
            ibv_dereg_mr(rail.mr);
        }
    }
    buffers_.erase(it);
    return NIXL_SUCCESS;
}

std::string
dcDescriptorProvider::makeDescriptor(void *ptr, size_t size) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lk(mu_);

    if (size > UINT32_MAX) {
        NIXL_ERROR << "ibverbs_dc: request size " << size
                   << " exceeds the 32-bit SIZE field of a DC RDMA descriptor";
        return std::string();
    }

    // Find a buffer [base, base+len) containing [addr, addr+size). The nearest
    // base at or below addr normally covers it; when registrations are nested
    // (a pool and a sub-range of it), the enclosing one lies further back.
    const buffer *found = nullptr;
    for (auto it = buffers_.upper_bound(addr); it != buffers_.begin();) {
        --it;
        if (addr + size <= it->first + it->second.len) {
            found = &it->second;
            break;
        }
    }
    if (!found) {
        NIXL_ERROR << "ibverbs_dc: no registration covers ptr " << ptr << " size " << size;
        return std::string();
    }
    const buffer &buf = *found;

    const railMr &rail = buf.rails[pickRail(buf)];
    nicCtx &nic = nics_[rail.nicIdx];
    nicIssued_[rail.nicIdx]++;

    return formatDcDescriptor(static_cast<uint64_t>(addr),
                              static_cast<uint32_t>(size),
                              rail.mr->rkey,
                              nic.lid,
                              nic.dctns[nic.lagSeq++ % nic.numLagPorts],
                              nic.gid);
}
