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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_DC_DESCRIPTOR_PROVIDER_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_DC_DESCRIPTOR_PROVIDER_H

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include "dc_descriptor.h"
#include "rdma_descriptor_provider.h"

/// Maximum DCT QPs per NIC (one per physical LAG/bond port).
static constexpr int dc_max_lag_ports = 4;

/**
 * In-process libibverbs DC (Dynamically Connected) RDMA descriptor provider.
 *
 * Opens one DC target per NIC and registers every buffer on all of them, so a
 * request against any buffer can travel on any rail. An rkey covers any
 * sub-range of its MR, so the NIC a request uses is picked when its RDMA
 * descriptor is built: see pickRail. VRAM requests use only the NICs affine to
 * their GPU.
 *
 * The DC target is passive: the server (DCI side) initiates and performs all
 * one-sided RDMA, so this client runs no background threads, CM listener, or CQ
 * polling.
 */
class dcDescriptorProvider : public iRdmaDescriptorProvider {
public:
    /**
     * @param nics   NIC specifiers, IPv4 addresses or RDMA device names (e.g.
     *               mlx5_1); one DC target is opened per entry.
     * @param dc_key DC access key the server's DCI side must present.
     * @param sl     RoCE service level (0-15) of what the DC target sends.
     * @param traffic_class RoCE traffic class (ToS byte, DSCP = value >> 2).
     */
    dcDescriptorProvider(const std::vector<std::string> &nics,
                         uint64_t dc_key,
                         uint8_t sl,
                         uint8_t traffic_class);
    ~dcDescriptorProvider() override;

    dcDescriptorProvider(const dcDescriptorProvider &) = delete;
    dcDescriptorProvider &
    operator=(const dcDescriptorProvider &) = delete;

    bool
    isConnected() const override;
    nixl_status_t
    registerMemory(void *ptr, size_t size, int dev_id) override;
    nixl_status_t
    deregisterMemory(void *ptr) override;
    std::string
    makeDescriptor(void *ptr, size_t size) override;

private:
    /// Per-NIC DC target context, created once at construction.
    struct nicCtx {
        ibv_context *ctx = nullptr;
        ibv_pd *pd = nullptr;
        ibv_cq *cq = nullptr;
        ibv_srq *srq = nullptr;
        int numLagPorts = 1;
        ibv_qp *dctQps[dc_max_lag_ports] = {};
        uint32_t dctns[dc_max_lag_ports] = {};
        uint16_t lid = 0;
        uint8_t gid[16] = {};
        int gidIndex = 0;
        uint64_t lagSeq = 0; ///< round-robin cursor over this NIC's LAG ports
        uint64_t maxMrSize = 0; ///< largest MR the device accepts, 0 if unknown
        std::string devName; ///< ib device name (e.g. mlx5_1), for logging
        std::string pciPath; ///< canonical /sys/devices/... PCIe path of the NIC
        int numaNode = -1; ///< NUMA node of the NIC, or -1 if unknown
    };

    /// One rail of a registered buffer: an MR covering the WHOLE buffer on that
    /// NIC. An rkey is valid for any sub-range of its MR, so a request at any
    /// offset can be described from any rail the buffer is registered on.
    struct railMr {
        ibv_mr *mr = nullptr;
        int nicIdx = 0;
    };

    /// A caller registration, held on every NIC that accepted it.
    struct buffer {
        size_t len = 0;
        int devId = -1; ///< GPU ordinal (VRAM) or -1 (host memory)
        int refs = 1; ///< identical registrations sharing these MRs
        std::vector<railMr> rails;
    };

    /// Build (and INIT->RTR) a NIC's DC context. Returns false on error.
    bool
    setupNic(const std::string &nic_spec, uint64_t dc_key, nicCtx &nic);
    /// Release whatever setupNic created for a NIC, including after a failure.
    static void
    releaseNic(nicCtx &nic);
    /// Detect bond width for an IB device via sysfs; clamps to [1, dc_max_lag_ports].
    static int
    numLagPorts(const char *dev_name);
    /// Index into buf.rails for the next request against buf: the rail whose NIC
    /// has taken the fewest requests, among the NICs affine to the buffer's GPU for
    /// VRAM (all rails if none of them holds the buffer), among all rails for host
    /// memory. Caller must hold mu_.
    size_t
    pickRail(const buffer &buf);
    /// Register [ptr, ptr+size) on one NIC. Returns nullptr on failure.
    ibv_mr *
    registerRail(void *ptr, size_t size, int nic_idx);
    /// Resolve (and cache) the set of NIC indices affine to a GPU. Prefers NICs
    /// that share a PCIe switch with the GPU (PXB/PIX-local); when no NIC is
    /// switch-local (the RDMA NICs sit on a shared bridge equidistant from every
    /// GPU in the node), keeps all same-NUMA NICs so requests spread across both
    /// rails. Falls back to all NICs when no topology info is available.
    /// Caller must hold mu_.
    const std::vector<int> &
    affineNicsFor(int dev_id);
    /// One-shot NIXL_INFO dump of the registered MR layout. Caller holds mu_.
    void
    logLayout();
    /// NIXL_INFO summary of how requests were spread over the rails, and what
    /// registration cost. `detailed` adds the per-GPU breakdown. Caller holds mu_.
    void
    logRequestSpread(bool detailed);

    std::vector<nicCtx> nics_;
    bool connected_ = false;
    /// All NIC indices [0, nics_.size()); the affinity fallback.
    std::vector<int> allNics_;
    /// dev_id -> affine NIC indices, resolved once per GPU.
    std::map<int, std::vector<int>> gpuAffineNics_;
    /// Requests handed out per NIC; pickRail favours the least used.
    std::vector<uint64_t> nicIssued_;
    /// Requests that left the owning GPU's affine rails.
    uint64_t crossNuma_ = 0;
    /// Requests since the last spread summary.
    uint64_t sinceSpreadLog_ = 0;
    /// dev_id -> requests sent down each NIC, for the teardown dump.
    std::map<int, std::vector<uint64_t>> gpuNicRequests_;
    /// dev_id -> NUMA node, cached alongside gpuAffineNics_ for logging.
    std::map<int, int> gpuNuma_;
    /// Guard so the MR layout is dumped once, at the first transfer.
    bool layoutLogged_ = false;
    // Registration cost, reported with the rail summary.
    std::size_t regCalls_ = 0;
    std::size_t regRails_ = 0;
    std::size_t regBytes_ = 0;
    std::size_t regUs_ = 0;
    std::size_t deregUs_ = 0;
    /// RoCE service level and traffic class for the DCT address vector. They mark
    /// what this side sends (RDMA read responses and ACKs); the server marks what
    /// it sends itself.
    uint8_t sl_;
    uint8_t trafficClass_;
    mutable std::mutex mu_;
    /// base address -> buffer, ordered so a sub-address can be found by range.
    std::map<uintptr_t, buffer> buffers_;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_DC_DESCRIPTOR_PROVIDER_H
