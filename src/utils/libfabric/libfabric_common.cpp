/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

#include "libfabric_common.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <atomic>
#include <cstring>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#include <numa.h>

namespace LibfabricUtils {

namespace {

    /**
     * @brief Domain-name suffixes verbs uses for its XRC and DGRAM variants, which nixl rails
     * cannot use. Matched as suffixes so hyphenated device names are kept.
     */
    constexpr const char *k_verbs_unusable_domain_suffixes[] = {"-xrc", "-dgram"};

    bool
    hasUnusableVerbsSuffix(const std::string &domain) {
        for (const char *suffix : k_verbs_unusable_domain_suffixes) {
            const size_t suffix_len = strlen(suffix);
            if (domain.size() > suffix_len &&
                domain.compare(domain.size() - suffix_len, suffix_len, suffix) == 0) {
                return true;
            }
        }
        return false;
    }

} // namespace

std::string
baseProviderName(const std::string &prov_name) {
    const size_t layered = prov_name.find(';');
    return (layered == std::string::npos) ? prov_name : prov_name.substr(0, layered);
}

std::vector<std::string>
selectVerbsDomains(const nixlProviderDeviceMap &provider_device_map) {
    std::vector<std::string> domains;
    std::unordered_set<std::string> seen;

    for (const auto &entry : provider_device_map) {
        if (baseProviderName(entry.first) != "verbs") {
            continue;
        }
        // Rails need the ofi_rxm (RDM over MSG) stack; ofi_rxd reaches the same NICs over DGRAM.
        if (entry.first.find("ofi_rxm") == std::string::npos) {
            NIXL_DEBUG << "Ignoring verbs variant " << entry.first
                       << ": nixl rails need the ofi_rxm-layered endpoint";
            continue;
        }
        for (const auto &domain : entry.second) {
            if (hasUnusableVerbsSuffix(domain)) {
                NIXL_DEBUG << "Ignoring verbs domain " << domain << ": not an RDM-capable domain";
                continue;
            }
            if (seen.insert(domain).second) {
                domains.push_back(domain);
            }
        }
    }

    // Sorted so rail indices do not depend on the unordered map's iteration order.
    std::sort(domains.begin(), domains.end());
    return domains;
}

/****************************************
 * Provider table
 *****************************************/

namespace {

    // tcp and sockets support neither FI_MR_PROV_KEY nor FI_MR_VIRT_ADDR.
    constexpr uint64_t k_mr_mode_rdma =
        FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    constexpr uint64_t k_mr_mode_basic = FI_MR_LOCAL | FI_MR_ALLOCATED;

    const nixlLibfabricProviderInfo k_cxi_provider = {
        /*name=*/"cxi",
        /*description=*/"CXI",
        /*extra_caps=*/FI_RMA_EVENT,
        /*mr_mode=*/k_mr_mode_rdma | FI_MR_ENDPOINT,
        // Left to the provider: a key size here changes which fi_info entries cxi returns.
        /*mr_key_size=*/0,
        /*needs_prov_name_hint=*/false,
        /*needs_full_topology=*/true,
        /*needs_all_rail_progress=*/false,
        /*needs_write_tag=*/false,
        /*single_domain=*/false,
        /*select_domains=*/nullptr,
    };

    const nixlLibfabricProviderInfo k_efa_provider = {
        /*name=*/"efa",
        /*description=*/"EFA",
        /*extra_caps=*/0,
        /*mr_mode=*/k_mr_mode_rdma,
        /*mr_key_size=*/2,
        /*needs_prov_name_hint=*/false,
        /*needs_full_topology=*/true,
        /*needs_all_rail_progress=*/false,
        /*needs_write_tag=*/false,
        /*single_domain=*/false,
        /*select_domains=*/nullptr,
    };

    const nixlLibfabricProviderInfo k_verbs_provider = {
        /*name=*/"verbs",
        /*description=*/"verbs",
        /*extra_caps=*/0,
        /*mr_mode=*/k_mr_mode_rdma,
        /*mr_key_size=*/2,
        /*needs_prov_name_hint=*/true,
        /*needs_full_topology=*/true,
        // Layered over a connection-oriented core, so the passive side must poll for the accept.
        /*needs_all_rail_progress=*/true,
        // ofi_rxm gives a remote write no usable source address; see providerNeedsWriteTag().
        /*needs_write_tag=*/true,
        /*single_domain=*/false,
        /*select_domains=*/selectVerbsDomains,
    };

    const nixlLibfabricProviderInfo k_tcp_provider = {
        /*name=*/"tcp",
        /*description=*/"tcp (TCP fallback)",
        /*extra_caps=*/0,
        /*mr_mode=*/k_mr_mode_basic,
        /*mr_key_size=*/0,
        /*needs_prov_name_hint=*/false,
        /*needs_full_topology=*/false,
        /*needs_all_rail_progress=*/false,
        /*needs_write_tag=*/false,
        /*single_domain=*/true,
        /*select_domains=*/nullptr,
    };

    const nixlLibfabricProviderInfo k_sockets_provider = {
        /*name=*/"sockets",
        /*description=*/"sockets (TCP fallback)",
        /*extra_caps=*/0,
        /*mr_mode=*/k_mr_mode_basic,
        /*mr_key_size=*/0,
        /*needs_prov_name_hint=*/false,
        /*needs_full_topology=*/false,
        /*needs_all_rail_progress=*/false,
        /*needs_write_tag=*/false,
        /*single_domain=*/true,
        /*select_domains=*/nullptr,
    };

} // namespace

std::vector<std::string>
selectDomainsGeneric(const nixlProviderDeviceMap &provider_device_map, const char *name) {
    std::vector<std::string> domains;
    std::unordered_set<std::string> seen;

    const auto it = provider_device_map.find(name);
    if (it == provider_device_map.end()) {
        return domains;
    }
    for (const auto &domain : it->second) {
        if (seen.insert(domain).second) {
            domains.push_back(domain);
        }
    }
    return domains;
}

/*
 * Preference order, highest first. Use FI_PROVIDER to restrict the choice.
 *
 * constexpr so it has no destructor: the progress thread may still read it while exit() runs.
 */
constexpr std::array<const nixlLibfabricProviderInfo *, 5> k_known_providers = {
    &k_cxi_provider,
    &k_efa_provider,
    &k_verbs_provider,
    &k_tcp_provider,
    &k_sockets_provider,
};

std::span<const nixlLibfabricProviderInfo *const>
knownProviders() {
    return k_known_providers;
}

const nixlLibfabricProviderInfo *
findProviderInfo(const std::string &prov_name) {
    const std::string core = baseProviderName(prov_name);
    for (const nixlLibfabricProviderInfo *provider : knownProviders()) {
        if (core == provider->name) {
            return provider;
        }
    }
    return nullptr;
}

const nixlLibfabricProviderInfo &
defaultProviderInfo() {
    return k_efa_provider;
}

bool
providerNeedsAllRailProgress(const std::string &prov_name) {
    const nixlLibfabricProviderInfo *provider = findProviderInfo(prov_name);
    return (provider != nullptr) && provider->needs_all_rail_progress;
}

bool
providerNeedsWriteTag(const std::string &prov_name) {
    const nixlLibfabricProviderInfo *provider = findProviderInfo(prov_name);
    return (provider != nullptr) && provider->needs_write_tag;
}

bool
providerNeedsFullTopology(const std::string &prov_name) {
    const nixlLibfabricProviderInfo *provider = findProviderInfo(prov_name);
    return (provider != nullptr) && provider->needs_full_topology;
}

std::pair<std::string, std::vector<std::string>>
selectProvider(const nixlProviderDeviceMap &provider_device_map) {
    // First table entry with usable domains wins.
    for (const nixlLibfabricProviderInfo *provider : knownProviders()) {
        std::vector<std::string> domains = (provider->select_domains != nullptr) ?
            provider->select_domains(provider_device_map) :
            selectDomainsGeneric(provider_device_map, provider->name);
        if (domains.empty()) {
            continue;
        }
        if (provider->single_domain && domains.size() > 1) {
            domains.resize(1);
        }
        return {provider->name, domains};
    }
    return {"none", {}};
}

std::pair<std::string, std::vector<nixlLibfabricDeviceInfo>>
selectNetworkDevices(const struct fi_info *info) {
    nixlProviderDeviceMap provider_device_map;
    // Bus address and link speed per core provider and domain name. Several providers can report
    // the same domain name (verbs and psm3) for different NICs or with different metadata, so a
    // domain's metadata is only taken from the provider that lists it. Keyed by core provider,
    // since that is what selectProvider() returns and a provider's select_domains hook may draw
    // from its layered variants ("verbs;ofi_rxm"), which describe the same NICs. Within one
    // provider the first non-empty value is kept.
    std::unordered_map<std::string, std::unordered_map<std::string, nixlLibfabricDeviceInfo>>
        device_info;

    for (const struct fi_info *cur = info; cur; cur = cur->next) {
        if (cur->domain_attr && cur->domain_attr->name && cur->fabric_attr &&
            cur->fabric_attr->prov_name) {

            std::string device_name = cur->domain_attr->name;
            std::string provider_name = cur->fabric_attr->prov_name;

            NIXL_TRACE << "Found device - domain: " << device_name << ", provider=" << provider_name
                       << ", ep_type=" << (cur->ep_attr ? cur->ep_attr->type : FI_EP_UNSPEC)
                       << ", caps=" << std::hex << cur->caps << std::dec;

            provider_device_map[provider_name].push_back(device_name);

            nixlLibfabricDeviceInfo &dev =
                device_info[baseProviderName(provider_name)][device_name];
            dev.domain_name = device_name;

            if (cur->nic == nullptr) {
                continue;
            }
            if (dev.pcie_address.empty() && cur->nic->bus_attr != nullptr &&
                cur->nic->bus_attr->bus_type == FI_BUS_PCI &&
                cur->nic->bus_attr->attr.pci.domain_id != FI_ADDR_UNSPEC) {
                char pcie_addr[32];
                snprintf(pcie_addr,
                         sizeof(pcie_addr),
                         "%x:%02x:%02x.%x",
                         cur->nic->bus_attr->attr.pci.domain_id,
                         cur->nic->bus_attr->attr.pci.bus_id,
                         cur->nic->bus_attr->attr.pci.device_id,
                         cur->nic->bus_attr->attr.pci.function_id);
                dev.pcie_address = pcie_addr;
            }
            if (dev.link_speed == 0 && cur->nic->link_attr != nullptr) {
                dev.link_speed = cur->nic->link_attr->speed;
            }
        }
    }

    for (const auto &device_list : provider_device_map) {
        for (const auto &device : device_list.second) {
            NIXL_TRACE << "provider=" << device_list.first << ", device=" << device;
        }
    }

    const auto [provider_name, domains] = selectProvider(provider_device_map);
    if (provider_name == "none") {
        return {"none", {}};
    }

    const auto &provider_info = device_info[provider_name];
    std::vector<nixlLibfabricDeviceInfo> devices;
    devices.reserve(domains.size());
    for (const auto &domain : domains) {
        const auto it = provider_info.find(domain);
        if (it != provider_info.end()) {
            devices.push_back(it->second);
        } else {
            // A select_domains hook named a domain its provider did not list.
            nixlLibfabricDeviceInfo dev;
            dev.domain_name = domain;
            devices.push_back(dev);
        }
        NIXL_TRACE << "Selected " << provider_name << " domain " << devices.back().domain_name
                   << " pcie="
                   << (devices.back().pcie_address.empty() ? "unknown" :
                                                             devices.back().pcie_address)
                   << " link_speed=" << devices.back().link_speed;
    }
    return {provider_name, devices};
}

std::pair<std::string, std::vector<nixlLibfabricDeviceInfo>>
getAvailableNetworkDevices() {
    struct fi_info *hints, *info;
    hints = fi_allocinfo();
    if (!hints) {
        NIXL_ERROR << "Failed to allocate fi_info";
        return {"none", {}};
    }

    hints->caps = 0;
    hints->caps = FI_MSG | FI_RMA; // Basic messaging and RMA

    hints->caps |= FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;

    /*
     * Allow providers to advertise their supported mr_mode (excluding deprecated bits 0-1)
     *
     * This is the means used in the code for the fi_info command to retrieve all providers
     * from libfabric.  It excludes FI_MR_BASIC and FI_MR_SCALABLE, which are deprecated.
     *
     * It's not ideal to hard-code a constant but this makes the Slingshot (CXI) provider work
     * and it is constent with the libfabric fi_info example.
     */
    hints->domain_attr->mr_mode = ~3;

    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed " << fi_strerror(-ret);
        fi_freeinfo(hints);
        return {"none", {}};
    }

    auto result = selectNetworkDevices(info);

    fi_freeinfo(info);
    fi_freeinfo(hints);

    if (result.first != "none") {
        return result;
    }

    NIXL_ERROR << "No network devices found with any provider";
    return {"none", {}};
}

std::string
hexdump(const void *data, size_t size) {
    std::stringstream ss;
    ss.str().reserve(size * 3);
    const unsigned char *bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]) << " ";
    }
    return ss.str();
}

std::string
railIdsToString(const std::vector<size_t> &rail_ids) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < rail_ids.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << rail_ids[i];
    }
    ss << "]";
    return ss.str();
}

bool
getMaxNumaNode(int &node_id) {
    if (numa_available() < 0) {
        NIXL_ERROR << "Failed to retrieve maximum NUMA node id: libnuma is unavailable";
        return false;
    }
    int max_node = numa_max_node();
    if (max_node < 0) {
        NIXL_ERROR << "Failed to retrieve maximum NUMA node id, numa_max_node() returned: "
                   << max_node;
        return false;
    }
    node_id = max_node;
    return true;
}

bool
getNumConfiguredNumaNodes(int &node_count) {
    if (numa_available() < 0) {
        NIXL_ERROR << "Failed to retrieve number of configured NUMA nodes: libnuma is unavailable";
        return false;
    }
    int num_nodes = numa_num_configured_nodes();
    if (num_nodes < 0) {
        NIXL_ERROR << "Failed to retrieve number of configured NUMA nodes, "
                      "numa_num_configured_nodes() returned: "
                   << num_nodes;
        return false;
    }
    node_count = num_nodes;
    return true;
}

// Thread-safe atomic counters for optimized ID generation
static std::atomic<uint16_t> g_xfer_id_counter{1}; // 16-bit XFER_ID counter, start from 1
static std::atomic<uint8_t> g_seq_id_counter{0}; // 4-bit SEQ_ID counter, start from 0

uint16_t
getNextXferId() {
    uint16_t xfer_id = g_xfer_id_counter.fetch_add(1);

    // Handle wraparound: 16-bit field can hold 0 to 65,535
    if (xfer_id > NIXL_XFER_ID_MASK) {
        // Reset counter atomically and get a fresh ID
        uint16_t expected = xfer_id;
        while (expected > NIXL_XFER_ID_MASK &&
               !g_xfer_id_counter.compare_exchange_weak(expected, 1)) {
            expected = g_xfer_id_counter.load();
        }
        xfer_id = g_xfer_id_counter.fetch_add(1);
        // Ensure we don't exceed the mask after reset
        if (xfer_id > NIXL_XFER_ID_MASK) {
            xfer_id = 1;
        }
    }

    return xfer_id;
}

uint8_t
getNextSeqId() {
    uint8_t seq_id = g_seq_id_counter.fetch_add(1);

    // Handle wraparound: 4-bit field can hold 0 to 15
    if (seq_id > NIXL_SEQ_ID_MASK) {
        // Reset counter atomically and get a fresh ID
        uint8_t expected = seq_id;
        while (expected > NIXL_SEQ_ID_MASK &&
               !g_seq_id_counter.compare_exchange_weak(expected, 0)) {
            expected = g_seq_id_counter.load();
        }
        seq_id = g_seq_id_counter.fetch_add(1);
        // Ensure we don't exceed the mask after reset
        if (seq_id > NIXL_SEQ_ID_MASK) {
            seq_id = 0;
        }
    }

    return seq_id;
}

void
resetSeqId() {
    // Reset SEQ_ID counter for new postXfer
    g_seq_id_counter.store(0);
}

nixl_status_t
getCustomStringParam(const nixl_b_params_t &custom_params,
                     const std::string &key,
                     std::string &value) {
    // first check for environment variable override
    // we do this by using upper case name with NIXL_LIBFABRIC_ prefix
    std::string upper_key = key;
    std::transform(key.begin(), key.end(), upper_key.begin(), ::toupper);
    upper_key = std::string("NIXL_LIBFABRIC_") + upper_key;
    NIXL_DEBUG << "Checking override from env var: " << upper_key;
    char *env_value = getenv(upper_key.c_str());
    if (env_value != nullptr) {
        value = env_value;
        NIXL_TRACE << "Overriding configuration item " << key << " by corresponding environment "
                   << "variable " << upper_key;
        return NIXL_SUCCESS;
    }

    nixl_b_params_t::const_iterator itr = custom_params.find(key);
    if (itr != custom_params.end()) {
        value = itr->second;
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_NOT_FOUND;
}

nixl_status_t
getCustomIntParam(const nixl_b_params_t &custom_params, const std::string &key, size_t &value) {
    // first get string value
    std::string value_str;
    nixl_status_t res = getCustomStringParam(custom_params, key, value_str);
    if (res != NIXL_SUCCESS) {
        NIXL_DEBUG << "Using default " << key << ": " << value;
        return res;
    }

    // attempt to convert to integer
    try {
        if (value_str.empty()) {
            NIXL_WARN << "Empty " << key << " configuration value, using default: " << value;
            return NIXL_ERR_INVALID_PARAM;
        }
        if (value_str[0] == '-') {
            NIXL_WARN << "Invalid " << key << " configuration value '" << value_str
                      << "': expecting non-negative integer, using default: " << value;
            return NIXL_ERR_INVALID_PARAM;
        }
        std::size_t pos = 0;
        uint64_t parsed_value = std::stoull(value_str, &pos, 10);
        if (pos != value_str.size()) {
            NIXL_WARN << "Invalid " << key << " configuration value '" << value_str
                      << "': excess non-digit characters from position " << pos << "('"
                      << value_str.substr(pos) << "'), using default: " << value;
            return NIXL_ERR_INVALID_PARAM;
        }
        if (parsed_value > SIZE_MAX) {
            NIXL_WARN << "Invalid " << key << " configuration value '" << parsed_value
                      << "': exceeding maximum allowed " << SIZE_MAX
                      << ", using default: " << value;
            return NIXL_ERR_INVALID_PARAM;
        }

        // conversion is safe now
        value = (size_t)parsed_value;
        NIXL_DEBUG << "Using custom value " << key << ": " << value;
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Invalid " << key << " configuration value '" << value_str << "': " << e.what()
                  << ", expecting non-negative integer, using default: " << value;
        return NIXL_ERR_INVALID_PARAM;
    }
    return NIXL_SUCCESS;
}

static std::unique_ptr<nixlLibfaricCudaCtxMediator> cuda_ctx_mediator;

void
setCudaCtxMediator(std::unique_ptr<nixlLibfaricCudaCtxMediator> &&mediator) {
    cuda_ctx_mediator = std::move(mediator);
}

void
clearCudaCtxMediator() {
    cuda_ctx_mediator.reset(nullptr);
}

nixl_status_t
cudaSetCtx(bool &use_cuda_addr_wa) {
    if (cuda_ctx_mediator.get() == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return cuda_ctx_mediator->cudaSetCtx(use_cuda_addr_wa);
}

} // namespace LibfabricUtils
