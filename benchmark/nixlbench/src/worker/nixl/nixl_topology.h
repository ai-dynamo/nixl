/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_TOPOLOGY_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_TOPOLOGY_H

#include "utils/utils.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace nixlbench {

struct descriptorRange {
    size_t offset;
    size_t count;
};

inline std::vector<int>
exchangePeerRanks(bool is_sg,
                  const std::string &scheme,
                  bool is_initiator,
                  int rank,
                  int num_initiators,
                  int num_targets) {
    if (!is_sg) {
        return {is_initiator ? 1 : 0};
    }

    if (scheme == XFERBENCH_SCHEME_PAIRWISE) {
        return {is_initiator ? rank + num_initiators : rank - num_targets};
    }

    if (scheme == XFERBENCH_SCHEME_MANY_TO_ONE) {
        if (is_initiator) {
            return {num_initiators};
        }

        std::vector<int> peers;
        peers.reserve(static_cast<size_t>(num_initiators));
        for (int peer = 0; peer < num_initiators; ++peer) {
            peers.push_back(peer);
        }
        return peers;
    }

    // Preserve the existing rank-0/rank-1 behavior for other schemes.
    return {is_initiator ? 1 : 0};
}

inline std::optional<descriptorRange>
manyToOneDescriptorRange(size_t descriptor_count, size_t peer_count, size_t peer_index) {
    if (descriptor_count == 0 || peer_count == 0 || peer_index >= peer_count ||
        descriptor_count % peer_count != 0) {
        return std::nullopt;
    }

    const size_t descriptors_per_peer = descriptor_count / peer_count;
    return descriptorRange{peer_index * descriptors_per_peer, descriptors_per_peer};
}

} // namespace nixlbench

#endif
