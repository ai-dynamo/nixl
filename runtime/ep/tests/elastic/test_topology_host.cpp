// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

// Host-only launch-guard and capture-capacity regression; no GPU or transport is initialized.
// From runtime/ep, with CUDA_INCLUDE pointing to the CUDA include directory:
// g++ -std=c++20 -O2 -Wno-attributes -Inixl_ep_v2/include -I"$CUDA_INCLUDE" \
//     -I"$CUDA_INCLUDE/cccl" tests/elastic/test_topology_host.cpp -o /tmp/test_topology_host
// /tmp/test_topology_host
#include <nixl_ep_v2/common/topology.cuh>
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <string>

using namespace nixl_ep_v2::elastic;

static topology::EPTopology make(std::initializer_list<int> domains, std::initializer_list<int> positions) {
    topology::EPTopology t{};
    t.mode = domains.size() == 1 ? topology::kModeDirect : topology::kModeHybrid;
    t.scaleout_rank_idx = *domains.begin();
    t.scaleup_rank_idx = *positions.begin();
    for (int d : domains) {
        t.scaleout_mask |= 1u << d;
        if (d + 1 > t.num_scaleout_ranks) t.num_scaleout_ranks = d + 1;
    }
    for (int k : positions) t.scaleup_mask.words[k / 64] |= 1ull << (k % 64);
    t.num_scaleup_ranks = t.scaleup_mask.extent();
    return t;
}

int main() {
    unsigned accepted = 0, refused_small = 0, refused_inactive = 0;
    auto refuse = [&](const topology::EPTopology& t, bool small) {
        bool rejected = false;
        try { t.check_launch(); }
        catch (const EPException& e) {
            const std::string message = e.what();
            assert(message.find(small ? "at least two active GPUs" : "not active in the committed world") != std::string::npos);
            rejected = true;
        }
        assert(rejected);
        if (small) ++refused_small; else ++refused_inactive;
    };
    for (const auto& t : {make({0}, {0}), make({31}, {127})}) refuse(t, true);
    for (const auto& t : {make({0}, {0, 1}), make({0, 1}, {0}),
                         make({31}, {65, 127}), make({0, 31}, {127}),
                         make({0}, {0, 1, 2}), make({0, 1, 2}, {0}),
                         make({0}, {0, 64, 127}), make({0, 31}, {0, 127}),
                         make({0}, {0, 1, 2, 3, 4, 5, 6, 7}), make({0, 1}, {0, 1, 2, 3, 4, 5, 6, 7})}) {
        t.check_launch();
        ++accepted;
    }
    auto inactive_domain = make({0, 31}, {0});
    inactive_domain.scaleout_rank_idx = 1;
    refuse(inactive_domain, false);
    auto inactive_position = make({0}, {0, 127});
    inactive_position.scaleup_rank_idx = 64;
    refuse(inactive_position, false);
    auto out_of_extent = make({0}, {0, 1});
    out_of_extent.scaleup_rank_idx = 2;
    refuse(out_of_extent, false);

    // Local experts follow the extent, including holes; an admitted world has extent >= active count >= minimum.
    unsigned bound_cases = 0, odd_expert_cases = 0;
    for (int experts = 2; experts <= 2048; ++experts) {
        for (int d = 1; d <= topology::kNumMaxScaleoutRanksCap; ++d) {
            for (int k = 1; k <= topology::kNumMaxScaleupRanksCap; ++k) {
                const int extent = d * k;
                if (extent < topology::kMinNumActiveRanks || experts % extent != 0) continue;
                assert(experts / extent <= experts / topology::kMinNumActiveRanks);
                ++bound_cases;
                if (experts % 2) ++odd_expert_cases;
            }
        }
    }
    assert(odd_expert_cases > 0);  // Three-rank worlds permit expert counts not divisible by two.
    const int stats_capacity = 256 / topology::kMinNumActiveRanks;
    assert(stats_capacity == 128);
    for (int extent : {16, 8, 4, 2, 4, 8, 16}) assert(256 / extent <= stats_capacity);
    std::printf("Passed: %u accepted worlds, %u single-GPU refusals, %u inactive-rank refusals, "
                "%u capacity bounds (%u with odd expert counts); 256 experts reserve %d entries.\n",
                accepted, refused_small, refused_inactive, bound_cases, odd_expert_cases, stats_capacity);
}
