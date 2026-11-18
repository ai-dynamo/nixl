// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

// Host-only launch-guard and routing-coordinate regression; no GPU or transport is initialized.
// From runtime/ep, with CUDA_INCLUDE pointing to the CUDA include directory:
// g++ -std=c++20 -O2 -Wno-attributes -Ipython/include -I"$CUDA_INCLUDE" \
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
    unsigned accepted = 0, refused_inactive = 0;
    auto refuse = [&](const topology::EPTopology& t) {
        bool rejected = false;
        try { t.check_launch(); }
        catch (const EPException& e) {
            const std::string message = e.what();
            assert(message.find("not active in the committed world") != std::string::npos);
            rejected = true;
        }
        assert(rejected);
        ++refused_inactive;
    };
    for (const auto& t : {make({0}, {0}), make({31}, {127}),
                         make({0}, {0, 1}), make({0, 1}, {0}),
                         make({31}, {65, 127}), make({0, 31}, {127}),
                         make({0}, {0, 1, 2}), make({0, 1, 2}, {0}),
                         make({0}, {0, 64, 127}), make({0, 31}, {0, 127}),
                         make({0}, {0, 1, 2, 3, 4, 5, 6, 7}), make({0, 1}, {0, 1, 2, 3, 4, 5, 6, 7})}) {
        t.check_launch();
        ++accepted;
    }
    auto inactive_domain = make({0, 31}, {0});
    inactive_domain.scaleout_rank_idx = 1;
    refuse(inactive_domain);
    auto inactive_position = make({0}, {0, 127});
    inactive_position.scaleup_rank_idx = 64;
    refuse(inactive_position);
    auto out_of_extent = make({0}, {0, 1});
    out_of_extent.scaleup_rank_idx = 2;
    refuse(out_of_extent);

    // Public IDs stay fixed through 2x8 -> 2x4 -> a nonzero direct domain -> 2x8.
    constexpr int slots_per_rank = 4;
    const auto full = make({0, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
    const auto half = make({0, 1}, {0, 1, 2, 3});
    const auto direct = make({1}, {0, 1});
    const auto holes = make({0, 1}, {0, 2});
    for (int local_slot = 0; local_slot < slots_per_rank; ++local_slot) {
        const int slot = 8 * slots_per_rank + local_slot;
        assert((topology::world_expert_to_extent<8, slots_per_rank>(full, slot) == slot));
        assert((topology::world_expert_to_extent<8, slots_per_rank>(half, slot) == 4 * slots_per_rank + local_slot));
        assert((topology::world_expert_to_extent<8, slots_per_rank>(direct, slot) == 2 * slots_per_rank + local_slot));
        assert((topology::world_expert_to_extent<8, slots_per_rank>(holes, slot) == 3 * slots_per_rank + local_slot));
        assert((topology::world_expert_to_extent<8, slots_per_rank>(full, slot) == slot));
        // World rank 4 is inactive after shrink; translating it first would alias live world rank 8.
        assert((topology::world_expert_to_extent<8, slots_per_rank>(half, 4 * slots_per_rank + local_slot) == -1));
        assert((topology::world_expert_to_extent<8, slots_per_rank>(direct, local_slot) == -1));
        assert((topology::world_expert_to_extent<8, slots_per_rank>(holes, slots_per_rank + local_slot) == -1));
    }
    assert((topology::world_expert_to_extent<8, slots_per_rank>(half, -1) == -1));
    assert((topology::world_expert_to_extent<8, slots_per_rank>(full, 16 * slots_per_rank) == -1));

    // Non-power-of-two maximum width and positions above bit 63.
    const auto wide = make({1}, {65, 71});
    const auto shrunk = make({1}, {64, 65});
    assert(wide.num_scaleout_ranks * wide.num_scaleup_ranks * slots_per_rank == 576);
    assert(wide.num_active_scaleout_ranks() * wide.num_active_scaleup_ranks() * slots_per_rank == 8);
    for (int local_slot = 0; local_slot < slots_per_rank; ++local_slot) {
        const int slot = (1 * 72 + 65) * slots_per_rank + local_slot;
        assert((topology::world_expert_to_extent<72, slots_per_rank>(wide, slot) == slot));
        assert((topology::world_expert_to_extent<72, slots_per_rank>(shrunk, slot) == (1 * 66 + 65) * slots_per_rank + local_slot));
        assert((topology::world_expert_to_extent<72, slots_per_rank>(wide, (1 * 72 + 64) * slots_per_rank + local_slot) == -1));
        assert((topology::world_expert_to_extent<72, slots_per_rank>(wide, (2 * 72 + 65) * slots_per_rank + local_slot) == -1));
    }
    assert(topology::scaleup_peer<72>(wide, 65) == topology::scaleup_peer<72>(shrunk, 65));
    std::printf("Passed: %u accepted worlds, %u inactive-rank refusals, "
                "stable expert IDs through shrink, expansion and wide-domain holes.\n",
                accepted, refused_inactive);
}
