// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

// Host-only launch-guard and routing-coordinate regression; no GPU or transport is initialized.
// From runtime/ep, with CUDA_INCLUDE pointing to the CUDA include directory:
// g++ -std=c++20 -O2 -Wno-attributes -Ipython/include -I"$CUDA_INCLUDE" \
//     -I"$CUDA_INCLUDE/cccl" tests/elastic/test_topology_host.cpp -o /tmp/test_topology_host
// /tmp/test_topology_host
#include <nixl_ep_v2/common/topology.cuh>
#include <array>
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <numeric>
#include <string>

using namespace nixl_ep_v2::elastic;

static topology::EPTopology make(std::initializer_list<int> domains, std::initializer_list<int> positions) {
    static const auto identity = [] {
        std::array<int, topology::kNumMaxRanksCap> ranks{};
        std::iota(ranks.begin(), ranks.end(), 0);
        return ranks;
    }();
    topology::EPTopology t{};
    t.num_world_ranks = identity.size();
    t.world_to_slot = t.slot_to_world = identity.data();
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

template <size_t N>
static void map_ranks(topology::EPTopology& t, const std::array<int, N>& world_to_slot,
                      const std::array<int, N>& slot_to_world) {
    t.num_world_ranks = N;
    t.world_to_slot = world_to_slot.data();
    t.slot_to_world = slot_to_world.data();
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

    // Physical domains {0, 2} and {1, 3} need no topology-aware public rank numbering.
    const std::array<int, 4> permutation{0, 2, 1, 3};
    auto permuted = make({0, 1}, {0, 1});
    map_ranks(permuted, permutation, permutation);
    assert(topology::scaleup_peer<2>(permuted, 1) == 2);
    assert(topology::scaleout_peer<2>(permuted, 1) == 1);
    permuted.scaleup_rank_idx = 1;
    assert(topology::scaleout_peer<2>(permuted, 1) == 3);
    auto domain_zero = make({0}, {0, 1});
    map_ranks(domain_zero, permutation, permutation);
    for (int local_slot = 0; local_slot < slots_per_rank; ++local_slot) {
        assert((topology::world_expert_to_extent<2, slots_per_rank>(permuted, 2 * slots_per_rank + local_slot) ==
                slots_per_rank + local_slot));
        assert((topology::world_expert_to_extent<2, slots_per_rank>(permuted, slots_per_rank + local_slot) ==
                2 * slots_per_rank + local_slot));
        assert((topology::world_expert_to_extent<2, slots_per_rank>(domain_zero, 2 * slots_per_rank + local_slot) ==
                slots_per_rank + local_slot));
        assert((topology::world_expert_to_extent<2, slots_per_rank>(domain_zero, slots_per_rank + local_slot) == -1));
    }
    // Combine must classify expert owners using the same map, including after domain shrink.
    const int local_owner_slot = topology::rank_slot(domain_zero, 2);
    const int remote_owner_slot = topology::rank_slot(domain_zero, 1);
    assert(local_owner_slot == 1 && remote_owner_slot == 2);
    assert(domain_zero.is_active(local_owner_slot / 2, local_owner_slot % 2));
    assert(!domain_zero.is_active(remote_owner_slot / 2, remote_owner_slot % 2));

    // A singleton's local slot is zero even when the application's world rank is three.
    const std::array<int, 4> singleton_world{-1, -1, -1, 0};
    const std::array<int, 4> singleton_slots{3, -1, -1, -1};
    auto singleton = make({0}, {0});
    map_ranks(singleton, singleton_world, singleton_slots);
    assert(topology::scaleup_peer<2>(singleton, 0) == 3);
    assert(topology::scaleout_peer<2>(singleton, 0) == 3);
    assert((topology::world_expert_to_extent<2, slots_per_rank>(singleton, 3 * slots_per_rank + 1) == 1));
    assert((topology::world_expert_to_extent<2, slots_per_rank>(singleton, 1) == -1));
    for (int owner : {-1, 0, 4}) {
        const int owner_slot = topology::rank_slot(singleton, owner);
        assert(owner_slot == -1);
        assert(!singleton.is_active(owner_slot / 2, owner_slot % 2));
    }

    // Mapping does not truncate positions above bit 63 or require a power-of-two maximum width.
    std::array<int, 144> wide_permutation{};
    std::iota(wide_permutation.begin(), wide_permutation.end(), 0);
    std::swap(wide_permutation[5], wide_permutation[137]);
    auto wide_mapped = wide;
    map_ranks(wide_mapped, wide_permutation, wide_permutation);
    assert(topology::scaleup_peer<72>(wide_mapped, 65) == 5);
    assert(topology::rank_slot(wide_mapped, 5) == 137);
    assert((topology::world_expert_to_extent<72, slots_per_rank>(wide_mapped, 5 * slots_per_rank + 1) ==
            137 * slots_per_rank + 1));
    std::printf("Passed: %u accepted worlds, %u inactive-rank refusals, "
                "stable expert IDs through shrink, expansion, permuted ranks, singleton setup and wide-domain holes.\n",
                accepted, refused_inactive);
}
