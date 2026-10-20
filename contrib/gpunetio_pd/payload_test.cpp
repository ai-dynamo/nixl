/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "payload.h"
#include <cassert>
#include <set>
#include <cstdint>
#include <iostream>

int
main() {
    std::set<uint64_t> identities;
    for (uint64_t epoch : {1ULL, 127ULL, 128ULL, 611ULL, 1001000ULL}) {
        for (unsigned peer = 0; peer < 4; ++peer) {
            for (unsigned layer = 0; layer < 27; ++layer) {
                for (unsigned row = 0; row < 64; ++row) {
                    for (unsigned slot = 0; slot < 3; ++slot) {
                        uint64_t id = 0;
                        for (unsigned byte = 0; byte < 8; ++byte) {
                            id |= uint64_t(payloadByte(epoch, peer, layer, row, slot, byte))
                                << (byte * 8);
                        }
                        assert(identities.insert(id).second);
                    }
                }
            }
        }
    }
    assert(payloadByte(1, 0, 0, 0, 0, 1) != payloadByte(1, 0, 1, 59, 0, 1));
    std::cout << "UNIQUE_PAYLOAD_ID_PASS " << identities.size() << '\n';
}
