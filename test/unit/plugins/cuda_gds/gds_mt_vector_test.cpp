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

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "gds_mt_engine.h"

namespace {
constexpr size_t block_size = 4096;
int failures = 0;

void
expect(bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
        ++failures;
    }
}

gdsXferReq
makeReq(size_t index,
        CUfileHandle_t fh,
        CUfileOpcode_t op = CUFILE_READ,
        size_t file_offset = 0,
        size_t size = block_size) {
    return {reinterpret_cast<void *>(static_cast<uintptr_t>((index + 1) * block_size)),
            size,
            file_offset,
            fh,
            op};
}

std::vector<gdsXferReq>
makeContiguousReqs(size_t count, CUfileHandle_t fh, CUfileOpcode_t op = CUFILE_READ) {
    std::vector<gdsXferReq> reqs;
    reqs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        reqs.push_back(makeReq(i, fh, op, i * block_size));
    }
    return reqs;
}

const nixlGdsMtVectorReq *
vectorAt(const std::vector<nixlGdsMtWorkItem> &work_list, size_t index) {
    if (index >= work_list.size()) {
        return nullptr;
    }
    return std::get_if<nixlGdsMtVectorReq>(&work_list[index]);
}

void
testVectorDisabledLeavesScalarWork() {
    auto reqs = makeContiguousReqs(32, reinterpret_cast<CUfileHandle_t>(0x1000));
    const auto work_list = buildGdsMtWorkList(reqs, false, 32);

    expect(work_list.size() == reqs.size(), "vector_io=off must preserve 32 scalar items");
    for (const auto &work : work_list) {
        expect(std::holds_alternative<gdsXferReq>(work),
               "vector_io=off emitted a vector work item");
    }
}

void
testContiguousRequestsFormOneVector() {
    auto reqs = makeContiguousReqs(32, reinterpret_cast<CUfileHandle_t>(0x1000));
    const auto work_list = buildGdsMtWorkList(reqs, true, 32);
    const auto *vector_req = vectorAt(work_list, 0);

    expect(work_list.size() == 1, "32 contiguous requests must form one work item");
    expect(vector_req != nullptr, "32 contiguous requests must form a vector item");
    if (vector_req != nullptr) {
        expect(vector_req->entries.size() == 32, "vector item must contain 32 entries");
        expect(vector_req->total_size == 32 * block_size, "vector total size is incorrect");
        expect(vector_req->file_offset == 0, "vector starting file offset is incorrect");
    }
}

void
testMaxEntriesSplitsVectors() {
    auto reqs = makeContiguousReqs(32, reinterpret_cast<CUfileHandle_t>(0x1000));
    const auto work_list = buildGdsMtWorkList(reqs, true, 8);

    expect(work_list.size() == 4, "vector_max_entries=8 must split 32 requests four ways");
    for (size_t i = 0; i < work_list.size(); ++i) {
        const auto *vector_req = vectorAt(work_list, i);
        expect(vector_req != nullptr, "each capped group must remain a vector item");
        if (vector_req != nullptr) {
            expect(vector_req->entries.size() == 8, "each capped vector must contain 8 entries");
            expect(vector_req->file_offset == i * 8 * block_size,
                   "capped vector starts at the wrong file offset");
        }
    }
}

void
testTrailingSingletonStaysScalar() {
    auto reqs = makeContiguousReqs(33, reinterpret_cast<CUfileHandle_t>(0x1000));
    const auto work_list = buildGdsMtWorkList(reqs, true, 32);
    const auto *vector_req = vectorAt(work_list, 0);

    expect(work_list.size() == 2, "33 requests with a 32-entry cap must make two items");
    expect(vector_req != nullptr && vector_req->entries.size() == 32,
           "the first item must be a 32-entry vector");
    expect(std::holds_alternative<gdsXferReq>(work_list[1]),
           "a trailing one-entry group must remain scalar");
}

void
testGroupingBoundaries() {
    auto *handle_a = reinterpret_cast<CUfileHandle_t>(0x1000);
    auto *handle_b = reinterpret_cast<CUfileHandle_t>(0x2000);
    std::vector<gdsXferReq> reqs = {
        makeReq(0, handle_a, CUFILE_READ, 0),
        makeReq(1, handle_a, CUFILE_READ, block_size),
        makeReq(2, handle_a, CUFILE_READ, 3 * block_size),
        makeReq(3, handle_b, CUFILE_READ, 4 * block_size),
        makeReq(4, handle_b, CUFILE_WRITE, 5 * block_size),
    };
    const auto work_list = buildGdsMtWorkList(reqs, true, 32);
    const auto *first = vectorAt(work_list, 0);

    expect(work_list.size() == 4,
           "offset, handle, and operation boundaries must split vector groups");
    expect(first != nullptr && first->entries.size() == 2,
           "the initial contiguous pair must remain vectorized");
    for (size_t i = 1; i < work_list.size(); ++i) {
        expect(std::holds_alternative<gdsXferReq>(work_list[i]),
               "one-entry boundary groups must remain scalar");
    }
}
} // namespace

int
main() {
    testVectorDisabledLeavesScalarWork();
    testContiguousRequestsFormOneVector();
    testMaxEntriesSplitsVectors();
    testTrailingSingletonStaysScalar();
    testGroupingBoundaries();

    if (failures != 0) {
        std::cerr << failures << " GDS_MT vector grouping assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "GDS_MT vector grouping tests passed" << std::endl;
    return 0;
}
