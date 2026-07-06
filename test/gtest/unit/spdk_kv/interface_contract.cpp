/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "spdk_kv_engine.h"

namespace {

class InspectableSpdkKvEngine final : public nixlSpdkKvEngine {
public:
    explicit InspectableSpdkKvEngine(const nixlBackendInitParams *init_params)
        : nixlSpdkKvEngine(init_params) {}

    InspectableSpdkKvEngine(const nixlBackendInitParams *init_params,
                            std::unique_ptr<iSpdkKvDevice> device)
        : nixlSpdkKvEngine(init_params, std::move(device)) {}

    void
    lockDeviceFromConstMethod() const {
        const std::lock_guard<std::mutex> lock(deviceMutex_);
    }

protected:
    bool
    deriveKey(const std::string &, uint8_t, std::vector<uint8_t> &) const override {
        return false;
    }
};

using max_key_len_signature_t = uint8_t (iSpdkKvDevice::*)() const;
using max_value_len_signature_t = size_t (iSpdkKvDevice::*)() const;
using store_signature_t = spdk_kv_status_t (iSpdkKvDevice::*)(const void *,
                                                              uint8_t,
                                                              const void *,
                                                              size_t);
using retrieve_signature_t =
    spdk_kv_status_t (iSpdkKvDevice::*)(const void *, uint8_t, void *, size_t, size_t *);

static_assert(std::is_same_v<decltype(&iSpdkKvDevice::maxKeyLen), max_key_len_signature_t>);
static_assert(std::is_same_v<decltype(&iSpdkKvDevice::maxValueLen), max_value_len_signature_t>);
static_assert(std::is_same_v<decltype(&iSpdkKvDevice::store), store_signature_t>);
static_assert(std::is_same_v<decltype(&iSpdkKvDevice::retrieve), retrieve_signature_t>);
static_assert(std::is_constructible_v<InspectableSpdkKvEngine, const nixlBackendInitParams *>);
static_assert(std::is_constructible_v<InspectableSpdkKvEngine,
                                      const nixlBackendInitParams *,
                                      std::unique_ptr<iSpdkKvDevice>>);

} // namespace
