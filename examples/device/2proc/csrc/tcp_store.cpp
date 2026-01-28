/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tcp_store.h"

#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <chrono>

namespace tcp_store {

class TCPStore::Impl {
public:
    std::shared_ptr<c10d::TCPStore> store;
};

TCPStore::TCPStore(const std::string &host, int port, bool is_master, int timeout_ms)
    : impl_(std::make_unique<Impl>()) {

    c10d::TCPStoreOptions opts;
    opts.port = port;
    opts.isServer = is_master;
    opts.numWorkers = 2; // For 2proc example
    opts.waitWorkers = true;
    opts.timeout = std::chrono::milliseconds(timeout_ms);
    opts.multiTenant = false;

    impl_->store = std::make_shared<c10d::TCPStore>(host, opts);
}

TCPStore::~TCPStore() = default;

void
TCPStore::set(const std::string &key, const std::string &value) {
    std::vector<uint8_t> data(value.begin(), value.end());
    impl_->store->set(key, data);
}

std::string
TCPStore::get(const std::string &key) {
    auto data = impl_->store->get(key);
    return std::string(data.begin(), data.end());
}

bool
TCPStore::wait(const std::string &key, int timeout_ms) {
    try {
        std::vector<std::string> keys = {key};
        impl_->store->wait(keys, std::chrono::milliseconds(timeout_ms));
        return true;
    }
    catch (const std::exception &) {
        return false;
    }
}

void
TCPStore::delete_key(const std::string &key) {
    impl_->store->deleteKey(key);
}

} // namespace tcp_store
