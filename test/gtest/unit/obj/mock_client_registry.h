/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_TEST_GTEST_UNIT_OBJ_MOCK_CLIENT_REGISTRY_H
#define NIXL_TEST_GTEST_UNIT_OBJ_MOCK_CLIENT_REGISTRY_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/nixl_log.h"

namespace gtest::obj {

class mockS3Client;

using mock_s3_client_factory_t = std::function<std::shared_ptr<mockS3Client>()>;

class mockClientRegistry {
public:
    static mockClientRegistry &
    instance() {
        static mockClientRegistry registry;
        return registry;
    }

    void
    add(const std::string &type, mock_s3_client_factory_t factory) {
        auto [it, inserted] = entries_.emplace(type, std::move(factory));
        if (!inserted) {
            NIXL_WARN << "Mock client type '" << type
                      << "' is already registered; ignoring duplicate registration";
        }
    }

    std::shared_ptr<mockS3Client>
    create(const std::string &type) const {
        auto it = entries_.find(type);
        if (it != entries_.end()) return it->second();
        return nullptr;
    }

private:
    mockClientRegistry() = default;
    std::unordered_map<std::string, mock_s3_client_factory_t> entries_;
};

struct mockClientRegistrar {
    mockClientRegistrar(const std::string &type, mock_s3_client_factory_t factory) {
        mockClientRegistry::instance().add(type, std::move(factory));
    }
};

} // namespace gtest::obj

#endif // NIXL_TEST_GTEST_UNIT_OBJ_MOCK_CLIENT_REGISTRY_H
