/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "obj_engine_registry.h"
#include "obj_backend.h"
#include "common/nixl_log.h"
#include <stdexcept>

objAccelEngineRegistry &
objAccelEngineRegistry::instance() {
    static objAccelEngineRegistry registry;
    return registry;
}

void
objAccelEngineRegistry::add(const std::string &type, objAccelEngineEntry entry) {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = entries_.emplace(type, std::move(entry));
    if (!inserted) {
        NIXL_WARN << "Accelerated engine type '" << type
                  << "' is already registered; ignoring duplicate registration";
    }
}

bool
objAccelEngineRegistry::has(const std::string &type) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return entries_.count(type) > 0;
}

objAccelEngineEntry
objAccelEngineRegistry::lookupOrThrow(const std::string &type) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(type);
    if (it != entries_.end()) return it->second;

    if (type.empty())
        throw std::runtime_error("Accelerated engine support not available (not compiled)");

    throw std::runtime_error("Accelerated engine type '" + type +
                             "' is not available (not compiled)");
}

std::unique_ptr<nixlObjEngineImpl>
objAccelEngineRegistry::create(const std::string &type, const nixlBackendInitParams *params) const {
    return lookupOrThrow(type).create(params);
}

std::unique_ptr<nixlObjEngineImpl>
objAccelEngineRegistry::create(const std::string &type,
                               const nixlBackendInitParams *params,
                               std::shared_ptr<iS3Client> s3_client,
                               std::shared_ptr<iS3Client> s3_client_crt) const {
    return lookupOrThrow(type).createWithClients(
        params, std::move(s3_client), std::move(s3_client_crt));
}
