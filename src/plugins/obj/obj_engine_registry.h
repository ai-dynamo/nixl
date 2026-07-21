/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_OBJ_OBJ_ENGINE_REGISTRY_H
#define NIXL_SRC_PLUGINS_OBJ_OBJ_ENGINE_REGISTRY_H

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

class nixlObjEngineImpl;
class iS3Client;
struct nixlBackendInitParams;

using obj_accel_create_fn_t =
    std::function<std::unique_ptr<nixlObjEngineImpl>(const nixlBackendInitParams *)>;
using obj_accel_create_with_clients_fn_t =
    std::function<std::unique_ptr<nixlObjEngineImpl>(const nixlBackendInitParams *,
                                                     std::shared_ptr<iS3Client>,
                                                     std::shared_ptr<iS3Client>)>;

struct objAccelEngineEntry {
    obj_accel_create_fn_t create;
    obj_accel_create_with_clients_fn_t createWithClients;
};

class objAccelEngineRegistry {
public:
    static objAccelEngineRegistry &
    instance();

    void
    add(const std::string &type, objAccelEngineEntry entry);
    bool
    has(const std::string &type) const;

    std::unique_ptr<nixlObjEngineImpl>
    create(const std::string &type, const nixlBackendInitParams *params) const;

    std::unique_ptr<nixlObjEngineImpl>
    create(const std::string &type,
           const nixlBackendInitParams *params,
           std::shared_ptr<iS3Client> s3_client,
           std::shared_ptr<iS3Client> s3_client_crt) const;

private:
    objAccelEngineRegistry() = default;
    objAccelEngineEntry
    lookupOrThrow(const std::string &type) const;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, objAccelEngineEntry> entries_;
};

struct objAccelEngineRegistrar {
    objAccelEngineRegistrar(const std::string &type,
                            obj_accel_create_fn_t create,
                            obj_accel_create_with_clients_fn_t create_with_clients) {
        objAccelEngineRegistry::instance().add(type,
                                               {std::move(create), std::move(create_with_clients)});
    }
};

#endif // NIXL_SRC_PLUGINS_OBJ_OBJ_ENGINE_REGISTRY_H
