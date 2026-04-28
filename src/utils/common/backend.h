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
#ifndef NIXL_SRC_UTILS_COMMON_BACKEND_H
#define NIXL_SRC_UTILS_COMMON_BACKEND_H

#include "configuration.h"
#include "nixl_types.h"

#include <stdexcept>
#include <string>

namespace nixl {

template<typename T>
[[nodiscard]] T
getBackendParam(const nixl_b_params_t &params, const std::string &key, const T fallback) {
    const auto it = params.find(key);

    if (it == params.end()) {
        return fallback;
    }

    try {
        return config::convertTraits<T>::convert(it->second);
    }
    catch (const std::exception &e) {
        throw std::runtime_error(e.what() + (" converting backend parameter " + key));
    }
}

template<typename T>
[[nodiscard]] T
getBackendParam(const nixl_b_params_t *params, const std::string &key, const T fallback) {
    return (params != nullptr) ? getBackendParam<T>(*params, key, fallback) : fallback;
}

} // namespace nixl

#endif
