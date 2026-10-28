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

#ifndef NIXL_SRC_UTILS_COMMON_CONFIGURATION_H
#define NIXL_SRC_UTILS_COMMON_CONFIGURATION_H

#include "config_traits.h"
#include "exception.h"
#include "nixl_log.h"
#include "nixl_types.h"
#include "toml_traits.h"

#include <toml++/toml.hpp>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <type_traits>
#include <typeinfo>

// General design guideline for configuration handling:
// - When something does not exist and there is a fallback, use the fallback.
// - When something does exist and there is an error using it, propagate the error.
// Using the fallback in case of errors only hides the error and can lead to
// unintended effects when mistakes are hidden or ignored instead of being fixed.

namespace nixl::config {

namespace internal {

    [[nodiscard]] std::optional<std::string>
    getenvOptional(const std::string &name);

    [[nodiscard]] std::string
    getenvDefaulted(const std::string &name, const std::string &fallback);

    [[nodiscard]] toml::node_view<const toml::node>
    findTomlNode(const toml::path &path);

    [[nodiscard]] toml::node_view<const toml::node>
    findTomlNode(const std::string &path);

    void
    warnIgnoreToml(const std::string &path);

    template<typename T> struct convertTraits {
        [[nodiscard]] static decltype(auto)
        convert(const std::string &value) {
            return configTraits<T>::convert(value);
        }

        [[nodiscard]] static decltype(auto)
        convert(const toml::node_view<const toml::node> &view) {
            return tomlTraits<T>::convert(view);
        }
    };

} // namespace internal

template<typename type, template<typename...> class traits = internal::convertTraits>
[[nodiscard]] nixl_status_t
getValueWithStatus(type &result, const std::string &env) {
    if (const auto opt = internal::getenvOptional(env)) {
        try {
            result = traits<std::decay_t<type>>::convert(*opt);
        }
        catch (const std::exception &e) {
            NIXL_DEBUG << "Unable to convert environment variable '" << env << "' to target type "
                       << typeid(type).name();
            return NIXL_ERR_MISMATCH;
        }
        internal::warnIgnoreToml(env);
        return NIXL_SUCCESS;
    }

    if (const auto view = internal::findTomlNode(env)) {
        try {
            result = traits<std::decay_t<type>>::convert(view);
        }
        catch (const std::exception &e) {
            NIXL_DEBUG << "Unable to convert config value '" << env << "' to target type "
                       << typeid(type).name();
            return NIXL_ERR_MISMATCH;
        }
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_NOT_FOUND;
}

template<typename type, template<typename...> class traits = internal::convertTraits>
[[nodiscard]] type
getValue(const std::string &env) {
    if (const auto opt = internal::getenvOptional(env)) {
        auto result = traits<type>::convert(*opt);
        internal::warnIgnoreToml(env);
        return result;
    }

    if (const auto view = internal::findTomlNode(env)) {
        return traits<type>::convert(view);
    }
    throwRuntimeError("Missing config entry '", env, "'");
}

template<typename type, template<typename...> class traits = internal::convertTraits>
[[nodiscard]] std::optional<type>
getValueOptional(const std::string &env) {
    if (const auto opt = internal::getenvOptional(env)) {
        auto result = traits<type>::convert(*opt);
        internal::warnIgnoreToml(env);
        return result;
    }

    if (const auto view = internal::findTomlNode(env)) {
        return traits<type>::convert(view);
    }
    return std::nullopt;
}

template<typename type, template<typename...> class traits = internal::convertTraits>
[[nodiscard]] type
getValueDefaulted(const std::string &env, const type &fallback) {
    return getValueOptional<type, traits>(env).value_or(fallback);
}

[[nodiscard]] inline std::string
getNonEmptyString(const std::string &env) {
    const std::string result = getValue<std::string>(env);

    if (result.empty()) {
        throwRuntimeError("Config parameter '", env, "' needs non-empty value");
    }
    return result;
}

[[nodiscard]] inline bool
checkExistence(const std::string &env) {
    return (std::getenv(env.c_str()) != nullptr) || bool(internal::findTomlNode(env));
}

} // namespace nixl::config

#endif
