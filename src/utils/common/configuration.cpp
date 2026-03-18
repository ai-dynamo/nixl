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

#include "configuration.h"

namespace nixl::config {

const char *home_filename __attribute__((weak)) = ".nixl.cfg";
const char *nixl_filename __attribute__((weak)) = "/etc/nixl.cfg";
const char *nixl_variable __attribute__((weak)) = "NIXL_CONFIG_FILE";

namespace {

    [[nodiscard]] toml::table
    readFile(const std::filesystem::path &file) {
        try {
            return toml::parse_file(file.native());
        }
        catch (const std::exception &e) {
            NIXL_THROW_RUNTIME_ERROR("Error " << e.what() << " reading TOML config file " << file);
        }
    }

    [[nodiscard]] toml::table
    readFile() {
        // First choice, use $NIXL_CONFIG_FILE as config file if the environment variable is set.

        if (nixl_variable != nullptr) {
            if (const auto env = internal::getenvOptional(nixl_variable)) {
                const std::filesystem::path file(std::filesystem::weakly_canonical(*env));
                NIXL_DEBUG << "Reading nixl config file " << file << " from " << nixl_variable;
                return readFile(file);
            }
        }

        // Second choice, use ~/.nixl.cfg (default) as config file if the file exists.

        if (home_filename && *home_filename) {
            if (const auto env = internal::getenvOptional("HOME")) {
                const std::filesystem::path home(*env);
                const std::filesystem::path file =
                    std::filesystem::weakly_canonical(home / home_filename);
                if (std::filesystem::exists(file)) {
                    NIXL_DEBUG << "Reading nixl config file " << file;
                    return readFile(file);
                }
                NIXL_DEBUG << "Config file " << file << " does not exist";
            }
        }

        // Third choice, use /etc/nixl.cfg (default) as config file if the file exists.

        if (nixl_filename && *nixl_filename) {
            const std::filesystem::path file(nixl_filename);
            if (std::filesystem::exists(file)) {
                NIXL_DEBUG << "Reading nixl config file " << file;
                return readFile(file);
            }
            NIXL_DEBUG << "Config file " << file << " does not exist";
        }

        // Fallback, empty config file and only use environment variables.

        NIXL_DEBUG << "Using empty config without config file";
        return toml::table();
    }

    [[nodiscard]] const toml::table &
    getConfig() {
        static const toml::table config = readFile();
        return config;
    }

} // namespace

namespace internal {

    [[nodiscard]] std::optional<std::string>
    getenvOptional(const std::string &name) {
        if (const char *value = std::getenv(name.c_str())) {
            NIXL_DEBUG << "Found environment variable " << name << "=" << value;
            return std::string(value);
        }
        NIXL_DEBUG << "Missing environment variable " << name;
        return std::nullopt;
    }

    [[nodiscard]] std::string
    getenvDefaulted(const std::string &name, const std::string &fallback) {
        if (const char *value = std::getenv(name.c_str())) {
            NIXL_DEBUG << "Found environment variable " << name;
            return std::string(value);
        }
        NIXL_DEBUG << "Using default '" << fallback << "' for missing environment variable "
                   << name;
        return fallback;
    }

    [[nodiscard]] toml::node_view<const toml::node>
    findTomlNode(const toml::path &path) {
        const auto result = getConfig().at_path(path);
        if (result) {
            NIXL_DEBUG << "Found config file entry '" << path << "'";
        } else {
            NIXL_DEBUG << "Missing config file entry '" << path << "'";
        }
        return result;
    }

    [[nodiscard]] toml::node_view<const toml::node>
    findTomlNode(const std::string &path) {
        return findTomlNode(toml::path(path));
    }

    void
    warnIgnoreToml(const std::string &path) {
        if (getConfig().at_path(toml::path(path))) {
            NIXL_WARN << "Ignoring config file entry '" << path << "' for environment variable";
        }
    }

} // namespace internal

} // namespace nixl::config
