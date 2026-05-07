/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/metadata_plugin_command.h"

namespace nixlbench {
namespace {

    metadataPluginOptionValue
    makeOptionValue(const std::string &default_value,
                    nixl_backend_option_type_t type) {
        return {default_value,
                type == nixl_backend_option_type_t::BOOL &&
                    (default_value == "true" || default_value == "1"),
                false};
    }

    void
    appendFileWorkloadOptions(std::vector<cliOption> &options,
                              metadata_plugin_option_map_t &option_values) {
        auto &filepath = option_values["filepath"];
        filepath = {"", false, false};
        options.push_back({"filepath",
                           "Storage file path",
                           option_kind_t::VALUE,
                           &filepath,
                           false,
                           &filepath.isProvided});

        auto &filenames = option_values["filenames"];
        filenames = {"", false, false};
        options.push_back({"filenames",
                           "Comma seperated list of filenames to use for storage",
                           option_kind_t::VALUE,
                           &filenames,
                           false,
                           &filenames.isProvided});

        auto &num_files = option_values["num_files"];
        num_files = {"1", false, false};
        options.push_back({"num_files",
                           "Storage file count",
                           option_kind_t::VALUE,
                           &num_files,
                           false,
                           &num_files.isProvided});

        auto &enable_direct = option_values["enable_direct"];
        enable_direct = {"false", false, false};
        options.push_back({"enable_direct",
                           "Enable direct storage I/O (O_DIRECT)",
                           option_kind_t::FLAG,
                           &enable_direct,
                           false,
                           &enable_direct.isProvided});
    }

    std::vector<cliOption>
    buildOptions(const nixl_backend_option_list_t &option_specs,
                 metadata_plugin_option_map_t &option_values,
                 nixlBackendPluginCapabilities capabilities) {
        std::vector<cliOption> options;
        options.reserve(option_specs.size() + (capabilities.canReadWriteFiles ? 4 : 0));
        option_values.reserve(options.capacity());
        for (const auto &spec : option_specs) {
            const auto kind = spec.type == nixl_backend_option_type_t::BOOL ? option_kind_t::FLAG :
                                                                               option_kind_t::VALUE;
            auto &value = option_values[spec.name];
            value = makeOptionValue(spec.defaultValue, spec.type);

            options.push_back(
                {spec.name, spec.help, kind, &value, spec.required, &value.isProvided});
        }
        if (capabilities.canReadWriteFiles) {
            appendFileWorkloadOptions(options, option_values);
        }
        return options;
    }

} // namespace

metadataPluginCommand::metadataPluginCommand(std::string backend_name,
                                             nixlBackendPluginCapabilities capabilities,
                                             nixl_backend_option_list_t option_specs)
    : name_(std::move(backend_name)),
      capabilities_(capabilities),
      optionSpecs_(std::move(option_specs))
{
    // convert the optionSpecs_ to optionValues_
    options_ = buildOptions(optionSpecs_, optionValues_, capabilities_);

}

std::string_view
metadataPluginCommand::name() const {
    return name_;
}

std::string_view
metadataPluginCommand::description() const {
    return description_;
}

const std::vector<cliOption> &
metadataPluginCommand::getOptions() const {
    return options_;
}

const nixlBackendPluginCapabilities &
metadataPluginCommand::capabilities() const {
    return capabilities_;
}

const metadata_plugin_option_map_t &
metadataPluginCommand::metadataOptions() const {
    return optionValues_;
}

} // namespace nixlbench
