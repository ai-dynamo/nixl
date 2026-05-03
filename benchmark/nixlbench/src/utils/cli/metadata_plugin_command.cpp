/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/metadata_plugin_command.h"

namespace nixlbench {
namespace {

    void
    appendFileWorkloadOptions(std::vector<cliOption> &options, fileWorkloadRequest &file_workload) {
        options.push_back(cliOption::option(
            "filepath", "Storage file path", &file_workload.filepath));
        options.push_back(
            cliOption::option("filenames", "Storage filenames", &file_workload.filenames));
        options.push_back(cliOption::option(
            "num_files", "Storage file count", &file_workload.num_files));
        options.push_back(cliOption::flag("enable_direct",
                                          "Enable direct storage I/O (O_DIRECT)",
                                          &file_workload.enable_direct));
    }

    std::vector<cliOption>
    buildOptions(const nixl_backend_option_list_t &option_specs,
                 metadata_plugin_option_map_t &option_values,
                 nixlBackendPluginCapabilities capabilities,
                 fileWorkloadRequest &file_workload) {
        std::vector<cliOption> options;
        options.reserve(option_specs.size() + (capabilities.canReadWriteFiles ? 4 : 0));
        for (const auto &spec : option_specs) {
            const auto kind = spec.type == nixl_backend_option_type_t::BOOL ? option_kind_t::FLAG :
                                                                               option_kind_t::VALUE;
            auto &value = option_values[spec.name];
            if (spec.type == nixl_backend_option_type_t::BOOL) {
                value.boolValue = spec.defaultValue == "true" || spec.defaultValue == "1";
            } else {
                value.value = spec.defaultValue;
            }

            options.push_back(
                {spec.name, spec.help, kind, &value, spec.required, &value.isProvided});
        }
        if (capabilities.canReadWriteFiles) {
            appendFileWorkloadOptions(options, file_workload);
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
    optionValues_.reserve(optionSpecs_.size());
    options_ = buildOptions(optionSpecs_, optionValues_, capabilities_, fileWorkload_);
    for (const auto &spec : optionSpecs_) {
        optionValues_[spec.name] = {
            spec.defaultValue, spec.type == nixl_backend_option_type_t::BOOL ? true : false, false};
    }

    if (capabilities.canReadWriteFiles) {
        optionValues_["filepath"] = {
            fileWorkload_.filepath.value, false, false};
        optionValues_["filenames"] = {
            fileWorkload_.filenames.value, false, false};
        optionValues_["num_files"] = {
            std::to_string(fileWorkload_.num_files.value), false, false};
        optionValues_["enable_direct"] = {
            fileWorkload_.enable_direct.value ? "true" : "false", true, false};
    }

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
