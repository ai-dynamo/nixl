/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/benchmark_cli_builder.h"
#include "utils/cli/plugin_registry.h"

#include <CLI/CLI.hpp>

#include <iostream>

namespace nixlbench {
namespace {

struct providedOption {
    CLI::Option *option;
    bool *provided;
};

std::string
buildCliName(const std::string &name) {
    std::string cli_name;
    size_t start = 0;
    while (start < name.size()) {
        const size_t end = name.find(',', start);
        if (!cli_name.empty()) {
            cli_name += ',';
        }
        cli_name += "--";
        cli_name += name.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return cli_name;
}

void
bindOption(CLI::App &app, const cliOption &option, std::vector<providedOption> &provided_options) {
    std::string cli_name = buildCliName(option.name);
    std::visit(
        [&](auto *target) {
            CLI::Option *cliOption = nullptr;
            if (option.kind == option_kind_t::FLAG) {
                cliOption = app.add_flag(cli_name, *target, option.help);
            } else {
                cliOption = app.add_option(cli_name, *target, option.help);
            }
            if (option.required) {
                cliOption->required();
            }
            if (option.provided != nullptr) {
                provided_options.push_back({cliOption, option.provided});
            }
        },
        option.target);
}

void
bindOptions(CLI::App &app,
            const benchmarkCommand &command,
            std::vector<providedOption> &provided_options) {
    for (const auto &option : command.getOptions()) {
        bindOption(app, option, provided_options);
    }
}

void
bindPlugin(CLI::App &parent,
           southboundPluginBenchmarkCommand &plugin,
           southboundPluginBenchmarkCommand *&selected_plugin,
           std::vector<providedOption> &provided_options) {
    auto *subcommand = parent.add_subcommand(std::string(plugin.name()), std::string(plugin.description()));
    bindOptions(*subcommand, plugin, provided_options);
    subcommand->callback([&selected_plugin, &plugin]() { selected_plugin = &plugin; });
}

void
bindScenario(CLI::App &scenario_root,
             benchmarkScenario &scenario,
             std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>> &plugins,
             benchmarkScenario *&selected_scenario,
             southboundPluginBenchmarkCommand *&selected_plugin,
             std::vector<providedOption> &provided_options) {
    auto *subcommand = scenario_root.add_subcommand(std::string(scenario.name()),
                                                    std::string(scenario.description()));
    bindOptions(*subcommand, scenario, provided_options);
    for (auto &plugin : plugins) {
        if (scenario.supportsPlugin(plugin->pluginType()) &&
            plugin->supportsScenario(scenario.scenarioType())) {
            bindPlugin(*subcommand, *plugin, selected_plugin, provided_options);
        }
    }
    subcommand->require_subcommand(1);
    subcommand->callback([&selected_scenario, &scenario]() { selected_scenario = &scenario; });
}

} // namespace

benchmarkCliBuilder::benchmarkCliBuilder() {}

int
benchmarkCliBuilder::parse(int argc, char **argv) {
    selectedScenario_ = nullptr;
    selectedPlugin_ = nullptr;
    help_ = false;

    auto &registry = southboundPluginRegistry::instance();
    g3Plugins_ = registry.createAll();
    g4Plugins_ = registry.createAll();
    rawPlugins_ = registry.createAll();
 
    CLI::App app("NIXL Benchmark");
    app.require_subcommand(1);
    std::vector<providedOption> provided_options;

    auto *scenario_root = app.add_subcommand("scenario", "Run a benchmark scenario");
    scenario_root->require_subcommand(1);
    bindScenario(*scenario_root, g3_, g3Plugins_, selectedScenario_, selectedPlugin_, provided_options);
    bindScenario(*scenario_root, g4_, g4Plugins_, selectedScenario_, selectedPlugin_, provided_options);

    auto *raw_subcommand = app.add_subcommand(std::string(raw_.name()), std::string(raw_.description()));
    bindOptions(*raw_subcommand, raw_, provided_options);
    for (auto &plugin : rawPlugins_) {
        bindPlugin(*raw_subcommand, *plugin, selectedPlugin_, provided_options);
    }
    raw_subcommand->require_subcommand(1);
    raw_subcommand->callback([this]() { selectedScenario_ = &raw_; });

    try {
        app.parse(argc, argv);
    }
    catch (const CLI::CallForHelp &e) {
        help_ = true;
        return app.exit(e);
    }
    catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    for (const auto &provided_option : provided_options) {
        *provided_option.provided = provided_option.option->count() > 0;
    }

    if (selectedScenario_ == nullptr || selectedPlugin_ == nullptr) {
        return 1;
    }
    if (!selectedScenario_->supportsPlugin(selectedPlugin_->pluginType()) ||
        !selectedPlugin_->supportsScenario(selectedScenario_->scenarioType())) {
        std::cerr << "Unsupported scenario/plugin combination" << std::endl;
        return 1;
    }
    if (selectedScenario_->scenarioType() == scenario_type_t::RAW) {
        std::string error;
        if (!raw_.finalizeRequest(*selectedPlugin_, error)) {
            std::cerr << error << std::endl;
            return 1;
        }
    }

    return 0;
}

int
benchmarkCliBuilder::run() {
    if (help_) {
        return 0;
    }
    if (selectedScenario_ == nullptr || selectedPlugin_ == nullptr) {
        return 1;
    }
    return selectedScenario_->run(*selectedPlugin_);
}

bool
benchmarkCliBuilder::helpRequested() const {
    return help_;
}

const benchmarkScenario *
benchmarkCliBuilder::selectedScenario() const {
    return selectedScenario_;
}

const southboundPluginBenchmarkCommand *
benchmarkCliBuilder::selectedPlugin() const {
    return selectedPlugin_;
}

} // namespace nixlbench
