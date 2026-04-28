/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/benchmark_cli_builder.h"

#include <CLI/CLI.hpp>

#include <iostream>
#include <utility>

namespace nixlbench {
namespace {

struct ProvidedOption {
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
bindOption(CLI::App &app, const CliOption &option, std::vector<ProvidedOption> &provided_options) {
    std::string cli_name = buildCliName(option.name);
    std::visit(
        [&](auto *target) {
            CLI::Option *cli_option = nullptr;
            if (option.kind == OptionKind::Flag) {
                cli_option = app.add_flag(cli_name, *target, option.help);
            } else {
                cli_option = app.add_option(cli_name, *target, option.help);
            }
            if (option.required) {
                cli_option->required();
            }
            if (option.provided != nullptr) {
                provided_options.push_back({cli_option, option.provided});
            }
        },
        option.target);
}

void
bindOptions(CLI::App &app,
            const IBenchmarkCommand &command,
            std::vector<ProvidedOption> &provided_options) {
    for (const auto &option : command.getOptions()) {
        bindOption(app, option, provided_options);
    }
}

void
bindPlugin(CLI::App &parent,
           ISouthboundPluginBenchmarkCommand &plugin,
           ISouthboundPluginBenchmarkCommand *&selected_plugin,
           std::vector<ProvidedOption> &provided_options) {
    auto *subcommand = parent.add_subcommand(std::string(plugin.name()), std::string(plugin.description()));
    bindOptions(*subcommand, plugin, provided_options);
    subcommand->callback([&selected_plugin, &plugin]() { selected_plugin = &plugin; });
}

void
bindScenario(CLI::App &scenario_root,
             IBenchmarkScenario &scenario,
             std::initializer_list<std::reference_wrapper<ISouthboundPluginBenchmarkCommand>> plugins,
             IBenchmarkScenario *&selected_scenario,
             ISouthboundPluginBenchmarkCommand *&selected_plugin,
             std::vector<ProvidedOption> &provided_options) {
    auto *subcommand = scenario_root.add_subcommand(std::string(scenario.name()),
                                                    std::string(scenario.description()));
    bindOptions(*subcommand, scenario, provided_options);
    for (auto plugin_ref : plugins) {
        auto &plugin = plugin_ref.get();
        if (scenario.supportsPlugin(plugin.pluginType()) &&
            plugin.supportsScenario(scenario.scenarioType())) {
            bindPlugin(*subcommand, plugin, selected_plugin, provided_options);
        }
    }
    subcommand->require_subcommand(1);
    subcommand->callback([&selected_scenario, &scenario]() { selected_scenario = &scenario; });
}

} // namespace

BenchmarkCliBuilder::BenchmarkCliBuilder()
    : raw_posix_(true) {}

int
BenchmarkCliBuilder::parse(int argc, char **argv) {
    selected_scenario_ = nullptr;
    selected_plugin_ = nullptr;
    help_ = false;

    CLI::App app("NIXL Benchmark");
    app.require_subcommand(1);
    std::vector<ProvidedOption> provided_options;

    auto *scenario_root = app.add_subcommand("scenario", "Run a benchmark scenario");
    scenario_root->require_subcommand(1);
    bindScenario(*scenario_root,
                 g3_,
                 {scenario_g3_posix_},
                 selected_scenario_,
                 selected_plugin_,
                 provided_options);
    bindScenario(*scenario_root,
                 g4_,
                 {scenario_g4_posix_, scenario_g4_obj_},
                 selected_scenario_,
                 selected_plugin_,
                 provided_options);

    auto *raw_subcommand = app.add_subcommand(std::string(raw_.name()), std::string(raw_.description()));
    bindOptions(*raw_subcommand, raw_, provided_options);
    bindPlugin(*raw_subcommand, raw_posix_, selected_plugin_, provided_options);
    bindPlugin(*raw_subcommand, raw_obj_, selected_plugin_, provided_options);
    bindPlugin(*raw_subcommand, raw_gds_, selected_plugin_, provided_options);
    bindPlugin(*raw_subcommand, raw_gds_mt_, selected_plugin_, provided_options);
    bindPlugin(*raw_subcommand, raw_gpunetio_, selected_plugin_, provided_options);
    bindPlugin(*raw_subcommand, raw_azure_blob_, selected_plugin_, provided_options);
    bindPlugin(*raw_subcommand, raw_hf3fs_, selected_plugin_, provided_options);
    bindPlugin(*raw_subcommand, raw_gusli_, selected_plugin_, provided_options);
    raw_subcommand->require_subcommand(1);
    raw_subcommand->callback([this]() { selected_scenario_ = &raw_; });

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

    if (selected_scenario_ == nullptr || selected_plugin_ == nullptr) {
        return 1;
    }
    if (!selected_scenario_->supportsPlugin(selected_plugin_->pluginType()) ||
        !selected_plugin_->supportsScenario(selected_scenario_->scenarioType())) {
        std::cerr << "Unsupported scenario/plugin combination" << std::endl;
        return 1;
    }
    if (selected_scenario_->scenarioType() == ScenarioType::Raw) {
        std::string error;
        if (!raw_.finalizeRequest(*selected_plugin_, error)) {
            std::cerr << error << std::endl;
            return 1;
        }
    }

    return 0;
}

int
BenchmarkCliBuilder::run() {
    if (help_) {
        return 0;
    }
    if (selected_scenario_ == nullptr || selected_plugin_ == nullptr) {
        return 1;
    }
    return selected_scenario_->run(*selected_plugin_);
}

bool
BenchmarkCliBuilder::helpRequested() const {
    return help_;
}

const IBenchmarkScenario *
BenchmarkCliBuilder::selectedScenario() const {
    return selected_scenario_;
}

const ISouthboundPluginBenchmarkCommand *
BenchmarkCliBuilder::selectedPlugin() const {
    return selected_plugin_;
}

} // namespace nixlbench
