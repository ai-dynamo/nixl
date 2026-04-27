/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/benchmark_cli_builder.h"

#include "utils/cli/benchmark_command.h"
#include "utils/cli/g3_scenario_command.h"
#include "utils/cli/g4_scenario_command.h"
#include "utils/cli/obj_plugin_command.h"
#include "utils/cli/posix_plugin_command.h"
#include "utils/cli/raw_command.h"

#include <CLI/CLI.hpp>

#include <iostream>

namespace nixlbench {
namespace {

void
bindOption(CLI::App &app, const CliOption &option) {
    std::string cli_name = "--" + option.name;
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
        },
        option.target);
}

void
bindOptions(CLI::App &app, const IBenchmarkCommand &command) {
    for (const auto &option : command.getOptions()) {
        bindOption(app, option);
    }
}

void
bindPlugin(CLI::App &parent,
           const ISouthboundPluginBenchmarkCommand &plugin,
           CLI::App *&selected_plugin,
           PluginType &selected_type) {
    auto *subcommand = parent.add_subcommand(std::string(plugin.name()), std::string(plugin.description()));
    bindOptions(*subcommand, plugin);
    subcommand->callback([&selected_plugin, &selected_type, subcommand, &plugin]() {
        selected_plugin = subcommand;
        selected_type = plugin.pluginType();
    });
}

template <typename ScenarioCommand>
CLI::App *
bindScenario(CLI::App &scenario_root,
             ScenarioCommand &scenario,
             PosixPluginCommand &posix,
             ObjPluginCommand &obj,
             CLI::App *&selected_scenario,
             ScenarioType &selected_type,
             CLI::App *&selected_plugin,
             PluginType &selected_plugin_type) {
    auto *subcommand = scenario_root.add_subcommand(std::string(scenario.name()),
                                                    std::string(scenario.description()));
    bindOptions(*subcommand, scenario);
    if (posix.supportsScenario(scenario.scenarioType())) {
        bindPlugin(*subcommand, posix, selected_plugin, selected_plugin_type);
    }
    if (obj.supportsScenario(scenario.scenarioType())) {
        bindPlugin(*subcommand, obj, selected_plugin, selected_plugin_type);
    }
    subcommand->require_subcommand(1);
    subcommand->callback([&selected_scenario, &selected_type, subcommand, &scenario]() {
        selected_scenario = subcommand;
        selected_type = scenario.scenarioType();
    });
    return subcommand;
}

} // namespace

int
BenchmarkCliBuilder::parse(int argc, char **argv, ParsedBenchmarkCommand &parsed) const {
    G3ScenarioCommand g3;
    G4ScenarioCommand g4;
    RawCommand raw;
    PosixPluginCommand scenario_posix;
    ObjPluginCommand scenario_obj;
    PosixPluginCommand raw_posix(true);

    CLI::App app("NIXL Benchmark");
    app.require_subcommand(1);

    CLI::App *selected_scenario = nullptr;
    ScenarioType selected_scenario_type = ScenarioType::None;
    CLI::App *selected_plugin = nullptr;
    PluginType selected_plugin_type = PluginType::None;
    CLI::App *selected_raw_plugin = nullptr;
    PluginType selected_raw_plugin_type = PluginType::None;

    auto *scenario_root = app.add_subcommand("scenario", "Run a benchmark scenario");
    scenario_root->require_subcommand(1);
    bindScenario(*scenario_root,
                 g3,
                 scenario_posix,
                 scenario_obj,
                 selected_scenario,
                 selected_scenario_type,
                 selected_plugin,
                 selected_plugin_type);
    bindScenario(*scenario_root,
                 g4,
                 scenario_posix,
                 scenario_obj,
                 selected_scenario,
                 selected_scenario_type,
                 selected_plugin,
                 selected_plugin_type);

    auto *raw_subcommand = app.add_subcommand(std::string(raw.name()), std::string(raw.description()));
    bindOptions(*raw_subcommand, raw);
    bindPlugin(*raw_subcommand, raw_posix, selected_raw_plugin, selected_raw_plugin_type);
    raw_subcommand->require_subcommand(1);

    try {
        app.parse(argc, argv);
    }
    catch (const CLI::CallForHelp &e) {
        parsed.path = CommandPath::Help;
        return app.exit(e);
    }
    catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    if (*raw_subcommand) {
        parsed.path = CommandPath::Raw;
        parsed.plugin = selected_raw_plugin ? selected_raw_plugin_type : PluginType::None;
        parsed.raw = raw.request();
        if (parsed.plugin == PluginType::Posix) {
            parsed.posix = raw_posix.request();
        }
        return 0;
    }

    if (*scenario_root) {
        parsed.path = CommandPath::Scenario;
        parsed.scenario = selected_scenario ? selected_scenario_type : ScenarioType::None;
        parsed.plugin = selected_plugin ? selected_plugin_type : PluginType::None;
        if (parsed.scenario == ScenarioType::G3) {
            parsed.g3 = g3.request();
        } else if (parsed.scenario == ScenarioType::G4) {
            parsed.g4 = g4.request();
        }
        if (parsed.plugin == PluginType::Posix) {
            parsed.posix = scenario_posix.request();
        } else if (parsed.plugin == PluginType::Obj) {
            parsed.obj = scenario_obj.request();
        }
        return 0;
    }

    return 1;
}

} // namespace nixlbench
