/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_PLUGIN_REGISTRY_H
#define NIXLBENCH_PLUGIN_REGISTRY_H

#include "utils/cli/benchmark_command.h"

#include <functional>
#include <memory>
#include <vector>

namespace nixlbench {

class SouthboundPluginRegistry {
public:
    using Factory = std::function<std::unique_ptr<ISouthboundPluginBenchmarkCommand>()>;

    static SouthboundPluginRegistry &instance();

    bool registerPlugin(Factory factory);

    std::vector<std::unique_ptr<ISouthboundPluginBenchmarkCommand>> createAll() const;

private:
    SouthboundPluginRegistry() = default;
    std::vector<Factory> factories_;
};

} // namespace nixlbench

#define REGISTER_SOUTHBOUND_PLUGIN(PluginClass)                                    \
    namespace {                                                                    \
    const bool PluginClass##_registered_ =                                         \
        ::nixlbench::SouthboundPluginRegistry::instance().registerPlugin(          \
            [] { return std::make_unique<::nixlbench::PluginClass>(); });          \
    }

#endif // NIXLBENCH_PLUGIN_REGISTRY_H
