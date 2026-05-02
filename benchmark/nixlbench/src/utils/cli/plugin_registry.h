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

class southboundPluginRegistry {
public:
    using Factory = std::function<std::unique_ptr<southboundPluginBenchmarkCommand>()>;

    static southboundPluginRegistry &instance();

    bool registerPlugin(Factory factory);

    std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>> createAll() const;

private:
    southboundPluginRegistry() = default;
    std::vector<Factory> factories_;
};

} // namespace nixlbench

#define REGISTER_SOUTHBOUND_PLUGIN(plugin_class)                                    \
    namespace {                                                                    \
    const bool plugin_class##_registered_ =                                         \
        ::nixlbench::southboundPluginRegistry::instance().registerPlugin(          \
            [] { return std::make_unique<::nixlbench::plugin_class>(); });          \
    }

#endif // NIXLBENCH_PLUGIN_REGISTRY_H
