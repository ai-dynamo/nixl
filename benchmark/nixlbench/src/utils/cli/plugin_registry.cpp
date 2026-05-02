/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/plugin_registry.h"

namespace nixlbench {

southboundPluginRegistry &
southboundPluginRegistry::instance() {
    static southboundPluginRegistry registry;
    return registry;
}

bool
southboundPluginRegistry::registerPlugin(Factory factory) {
    factories_.push_back(std::move(factory));
    return true;
}

std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>>
southboundPluginRegistry::createAll() const {
    std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>> plugins;
    plugins.reserve(factories_.size());
    for (const auto &factory : factories_) {
        plugins.push_back(factory());
    }
    return plugins;
}

} // namespace nixlbench
