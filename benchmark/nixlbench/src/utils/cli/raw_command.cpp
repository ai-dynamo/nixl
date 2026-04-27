/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/raw_command.h"

namespace nixlbench {

RawCommand::RawCommand()
    : options_{CliOption::option("num-iterations,num_iter", "Benchmark iterations", &request_.num_iterations)} {}

std::string_view RawCommand::name() const { return "raw"; }

std::string_view RawCommand::description() const { return "Run compatibility benchmark command"; }

const std::vector<CliOption> &RawCommand::getOptions() const { return options_; }

const RawRequest &RawCommand::request() const { return request_; }

} // namespace nixlbench
