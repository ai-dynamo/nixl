/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_RAW_COMMAND_H
#define NIXLBENCH_RAW_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class RawCommand : public IBenchmarkCommand {
public:
    RawCommand();

    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    const RawRequest &request() const;

private:
    RawRequest request_;
    std::vector<CliOption> options_;
};

} // namespace nixlbench

#endif // NIXLBENCH_RAW_COMMAND_H
