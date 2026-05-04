/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "config.h"
#include "benchmark_runner.h"
#include <iostream>
#include "utils/utils.h"
#include <string_view>
#include "utils/cli/benchmark_cli_builder.h"

static int runConfiguredBenchmark(int argc, char *argv[]) {
    xferBenchConfig config;
    int ret = config.parseConfig(argc, argv);
    if (0 != ret) {
        return config.cliHelpRequested() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return runBenchmark(config);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        std::string_view first_arg(argv[1]);
        if (first_arg == "scenario" || first_arg == "raw" || first_arg == "--help" || first_arg == "-h") {
            nixlbench::benchmarkCliBuilder cli;
            int ret = cli.parse(argc, argv);
            if (cli.helpRequested()) {
                return EXIT_SUCCESS;
            }
            if (ret != 0) {
                return EXIT_FAILURE;
            }
            return cli.run() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }

    // Legacy support for flags-only mode
    return runConfiguredBenchmark(argc, argv);
}
