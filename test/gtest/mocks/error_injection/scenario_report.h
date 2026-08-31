/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * Turns finished scenarios into output: an aligned table for reading, or csv for
 * a spreadsheet. Column widths and the verdict wording live in the .cpp, so a
 * driver only decides which rows to emit.
 */
#ifndef TEST_GTEST_MOCKS_ERROR_INJECTION_SCENARIO_REPORT_H
#define TEST_GTEST_MOCKS_ERROR_INJECTION_SCENARIO_REPORT_H

#include <cstddef>

#include "mocks/error_injection/error_injection.h"

namespace mocks::error_injection {

/* Whether the run agreed with the status and side effect the scenario declared. */
bool
passed(const scenario &s, const observation &obs);

/* Column names, preceded by a title and a rule unless csv is set. */
void
printHeader(bool csv);

void
printRow(const scenario &s, const observation &obs, bool csv);

/* The scenario and failure counts. Suppressed for csv, which stays parsable. */
void
printSummary(size_t selected, size_t failures, bool csv);

} // namespace mocks::error_injection

#endif // TEST_GTEST_MOCKS_ERROR_INJECTION_SCENARIO_REPORT_H
