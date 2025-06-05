/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <iostream>

#include "serdes/serdes.h"

int main() {

    const int i = 0xff;
    const std::string s = "testString";
    const std::string t1 = "i";
    const std::string t2 = "s";

    nixlSerializer nser;

    nser.addInt(t1, i);
    nser.addStr(t2, s);

    const std::string sdbuf = nser.exportStr();
    assert(sdbuf.size() > 0);

    std::cout << "exported string: " << sdbuf << "\n";

    // "nixlSDBegin|i   00000004000000ff|s   0000000AtestString|nixlSDEnd
    // |token      |tag|size.  |value.  |tag|size   |          |token

    nixlDeserializer ndes;
    nixl_status_t ret = ndes.importStr(sdbuf);
    assert(ret == 0);

    int j = 0;
    ret = ndes.getInt(t1, j);
    assert(ret == 0);
    assert(i == j);

    const std::string s2 = ndes.getStr(t2);
    assert(s2.size() > 0);
    assert(s2.compare("testString") == 0);

    return 0;
}
