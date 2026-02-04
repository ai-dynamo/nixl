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
#include "serdes/serdes.h"
#include <cassert>
#include <iostream>

int main() {

    int i = 0xff;
    std::string s = "testString";
    std::string t1 = "i", t2 = "s";
    int ret;

    nixlSerDes sd;

    ret = sd.addBuf(t1, &i, sizeof(i));
    assert(ret == 0);

    ret = sd.addStr(t2, s);
    assert(ret == 0);

    std::string sdbuf = sd.exportStr();
    assert(sdbuf.size() > 0);

    // "nixlSDBegin|i   00000004000000ff|s   0000000AtestString|nixlSDEnd
    // |token      |tag|size.  |value.  |tag|size   |          |token

    nixlSerDes sd2;
    ret = sd2.importStr(sdbuf);
    assert(ret == 0);

    size_t osize = sd2.getBufLen(t1);
    assert(osize > 0);

    void *ptr = malloc(osize);
    ret = sd2.getBuf(t1, ptr, osize);
    assert(ret == 0);

    std::string s2 =  sd2.getStr(t2);
    assert(s2.size() > 0);

    assert(*((int*) ptr) == 0xff);

    assert(s2.compare("testString") == 0);

    free(ptr);

    return 0;
}
