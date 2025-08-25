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
#include <array>
#include <cassert>
#include <cstring>

#include "serdes/serdes.h"

void
oldMain() {
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

    nixlSerDes sd2;
    ret = sd2.importStr(sdbuf);
    assert(ret == 0);

    size_t osize = sd2.getBufLen(t1);
    assert(osize > 0);

    void *ptr = malloc(osize);
    ret = sd2.getBuf(t1, ptr, osize);
    assert(ret == 0);

    std::string s2 = sd2.getStr(t2);
    assert(s2.size() > 0);

    assert(*((int *)ptr) == 0xff);

    assert(s2.compare("testString") == 0);

    free(ptr);
}

constexpr size_t is = 4;

void
testEmpty() {
    // Test empty tag.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 5;
        const std::string val = "foo";
        sd1.addStr("", val);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr("");
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str == val);
        const std::string enc("N1XL\0\x03"
                              "foo",
                              9);
        assert(sd1.exportStr() == enc);
        // Test buffer add compatibility.
        {
            nixlSerDes sd2;
            sd2.addBuf("", val.data(), val.size());
            assert(sd1.exportStr() == sd2.exportStr());
        }
        // Test buffer compatibility without getBufLen.
        {
            nixlSerDes sd2;
            assert(sd2.importStr(sd1.exportStr()) == NIXL_SUCCESS);
            std::array<char, 3> buf;
            assert(sd2.getBuf("", buf.data(), buf.size()) == NIXL_SUCCESS);
            assert(std::memcmp(buf.data(), val.data(), 3) == 0);
        }
        // Test buffer compatibility with getBufLen.
        {
            nixlSerDes sd2;
            assert(sd2.importStr(sd1.exportStr()) == NIXL_SUCCESS);
            const auto size = sd2.getBufLen("");
            assert(size == 3);
            std::array<char, 3> buf;
            assert(sd2.getBuf("", buf.data(), buf.size()) == NIXL_SUCCESS);
            assert(std::memcmp(buf.data(), val.data(), 3) == 0);
        }
    }
    // Test empty string.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 5;
        const std::string key = "bar";
        sd1.addStr(key, "");
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr(key);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str.empty());
        const std::string enc("N1XL\x03"
                              "bar\0",
                              9);
        assert(sd1.exportStr() == enc);
        // Test buffer add compatibility.
        {
            nixlSerDes sd2;
            sd2.addBuf(key, "", 0);
            assert(sd1.exportStr() == sd2.exportStr());
        }
        // Test buffer compatibility without getBufLen.
        {
            nixlSerDes sd2;
            assert(sd2.importStr(sd1.exportStr()) == NIXL_SUCCESS);
            std::array<char, 1> buf;
            assert(sd2.getBuf(key, buf.data(), 0) == NIXL_SUCCESS);
        }
        // Test buffer compatibility with getBufLen.
        {
            nixlSerDes sd2;
            assert(sd2.importStr(sd1.exportStr()) == NIXL_SUCCESS);
            const auto size = sd2.getBufLen(key);
            assert(size == 0);
            std::array<char, 1> buf;
            assert(sd2.getBuf(key, buf.data(), 0) == NIXL_SUCCESS);
        }
    }
    // Test empty tag and empty string.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 2;
        sd1.addStr("", "");
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr("");
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str.empty());
        const std::string enc("N1XL\0\0", 6);
        assert(sd1.exportStr() == enc);
        // Test buffer add compatibility.
        {
            nixlSerDes sd2;
            sd2.addBuf("", "", 0);
            assert(sd1.exportStr() == sd2.exportStr());
        }
        // Test buffer compatibility without getBufLen.
        {
            nixlSerDes sd2;
            assert(sd2.importStr(sd1.exportStr()) == NIXL_SUCCESS);
            std::array<char, 1> buf;
            assert(sd2.getBuf("", buf.data(), 0) == NIXL_SUCCESS);
        }
        // Test buffer compatibility with getBufLen.
        {
            nixlSerDes sd2;
            assert(sd2.importStr(sd1.exportStr()) == NIXL_SUCCESS);
            const auto size = sd2.getBufLen("");
            assert(size == 0);
            std::array<char, 1> buf;
            assert(sd2.getBuf("", buf.data(), 0) == NIXL_SUCCESS);
        }
    }
}

void
testBufString(const std::string &buffer, const std::string &key, const std::string &val) {
    // Test buffer add compatibility.
    {
        nixlSerDes sd2;
        sd2.addBuf(key, val.data(), val.size());
        assert(buffer == sd2.exportStr());
    }
    // Test buffer compatibility without getBufLen.
    {
        nixlSerDes sd2;
        assert(sd2.importStr(buffer) == NIXL_SUCCESS);
        std::vector<char> buf;
        buf.resize(val.size());
        assert(sd2.getBuf(key, buf.data(), buf.size()) == NIXL_SUCCESS);
        assert(std::memcmp(buf.data(), val.data(), val.size()) == 0);
    }
    // Test buffer compatibility with getBufLen.
    {
        nixlSerDes sd2;
        assert(sd2.importStr(buffer) == NIXL_SUCCESS);
        const size_t size = sd2.getBufLen(key);
        assert(size == val.size());
        std::vector<char> buf;
        buf.resize(val.size());
        assert(sd2.getBuf(key, buf.data(), buf.size()) == NIXL_SUCCESS);
        assert(std::memcmp(buf.data(), val.data(), val.size()) == 0);
    }
}

void
testShortString() {
    // Test shortest.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 4;
        sd1.addStr("k", "v");
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr("k");
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str == "v");
        const std::string enc("N1XL\x01"
                              "k\x01v");
        assert(sd1.exportStr() == enc);
        testBufString(sd1.exportStr(), "k", "v");
    }
    // Test longest.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 256;
        const std::string key = "Kabcdefghijklmnopqrstuvwxyz" + std::string(100, 'k');
        const std::string val = "vABCDEFGHIJKLMNOPQRSTUVWXYZ" + std::string(100, 'V');
        sd1.addStr(key, val);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr(key);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str == val);
        const std::string enc("N1XL\x7f" + key + "\x7f" + val);
        assert(sd1.exportStr() == enc);
        testBufString(sd1.exportStr(), key, val);
    }
}

void
testLongString() {
    // Test shortest with 1 length byte.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 260;
        const std::string key = "Kabcdefghijklmnopqrstuvwxyz" + std::string(101, 'k');
        const std::string val = "vABCDEFGHIJKLMNOPQRSTUVWXYZ" + std::string(101, 'V');
        sd1.addStr(key, val);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr(key);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str == val);
        const std::string enc("N1XL\xC1\x01" + key + "\xC1\x01" + val);
        assert(sd1.exportStr() == enc);
        testBufString(sd1.exportStr(), key, val);
    }
    // Test longest with 1 length byte.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 768;
        const std::string key = "Kabcdefghijklmnopqrstuvwxyz" + std::string(355, 'k');
        const std::string val = "vABCDEFGHIJKLMNOPQRSTUVWXYZ" + std::string(355, 'V');
        sd1.addStr(key, val);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr(key);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str == val);
        const std::string enc("N1XL\xC1\xFF" + key + "\xC1\xFF" + val);
        assert(sd1.exportStr() == enc);
        testBufString(sd1.exportStr(), key, val);
    }
    // Test shortest with 2 length bytes.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 772;
        const std::string key = "Kabcdefghijklmnopqrstuvwxyz" + std::string(356, 'k');
        const std::string val = "vABCDEFGHIJKLMNOPQRSTUVWXYZ" + std::string(356, 'V');
        sd1.addStr(key, val);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr(key);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str == val);
        const std::string enc(std::string("N1XL\xC2\x00\x01", 7) + key +
                              std::string("\xC2\x00\x01", 3) + val);
        assert(sd1.exportStr() == enc);
        testBufString(sd1.exportStr(), key, val);
    }
    // Test longest with 2 length bytes.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 131330;
        const std::string key = "Kabcdefghijklmnopqrstuvwxyz" + std::string(65635, 'k');
        const std::string val = "vABCDEFGHIJKLMNOPQRSTUVWXYZ" + std::string(65635, 'V');
        sd1.addStr(key, val);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr(key);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str == val);
        const std::string enc("N1XL\xC2\xFF\xFF" + key + "\xC2\xFF\xFF" + val);
        assert(sd1.exportStr() == enc);
        testBufString(sd1.exportStr(), key, val);
    }
    // Test shortest with 3 length bytes.
    {
        nixlSerDes sd1;
        constexpr size_t ds = 131334;
        const std::string key = "Kabcdefghijklmnopqrstuvwxyz" + std::string(65636, 'k');
        const std::string val = "vABCDEFGHIJKLMNOPQRSTUVWXYZ" + std::string(65636, 'V');
        sd1.addStr(key, val);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == ds);
        const auto str = sd1.getStr(key);
        assert(sd1.totalSize() == is + ds);
        assert(sd1.remainingSize() == 0);
        assert(str == val);
        const std::string enc(std::string("N1XL\xC3\x00\x00\x01", 8) + key +
                              std::string("\xC3\x00\x00\x01", 4) + val);
        assert(sd1.exportStr() == enc);
        testBufString(sd1.exportStr(), key, val);
    }
    // Assume that things work for greater lengths.
}

void
testTagMismatch() {
    nixlSerDes sd1;
    sd1.addStr("foo", "bar");
    assert(sd1.remainingSize() == 8);
    assert(sd1.getBufLen("foo") == 3);
    assert(sd1.remainingSize() == 8);
    assert(sd1.getBufLen("baz") == -1);
    assert(sd1.remainingSize() == 8);
    char buf[3];
    assert(sd1.getBuf("baz", buf, sizeof(buf)) == NIXL_ERR_NOT_FOUND);
    assert(sd1.remainingSize() == 8);
    assert(sd1.getStr("baz") == "");
    assert(sd1.remainingSize() == 8);
}

int
main() {
    oldMain();
    testEmpty();
    testShortString();
    testLongString();
    testTagMismatch();
    return 0;
}
