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
#include <iostream>
#include <vector>

#include "serdes/serdes.h"

namespace {
void
oldTest() {
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

    // std::cout << "exported data: " << sdbuf << std::endl;;

    nixlSerDes sd2;
    ret = sd2.importStr(sdbuf);
    assert(ret == 0);

    size_t osize = sd2.getBufLen(t1);
    assert(osize == 4);

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
testEmptyString() {
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

template<typename T>
void
testIntegral(const T val, const std::string &ref) {
    nixlSerDes sd1;
    const std::string key = "key";
    const std::string enc = "N1XL\x03" + key + ref;
    assert(sd1.addBuf(key, &val, sizeof(val)) == NIXL_SUCCESS);
    assert(sd1.totalSize() == enc.size());
    assert(sd1.remainingSize() == enc.size() - is);
    assert(sd1.exportStr() == enc);

    // Test without getBufLen.
    {
        nixlSerDes sd2;
        assert(sd2.importStr(enc) == NIXL_SUCCESS);
        T get = T(0xdeaddead);
        assert(sd2.getBuf(key, &get, sizeof(get)) == NIXL_SUCCESS);
        assert(get == val);
        assert(sd2.remainingSize() == 0);
    }
    // Test with getBufLen.
    {
        nixlSerDes sd2;
        assert(sd2.importStr(enc) == NIXL_SUCCESS);
        const size_t size = sd2.getBufLen(key);
        assert(size == sizeof(val));
        T get = T(0xdeaddead);
        assert(sd2.getBuf(key, &get, sizeof(get)) == NIXL_SUCCESS);
        assert(get == val);
        assert(sd2.remainingSize() == 0);
    }
}

void
testIntegral() {
    testIntegral(std::uint8_t(0), "\x80");
    testIntegral(std::uint8_t(1), "\x81");
    testIntegral(std::uint8_t(15), "\x8F");
    testIntegral(std::uint8_t(16), "\x01\x10");
    testIntegral(std::uint8_t(254), "\x01\xFE");
    testIntegral(std::uint8_t(255), "\x01\xFF");

    testIntegral(std::uint16_t(0), "\x90");
    testIntegral(std::uint16_t(1), "\x91");
    testIntegral(std::uint16_t(15), "\x9F");
    testIntegral(std::uint16_t(16), "\xD0\x10");
    testIntegral(std::uint16_t(254), "\xD0\xFE");
    testIntegral(std::uint16_t(255), "\xD0\xFF");
    testIntegral(std::uint16_t(256), std::string("\x02\x00\x01", 3));
    testIntegral(std::uint16_t(258), "\x02\x02\x01");
    testIntegral(std::uint16_t(-2), "\x02\xFE\xFF");
    testIntegral(std::uint16_t(-1), "\x02\xFF\xFF");

    testIntegral(std::uint32_t(0), "\xA0");
    testIntegral(std::uint32_t(1), "\xA1");
    testIntegral(std::uint32_t(15), "\xAF");
    testIntegral(std::uint32_t(16), "\xD1\x10");
    testIntegral(std::uint32_t(254), "\xD1\xFE");
    testIntegral(std::uint32_t(255), "\xD1\xFF");
    testIntegral(std::uint32_t(256), std::string("\xD2\x00\x01", 3));
    testIntegral(std::uint32_t(258), "\xD2\x02\x01");
    testIntegral(std::uint32_t(65534), "\xD2\xFE\xFF");
    testIntegral(std::uint32_t(65535), "\xD2\xFF\xFF");
    testIntegral(std::uint32_t(65536), std::string("\xD3\x00\x00\x01", 4));
    testIntegral(std::uint32_t(65538), std::string("\xD3\x02\x00\x01", 4));
    testIntegral(std::uint32_t(-2), "\x04\xFE\xFF\xFF\xFF");
    testIntegral(std::uint32_t(-1), "\x04\xFF\xFF\xFF\xFF");

    testIntegral(std::uint64_t(0), "\xB0");
    testIntegral(std::uint64_t(1), "\xB1");
    testIntegral(std::uint64_t(15), "\xBF");
    testIntegral(std::uint64_t(16), "\xD4\x10");
    testIntegral(std::uint64_t(254), "\xD4\xFE");
    testIntegral(std::uint64_t(255), "\xD4\xFF");
    testIntegral(std::uint64_t(256), std::string("\xD5\x00\x01", 3));
    testIntegral(std::uint64_t(258), "\xD5\x02\x01");
    testIntegral(std::uint64_t(65534), "\xD5\xFE\xFF");
    testIntegral(std::uint64_t(65535), "\xD5\xFF\xFF");
    testIntegral(std::uint64_t(65536), std::string("\xD6\x00\x00\x01", 4));
    testIntegral(std::uint64_t(65538), std::string("\xD6\x02\x00\x01", 4));
    testIntegral(std::uint64_t(std::uint32_t(-2)), "\xD7\xFE\xFF\xFF\xFF");
    testIntegral(std::uint64_t(std::uint32_t(-1)), "\xD7\xFF\xFF\xFF\xFF");
    testIntegral(std::uint64_t(-2), "\x08\xFE\xFF\xFF\xFF\xFF\xFF\xFF\xFF");
    testIntegral(std::uint64_t(-1), "\x08\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF");
}

void
testSpecials() {
    // Test string encoding length 1 small value.
    {
        nixlSerDes sd1;
        const std::string key = "k";
        assert(sd1.addStr(key, "\x02") == NIXL_SUCCESS);
        assert(sd1.exportStr() == "N1XL\x01k\x01\x02");
    }
    // Test getBufLen stability.
    {
        nixlSerDes sd1;
        assert(sd1.addStr("a", "b") == NIXL_SUCCESS);
        assert(sd1.addStr("c", "d") == NIXL_SUCCESS);
        assert(sd1.addStr("e", "f") == NIXL_SUCCESS);
        assert(sd1.totalSize() == 16);
        assert(sd1.remainingSize() == 12);
        assert(sd1.getBufLen("x") == -1);
        assert(sd1.remainingSize() == 12);
        assert(sd1.getBufLen("c") == 1);
        assert(sd1.remainingSize() == 8);
        assert(sd1.getBufLen("x") == -1);
        assert(sd1.remainingSize() == 8);
    }
    // Test duplicate keys.
    {
        nixlSerDes sd1;
        assert(sd1.addStr("A", "1") == NIXL_SUCCESS);
        assert(sd1.addStr("A", "2") == NIXL_SUCCESS);
        assert(sd1.addStr("A", "3") == NIXL_SUCCESS);
        assert(sd1.getStr("A") == "1");
        assert(sd1.getStr("A") == "2");
        assert(sd1.getStr("A") == "3");
    }
    // Test duplicate keys with intermediates.
    {
        nixlSerDes sd1;
        assert(sd1.addStr("A", "1") == NIXL_SUCCESS);
        assert(sd1.addStr("B", "2") == NIXL_SUCCESS);
        assert(sd1.addStr("C", "3") == NIXL_SUCCESS);
        assert(sd1.addStr("A", "4") == NIXL_SUCCESS);
        assert(sd1.addStr("A", "5") == NIXL_SUCCESS);
        assert(sd1.addStr("D", "6") == NIXL_SUCCESS);
        assert(sd1.getStr("A") == "1");
        assert(sd1.getStr("A") == "4");
        assert(sd1.getStr("A") == "5");
    }
}

void
testBadCases() {
    // Get int as string size 1.
    {
        nixlSerDes sd1;
        const std::string key = "k";
        const uint8_t val = 0;
        sd1.addBuf(key, &val, sizeof(val));
        assert(sd1.getStr(key) == "");
        assert(sd1.getBufLen(key) == sizeof(val));
    }
    // Test invalid key encoding stability.
    {
        nixlSerDes sd1;
        assert(sd1.importStr("N1XL\xFF" + std::string(300, '\0')) == NIXL_SUCCESS);
        const size_t size = sd1.remainingSize();
        assert(sd1.getStr("k") == "");
        assert(sd1.remainingSize() == size);
        assert(sd1.getBufLen("k") == -1);
        assert(sd1.remainingSize() == size);
    }
    // Test invalid value encoding stability.
    {
        nixlSerDes sd1;
        assert(sd1.importStr("N1XL\x01k\xFF" + std::string(300, '\0')) == NIXL_SUCCESS);
        const size_t size = sd1.remainingSize();
        assert(sd1.getStr("k") == "");
        assert(sd1.remainingSize() == size);
        assert(sd1.getBufLen("k") == -1);
        assert(sd1.remainingSize() == size);
    }
    // Test incomplete data stability.
    {
        nixlSerDes sd1;
        assert(sd1.importStr("N1XL\x01k\x01") == NIXL_SUCCESS);
        const size_t size = sd1.remainingSize();
        assert(sd1.getStr("k") == "");
        assert(sd1.remainingSize() == size);
        assert(sd1.getBufLen("k") == 1);
        assert(sd1.remainingSize() == size);
        char dummy;
        assert(sd1.getBuf("l", &dummy, 1) != NIXL_SUCCESS);
        assert(sd1.remainingSize() == size);
    }
}

} // namespace

int
main() {
    oldTest();
    testEmptyString();
    testShortString();
    testLongString();
    testIntegral();
    testSpecials();
    testBadCases();
    return 0;
}
