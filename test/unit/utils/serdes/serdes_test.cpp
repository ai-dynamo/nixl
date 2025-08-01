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
#include <cstring>
#include <iostream>

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

struct testSerializer : public nixl::serializer {
    using nixl::serializer::serializer;
    using nixl::serializer::addLongLength;
};

void
testLongLength() {
    // Test long length for 0.
    {
        testSerializer ser1;
        ser1.addLongLength(0);
        const std::string out = ser1.getBuffer();
        assert(out == std::string("N1XL\0", 5));
    }
    // Test long length for one byte.
    {
        testSerializer ser1;
        ser1.addLongLength(1);
        const std::string out = ser1.getBuffer();
        assert(out == "N1XL\xff\x01");
    }
    // Test long length for one byte.
    {
        testSerializer ser1;
        ser1.addLongLength(255);
        const std::string out = ser1.getBuffer();
        assert(out == "N1XL\xff\xff");
    }
    // Test long length for two bytes.
    {
        testSerializer ser1;
        ser1.addLongLength(258);
        const std::string out = ser1.getBuffer();
        assert(out == "N1XL\xfe\x02\x01");
    }
    // Test long length for two bytes.
    {
        testSerializer ser1;
        ser1.addLongLength(65535);
        const std::string out = ser1.getBuffer();
        assert(out == "N1XL\xfe\xff\xff");
    }
    // Test long length for three bytes.
    {
        testSerializer ser1;
        ser1.addLongLength(16777215);
        const std::string out = ser1.getBuffer();
        assert(out == "N1XL\xfd\xff\xff\xff");
    }
    // Test long length for eight bytes.
    {
        testSerializer ser1;
        ser1.addLongLength(size_t(-2));
        const std::string out = ser1.getBuffer();
        assert(out == "N1XL\xf8\xfe\xff\xff\xff\xff\xff\xff\xff");
    }
}

void
testString() {
    // Test empty serialization.
    {
        nixl::serializer ser1;
        const std::string out = ser1.getBuffer();
        const std::string moved = std::move(ser1).getBuffer();
        assert(out == moved);
        assert(out == "N1XL");
        nixlSerDes ser2;
        const std::string out2 = ser2.exportStr();
        assert(out2 == out);
        nixl::deserializer des(out.data(), out.size());
        assert(des.empty());
        nixlSerDes des2;
        des2.importStr(out);
        const std::string get2 = des2.getStr("");
        assert(get2.empty()); // Does not distinguish "empty string" from "not found".
    }
    // Test empty tag and string value.
    {
        nixl::serializer ser1;
        ser1.addString("", "");
        const std::string out = ser1.getBuffer();
        assert(out == std::string("N1XL\x00\x00", 6));
        nixlSerDes ser2;
        ser2.addStr("", "");
        const std::string out2 = ser2.exportStr();
        assert(out2 == out);
        nixl::deserializer des(out.data(), out.size());
        std::string get = "abc";
        const auto status = des.getString("", get);
        assert(status == NIXL_SUCCESS);
        assert(get == "");
        assert(des.empty());
        nixlSerDes des2;
        des2.importStr(out);
        const std::string get2 = des2.getStr("");
        assert(get2.empty()); // Does not distinguish "empty string" from "not found".
    }
    // Test short tag and short string value.
    {
        nixl::serializer ser1;
        ser1.addString("x", "yz");
        const std::string out = ser1.getBuffer();
        assert(out == "N1XL\x01x\x02yz");
        nixlSerDes ser2;
        ser2.addStr("x", "yz");
        const std::string out2 = ser2.exportStr();
        assert(out2 == out);
        nixl::deserializer des(out.data(), out.size());
        std::string get = "abc";
        const auto status = des.getString("x", get);
        assert(status == NIXL_SUCCESS);
        assert(get == "yz");
        assert(des.empty());
        nixlSerDes des2;
        des2.importStr(out);
        const std::string get2 = des2.getStr("x");
        assert(get2 == get);
    }
    // Test long tag and short string value.
    {
        nixl::serializer ser1;
        const std::string tmp = "abcdefghijklmnopqrstuvwxyz";
        const std::string str = tmp + tmp + tmp + tmp + tmp + tmp;
        assert(str.size() > 127);
        assert(str.size() < 256);
        ser1.addString(str, "xyz");
        const std::string out = ser1.getBuffer();
        const std::string ref = std::string("N1XL\xff") + char(str.size() & 0xff) + str + "\x03xyz";
        assert(out == ref);
        nixlSerDes ser2;
        ser2.addStr(str, "xyz");
        const std::string out2 = ser2.exportStr();
        assert(out2 == out);
        nixl::deserializer des(out.data(), out.size());
        std::string get = "abc";
        const auto status = des.getString(str, get);
        assert(status == NIXL_SUCCESS);
        assert(get == "xyz");
        assert(des.empty());
        nixlSerDes des2;
        des2.importStr(out);
        const std::string get2 = des2.getStr(str);
        assert(get2 == get);
    }
    // Test short tag and long string value.
    {
        nixl::serializer ser1;
        const std::string tmp = "abcdefghijklmnopqrstuvwxyz";
        const std::string str = tmp + tmp + tmp + tmp + tmp + tmp;
        assert(str.size() > 127);
        assert(str.size() < 256);
        ser1.addString("xyz", str);
        const std::string out = ser1.getBuffer();
        const std::string ref = std::string("N1XL") + "\x03xyz\xff" + char(str.size() & 0xff) + str;
        assert(out == ref);
        nixlSerDes ser2;
        ser2.addStr("xyz", str);
        const std::string out2 = ser2.exportStr();
        assert(out2 == out);
        nixl::deserializer des(out.data(), out.size());
        std::string get = "abc";
        const auto status = des.getString("xyz", get);
        assert(status == NIXL_SUCCESS);
        assert(get == str);
        assert(des.empty());
        nixlSerDes des2;
        des2.importStr(out);
        const std::string get2 = des2.getStr("xyz");
        assert(get2 == get);
    }
    // Test long tag and long string value.
    {
        nixl::serializer ser1;
        const std::string tmp = "abcdefghijklmnopqrstuvwxyz";
        const std::string str = tmp + tmp + tmp + tmp + tmp + tmp;
        assert(str.size() > 127);
        assert(str.size() < 256);
        ser1.addString(str, str);
        const std::string out = ser1.getBuffer();
        const std::string ref = std::string("N1XL") + "\xff" + char(str.size() & 0xff) + str +
            "\xff" + char(str.size() & 0xff) + str;
        assert(out == ref);
        nixlSerDes ser2;
        ser2.addStr(str, str);
        const std::string out2 = ser2.exportStr();
        assert(out2 == out);
        nixl::deserializer des(out.data(), out.size());
        std::string get = "abc";
        const auto status = des.getString(str, get);
        assert(status == NIXL_SUCCESS);
        assert(get == str);
        assert(des.empty());
        nixlSerDes des2;
        des2.importStr(out);
        const std::string get2 = des2.getStr(str);
        assert(get2 == get);
    }
}

template<typename T>
void
testIntegral(const T val, const std::string &ref) {
    // Test new serializer class.
    {
        nixl::serializer ser1;
        ser1.addIntegral("Mn", val);
        const std::string out = ser1.getBuffer();
        assert(out == "N1XL\x02Mn" + ref);
        nixl::deserializer des(out.data(), out.size());
        T get = T(-42);
        const auto status = des.getIntegral("Mn", get);
        assert(status == NIXL_SUCCESS);
        assert(get == val);
        assert(des.empty());
        assert(des.size() == 0);
    }
    // Test compatibility class.
    {
        nixlSerDes ser1;
        ser1.addBuf("Hi", &val, sizeof(val));
        const std::string out = ser1.exportStr();
        assert(out == "N1XL\x02Hi" + ref);
        {
            nixlSerDes des;
            des.importStr(out);
            T get = T(-42);
            const auto status = des.getBuf("Hi", &get, sizeof(get));
            assert(status == NIXL_SUCCESS);
            assert(get == val);
        }
        // Test with getBufLen.
        {
            nixlSerDes des;
            des.importStr(out);
            T get = T(-42);
            const auto length = des.getBufLen("Hi");
            assert(length == sizeof(get));
            const auto status = des.getBuf("Hi", &get, sizeof(get));
            assert(status == NIXL_SUCCESS);
            assert(get == val);
        }
        // Test using getStr.
        {
            nixlSerDes des;
            des.importStr(out);
            const auto get = des.getStr("Hi");
            assert(get.size() == sizeof(val));
            assert(std::memcmp(get.data(), &val, sizeof(val)) == 0);
        }
    }
}

void
testIntegral() {
    testIntegral(uint8_t(0), "\x80");
    testIntegral(uint16_t(0), "\x90");
    testIntegral(uint32_t(0), "\xa0");
    testIntegral(uint64_t(0), "\xb0");

    testIntegral(uint8_t(1), "\x81");
    testIntegral(uint16_t(2), "\x92");
    testIntegral(uint32_t(3), "\xa3");
    testIntegral(uint64_t(4), "\xb4");

    testIntegral(uint8_t(15), "\x8f");
    testIntegral(uint16_t(15), "\x9f");
    testIntegral(uint32_t(15), "\xaf");
    testIntegral(uint64_t(15), "\xbf");

    testIntegral(uint8_t(16), "\x01\x10");
    testIntegral(uint16_t(16), std::string("\x02\x10\x00", 3));
    testIntegral(uint32_t(16), std::string("\x04\x10\x00\x00\x00", 5));
    testIntegral(uint64_t(16), std::string("\x08\x10\x00\x00\x00\x00\x00\x00\x00", 9));

    testIntegral(uint8_t(-2), "\x01\xfe");
    testIntegral(uint16_t(-2), "\x02\xfe\xff");
    testIntegral(uint32_t(-2), "\x04\xfe\xff\xff\xff");
    testIntegral(uint64_t(-2), "\x08\xfe\xff\xff\xff\xff\xff\xff\xff");
}

void
testSequence() {
    nixl::serializer ser1;
    ser1.addString("a", "111");
    ser1.addString("b", "222");
    ser1.addString("f", "333");
    ser1.addString("f", "444");
    ser1.addString("h", "555");
    const std::string out = ser1.getBuffer();
    {
        nixl::deserializer des(out.data(), out.size());
        assert(!des.empty());
        std::string get;
        {
            const auto status = des.getString("a", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "111");
        }
        {
            const auto status = des.getString("b", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "222");
        }
        {
            const auto status = des.getString("f", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "333");
        }
        {
            const auto status = des.getString("h", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "555");
        }
        assert(des.empty());
    }
    {
        nixl::deserializer des(out.data(), out.size());
        assert(!des.empty());
        std::string get;
        {
            const auto status = des.getString("a", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "111");
        }
        {
            const auto status = des.getString("b", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "222");
        }
        {
            const auto status = des.getString("f", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "333");
        }
        {
            const auto status = des.getString("f", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "444");
        }
        {
            const auto status = des.getString("h", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "555");
        }
        assert(des.empty());
    }
    {
        nixl::deserializer des(out.data(), out.size());
        assert(!des.empty());
        std::string get;
        {
            const auto status = des.getString("x", get);
            assert(status == NIXL_ERR_NOT_FOUND);
            assert(!des.empty());
        }
        {
            const auto status = des.getString("a", get);
            assert(status == NIXL_SUCCESS);
            assert(get == "111");
        }
    }
    {
        nixl::deserializer des(out.data(), out.size());
        assert(!des.empty());
        std::string get;
        const auto status = des.getString("h", get);
        assert(status == NIXL_SUCCESS);
        assert(get == "555");
        assert(des.empty());
    }
}

} // namespace

int
main() {
    oldTest();
    testLongLength();
    testString();
    testIntegral();
    testSequence();
    return 0;
}
