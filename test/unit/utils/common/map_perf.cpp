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
#include <iostream>
#include <random>
#include <algorithm>
#include <map>
#include <unordered_map>

#include <sys/time.h>

#include "common/str_tools.h"
#include "test_utils.h"

std::string generate_random_string(size_t length) {
    const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_int_distribution<> distribution(0, characters.size() - 1);

    std::string random_string;
    for (size_t i = 0; i < length; ++i) {
        random_string += characters[distribution(generator)];
    }
    return random_string;
}

void test_comparison_perf(const int n_entries, const size_t str_len) {

    int n_iters = 1000000;

    std::unordered_map<std::string, uint32_t> normal_map;
    std::unordered_map<std::string, uint32_t, std::hash<std::string>, strEqual> custom_map;
    std::map<std::string, uint32_t> ordered_map;

    std::vector<std::string> ref(n_entries);

    struct timeval start_time, end_time, diff_time;
    uint64_t sum1 = 0, sum2 = 0;

    std::cout << "testing maps with " << n_entries << " entries and " << str_len << " len \n";

    for(int i = 0; i<n_entries; i++) {
        std::string index = generate_random_string(str_len);
        ref[i] = index;
    }

    gettimeofday(&start_time, NULL);
    for(int i = 0; i<n_iters; i++) {
        std::string index = ref.at(i % n_entries);
        normal_map[index] = i;
    }
    gettimeofday(&end_time, NULL);

    timersub(&end_time, &start_time, &diff_time);

    std::cout << "normal map insert test, total time for " << n_iters << " iters: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";

    gettimeofday(&start_time, NULL);
    for(int i = 0; i<n_iters; i++) {
        std::string index = ref.at(i % n_entries);
        custom_map[index] = i;
    }
    gettimeofday(&end_time, NULL);

    timersub(&end_time, &start_time, &diff_time);

    std::cout << "custom map insert test, total time for " << n_iters << " iters: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";

    gettimeofday(&start_time, NULL);
    for(int i = 0; i<n_iters; i++) {
        std::string index = ref.at(i % n_entries);
        ordered_map[index] = i;
    }
    gettimeofday(&end_time, NULL);

    timersub(&end_time, &start_time, &diff_time);

    std::cout << "ordered map insert test, total time for " << n_iters << " iters: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";

    gettimeofday(&start_time, NULL);
    for(int i = 0; i<n_iters; i++) {
        std::string index = ref.at(i % n_entries);
        sum1 += normal_map[index];
    }
    gettimeofday(&end_time, NULL);

    timersub(&end_time, &start_time, &diff_time);

    std::cout << "normal map lookup test, total time for " << n_iters << " iters: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";

    gettimeofday(&start_time, NULL);
    for(int i = 0; i<n_iters; i++) {
        std::string index = ref.at(i % n_entries);
        sum2 += custom_map[index];
    }
    gettimeofday(&end_time, NULL);

    timersub(&end_time, &start_time, &diff_time);

    std::cout << "custom map lookup test, total time for " << n_iters << " iters: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";

    nixl_exit_on_failure((sum1 == sum2), "Test failed", "test");

    gettimeofday(&start_time, NULL);
    for(int i = 0; i<n_iters; i++) {
        std::string index = ref.at(i % n_entries);
        sum2 += ordered_map[index];
    }
    gettimeofday(&end_time, NULL);

    timersub(&end_time, &start_time, &diff_time);

    std::cout << "ordered map lookup test, total time for " << n_iters << " iters: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";
}

int main()
{
    strEqual tester;
    nixl_exit_on_failure(
        (tester.operator()("abcdefgh", "abcdefgh") == true), "Test failed", "test");
    nixl_exit_on_failure(
        (tester.operator()("abcdefgh", "abdcefgh") == false), "Test failed", "test");
    nixl_exit_on_failure(
        (tester.operator()("abcdefgh123", "abcdefgh123") == true), "Test failed", "test");
    nixl_exit_on_failure(
        (tester.operator()("abcdefgh123", "aadcefgh123") == false), "Test failed", "test");
    nixl_exit_on_failure(
        (tester.operator()("12345678abcdefgh", "12345678abcdefgh") == true), "Test failed", "test");
    nixl_exit_on_failure((tester.operator()("12345678abcdefgh", "12345687abcdefgh") == false),
                         "Test failed",
                         "test");

    test_comparison_perf(16, 8);
    test_comparison_perf(16, 16);
    test_comparison_perf(128, 8);
    test_comparison_perf(128, 16);
    test_comparison_perf(1000, 8);
    test_comparison_perf(1000, 16);
    test_comparison_perf(1000000, 8);
    test_comparison_perf(1000000, 16);
    test_comparison_perf(1000000, 32);
    test_comparison_perf(1000000, 64);
    test_comparison_perf(1000000, 128);
}
