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
#include <pybind11/pybind11.h>

namespace py = pybind11;

// NB: for storage IO with O_DIRECT, allocated memory must match device block size
#define IO_SIZE 4096

//JUST FOR TESTING
uintptr_t malloc_passthru(int size) {
    return (uintptr_t)aligned_alloc(IO_SIZE, size);
}

//JUST FOR TESTING
void free_passthru(uintptr_t buf) {
    free((void*) buf);
}

//JUST FOR TESTING
void ba_buf(uintptr_t addr, int size) {
    uint8_t* buf = (uint8_t*) addr;
    for(int i = 0; i<size; i++) buf[i] = 0xba;
}

//JUST FOR TESTING
void verify_transfer(uintptr_t addr1, uintptr_t addr2, int size) {
    for(int i = 0; i<size; i++) assert(((uint8_t*) addr1)[i] == ((uint8_t*) addr2)[i]);
}

PYBIND11_MODULE(_utils, m) {
    m.def("malloc_passthru", &malloc_passthru);
    m.def("free_passthru", &free_passthru);
    m.def("ba_buf", &ba_buf);
    m.def("verify_transfer", &verify_transfer);
}
