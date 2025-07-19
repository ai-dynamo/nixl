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
#include <file_utils.h>

namespace py = pybind11;

//JUST FOR TESTING
uintptr_t malloc_passthru(int size) {
    return (uintptr_t) malloc(size);
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

// File utils wrapper functions
int open_file(const std::string& path, nixlFileMode mode) {
    int fd;
    nixl_status_t status = nixlFileUtils::openFile(path, mode, fd);
    if (status != NIXL_SUCCESS) {
        throw py::value_error("Failed to open file: " + path);
    }
    return fd;
}

void close_file(int fd) {
    nixl_status_t status = nixlFileUtils::closeFile(fd);
    if (status != NIXL_SUCCESS) {
        throw py::value_error("Failed to close file descriptor: " + std::to_string(fd));
    }
}

void unlink_file(const std::string& path) {
    nixl_status_t status = nixlFileUtils::unlinkFile(path);
    if (status != NIXL_SUCCESS) {
        throw py::value_error("Failed to unlink file: " + path);
    }
}

bool file_exists(const std::string& path) {
    nixl_status_t status = nixlFileUtils::fileExists(path);
    return status == NIXL_SUCCESS;
}

PYBIND11_MODULE(_utils, m) {
    m.def("malloc_passthru", &malloc_passthru);
    m.def("free_passthru", &free_passthru);
    m.def("ba_buf", &ba_buf);
    m.def("verify_transfer", &verify_transfer);

    py::enum_<nixlFileMode>(m, "FileMode")
        .value("CREATE", nixlFileMode::CREATE)
        .value("READ_ONLY", nixlFileMode::READ_ONLY)
        .value("WRITE_ONLY", nixlFileMode::WRITE_ONLY)
        .value("READ_WRITE", nixlFileMode::READ_WRITE)
        .export_values();

    py::enum_<nixlFileOperation>(m, "FileOperation")
        .value("OPEN", nixlFileOperation::OPEN)
        .value("CLOSE", nixlFileOperation::CLOSE)
        .value("UNLINK", nixlFileOperation::UNLINK)
        .value("STAT", nixlFileOperation::STAT)
        .export_values();

    m.def("open_file", &open_file, "Open a file with specified mode");
    m.def("close_file", &close_file, "Close a file descriptor");
    m.def("unlink_file", &unlink_file, "Delete a file");
    m.def("file_exists", &file_exists, "Check if a file exists");
}
