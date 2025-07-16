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
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "etcd_rt.h"

namespace py = pybind11;

PYBIND11_MODULE (etcd_runtime, m) {
    m.doc() = "Python bindings for ETCD runtime";

    py::class_<xferBenchEtcdRT>(m, "EtcdRuntime")
        .def(py::init<const std::string &, const std::string &, const int, int *>(),
             py::arg("benchmark_group"),
             py::arg("etcd_endpoints"),
             py::arg("size"),
             py::arg("terminate") = nullptr)
        .def("setup", &xferBenchEtcdRT::setup)
        .def("get_rank", &xferBenchEtcdRT::getRank)
        .def("get_size", &xferBenchEtcdRT::getSize)
        .def("send_int",
             [](xferBenchEtcdRT &self, int value, int dest_rank) {
                 return self.sendInt(&value, dest_rank);
             })
        .def("recv_int",
             [](xferBenchEtcdRT &self, int src_rank) {
                 int value;
                 int result = self.recvInt(&value, src_rank);
                 return py::make_tuple(result, value);
             })
        .def("send_char",
             [](xferBenchEtcdRT &self, const std::string &data, int dest_rank) {
                 return self.sendChar(const_cast<char *>(data.c_str()), data.size(), dest_rank);
             })
        .def("recv_char",
             [](xferBenchEtcdRT &self, size_t count, int src_rank) {
                 std::vector<char> buffer(count);
                 int result = self.recvChar(buffer.data(), count, src_rank);
                 std::string data(buffer.begin(), buffer.end());
                 return py::make_tuple(result, data);
             })
        .def("broadcast_int",
             [](xferBenchEtcdRT &self, py::array_t<int> buffer, int root_rank) {
                 py::buffer_info buf_info = buffer.request();
                 int *ptr = static_cast<int *>(buf_info.ptr);
                 size_t count = buf_info.size;
                 return self.broadcastInt(ptr, count, root_rank);
             })
        .def("reduce_sum_double",
             [](xferBenchEtcdRT &self, double local_value, int dest_rank) {
                 double global_value;
                 int result = self.reduceSumDouble(&local_value, &global_value, dest_rank);
                 return py::make_tuple(result, global_value);
             })
        .def("barrier", &xferBenchEtcdRT::barrier);
}
