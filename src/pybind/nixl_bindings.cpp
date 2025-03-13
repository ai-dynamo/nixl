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
#include <pybind11/operators.h>

#include <tuple>
#include <iostream>

#include "nixl.h"
#include "utils/serdes/serdes.h"

namespace py = pybind11;

PYBIND11_MODULE(_bindings, m) {

    //TODO: each nixl class and/or function can be documented in place
    m.doc() = "pybind11 NIXL plugin: Implements NIXL descriptors and lists, soon Agent API as well";

    //cast types
    py::enum_<nixl_mem_t>(m, "nixl_mem_t")
        .value("DRAM_SEG", DRAM_SEG)
        .value("VRAM_SEG", VRAM_SEG)
        .value("BLK_SEG", BLK_SEG)
        .value("FILE_SEG", FILE_SEG)
        .export_values();

    py::enum_<nixl_xfer_op_t>(m, "nixl_xfer_op_t")
        .value("NIXL_READ", NIXL_READ)
        .value("NIXL_RD_NOTIF", NIXL_RD_NOTIF)
        .value("NIXL_WRITE", NIXL_WRITE)
        .value("NIXL_WR_NOTIF", NIXL_WR_NOTIF)
        .export_values();

    py::enum_<nixl_status_t>(m, "nixl_status_t")
        .value("NIXL_IN_PROG", NIXL_IN_PROG)
        .value("NIXL_SUCCESS", NIXL_SUCCESS)
        .value("NIXL_ERR_NOT_POSTED", NIXL_ERR_NOT_POSTED)
        .value("NIXL_ERR_INVALID_PARAM", NIXL_ERR_INVALID_PARAM)
        .value("NIXL_ERR_BACKEND", NIXL_ERR_BACKEND)
        .value("NIXL_ERR_NOT_FOUND", NIXL_ERR_NOT_FOUND)
        .value("NIXL_ERR_MISMATCH", NIXL_ERR_MISMATCH)
        .value("NIXL_ERR_NOT_ALLOWED", NIXL_ERR_NOT_ALLOWED)
        .value("NIXL_ERR_REPOST_ACTIVE", NIXL_ERR_REPOST_ACTIVE)
        .value("NIXL_ERR_UNKNOWN", NIXL_ERR_UNKNOWN)
        .export_values();

    py::class_<nixl_xfer_dlist_t>(m, "nixlXferDList")
        .def(py::init<nixl_mem_t, bool, bool, int>(), py::arg("type"), py::arg("unifiedAddr")=true, py::arg("sorted")=false, py::arg("init_size")=0)
        .def(py::init([](nixl_mem_t mem, std::vector<py::tuple> descs, bool unifiedAddr, bool sorted) {
                nixl_xfer_dlist_t new_list(mem, unifiedAddr, sorted, descs.size());
                for(long unsigned int i = 0; i<descs.size(); i++)
                    new_list[i] = nixlBasicDesc(descs[i][0].cast<uintptr_t>(), descs[i][1].cast<size_t>(), descs[i][2].cast<uint32_t>());
                if (sorted) new_list.verifySorted();
                return new_list;
            }), py::arg("type"), py::arg("descs"), py::arg("unifiedAddr")=true, py::arg("sorted")=false)
        .def("getType", &nixl_xfer_dlist_t::getType)
        .def("isUnifiedAddr", &nixl_xfer_dlist_t::isUnifiedAddr)
        .def("descCount", &nixl_xfer_dlist_t::descCount)
        .def("isEmpty", &nixl_xfer_dlist_t::isEmpty)
        .def("isSorted", &nixl_xfer_dlist_t::isSorted)
        .def(py::self == py::self)
        .def("__getitem__", [](nixl_xfer_dlist_t &list, unsigned int i) ->
              std::tuple<uintptr_t, size_t, uint32_t> {
                    std::tuple<uintptr_t, size_t, uint32_t> ret;
                    nixlBasicDesc desc = list[i];
                    std::get<0>(ret) = desc.addr;
                    std::get<1>(ret) = desc.len;
                    std::get<2>(ret) = desc.devId;
                    return ret;
              })
        .def("__setitem__", [](nixl_xfer_dlist_t &list, unsigned int i, const py::tuple &desc) {
                list[i] = nixlBasicDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint32_t>());
            })
        .def("addDesc", [](nixl_xfer_dlist_t &list, const py::tuple &desc) {
                nixlBasicDesc newDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint32_t>());
                newDesc.print("bdesc");
                list.addDesc(nixlBasicDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint32_t>()));
            })
        .def("remDesc", &nixl_xfer_dlist_t::remDesc)
        .def("verifySorted", &nixl_xfer_dlist_t::verifySorted)
        .def("clear", &nixl_xfer_dlist_t::clear)
        .def("print", &nixl_xfer_dlist_t::print)
        .def(py::pickle(
            [](const nixl_xfer_dlist_t& self) { // __getstate__
                nixlSerDes serdes;
                self.serialize(&serdes);
                return py::bytes(serdes.exportStr());
            },
            [](py::bytes serdes_str) { // __setstate__
                nixlSerDes serdes;
                serdes.importStr(std::string(serdes_str));
                nixl_xfer_dlist_t newObj =
                    nixl_xfer_dlist_t(&serdes);
                return newObj;
            }
        ));

    py::class_<nixl_reg_dlist_t>(m, "nixlRegDList")
        .def(py::init<nixl_mem_t, bool, bool, int>(), py::arg("type"), py::arg("unifiedAddr")=true, py::arg("sorted")=false, py::arg("init_size")=0)
        .def(py::init([](nixl_mem_t mem, std::vector<py::tuple> descs, bool unifiedAddr, bool sorted) {
                nixl_reg_dlist_t new_list(mem, unifiedAddr, sorted, descs.size());
                for(long unsigned int i = 0; i<descs.size(); i++)
                    new_list[i] = nixlStringDesc(descs[i][0].cast<uintptr_t>(), descs[i][1].cast<size_t>(), descs[i][2].cast<uint32_t>(), descs[i][3].cast<std::string>());
                if (sorted) new_list.verifySorted();
                return new_list;
            }), py::arg("type"), py::arg("descs"), py::arg("unifiedAddr")=true, py::arg("sorted")=false)
        .def("getType", &nixl_reg_dlist_t::getType)
        .def("isUnifiedAddr", &nixl_reg_dlist_t::isUnifiedAddr)
        .def("descCount", &nixl_reg_dlist_t::descCount)
        .def("isEmpty", &nixl_reg_dlist_t::isEmpty)
        .def("isSorted", &nixl_reg_dlist_t::isSorted)
        .def(py::self == py::self)
        .def("__getitem__", [](nixl_reg_dlist_t &list, unsigned int i) ->
              std::tuple<uintptr_t, size_t, uint32_t, std::string> {
                    std::tuple<uintptr_t, size_t, uint32_t, std::string> ret;
                    nixlStringDesc desc = list[i];
                    std::get<0>(ret) = desc.addr;
                    std::get<1>(ret) = desc.len;
                    std::get<2>(ret) = desc.devId;
                    std::get<3>(ret) = desc.metaInfo;
                    return ret;
              })
        .def("__setitem__", [](nixl_reg_dlist_t &list, unsigned int i, const py::tuple &desc) {
                list[i] = nixlStringDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint32_t>(), desc[3].cast<std::string>());
            })
        .def("addDesc", [](nixl_reg_dlist_t &list, const py::tuple &desc) {
                uintptr_t addr = desc[0].cast<uintptr_t>();
                size_t size = desc[1].cast<size_t>();
                uint32_t dev_id = desc[2].cast<uint32_t>();
                std::string meta = desc[3].cast<std::string>();
                nixlStringDesc newDesc(addr, size, dev_id, meta);

                list.addDesc(newDesc);
            })
        .def("remDesc", &nixl_reg_dlist_t::remDesc)
        .def("verifySorted", &nixl_reg_dlist_t::verifySorted)
        .def("clear", &nixl_reg_dlist_t::clear)
        .def("print", &nixl_reg_dlist_t::print)
        .def(py::pickle(
            [](const nixl_reg_dlist_t& self) { // __getstate__
                nixlSerDes serdes;
                self.serialize(&serdes);
                return py::bytes(serdes.exportStr());
            },
            [](py::bytes serdes_str) { // __setstate__
                nixlSerDes serdes;
                serdes.importStr(std::string(serdes_str));
                nixl_reg_dlist_t newObj =
                    nixl_reg_dlist_t(&serdes);
                return newObj;
            }
        ));

    py::class_<nixlAgentConfig>(m, "nixlAgentConfig")
        //implicit constructor
        .def(py::init<bool>());

    //note: pybind will automatically convert notif_map to python types:
    //so, a Dictionary of string: List<string>

    py::class_<nixlAgent>(m, "nixlAgent")
        .def(py::init<std::string, nixlAgentConfig>())
        .def("createBackend", [](nixlAgent &agent, nixl_backend_t type, nixl_b_params_t initParams) -> uintptr_t {
                    nixlBackendH* backend;
                    nixl_status_t ret = agent.createBackend(type, initParams, backend);
                    if(ret < 0) return (uintptr_t) nullptr;
                    return (uintptr_t) backend;
            })
        .def("registerMem", [](nixlAgent &agent, nixl_reg_dlist_t descs, uintptr_t backend) -> nixl_status_t {
                    return agent.registerMem(descs, (nixlBackendH*) backend);
                })
        .def("deregisterMem", [](nixlAgent &agent, nixl_reg_dlist_t descs, uintptr_t backend) -> nixl_status_t {
                    return agent.deregisterMem(descs, (nixlBackendH*) backend);
                })
        .def("makeConnection", &nixlAgent::makeConnection)
        //note: slight API change, python cannot receive values by passing refs, so handle must be returned
        .def("createXferReq", [](nixlAgent &agent,
                                 const nixl_xfer_dlist_t &local_descs,
                                 const nixl_xfer_dlist_t &remote_descs,
                                 const std::string &remote_agent,
                                 const std::string &notif_msg,
                                 const nixl_xfer_op_t &operation,
                                 uintptr_t backend) -> uintptr_t {
                    nixlXferReqH* handle;
                    nixl_status_t ret = agent.createXferReq(local_descs, remote_descs, remote_agent, notif_msg, operation, handle, (nixlBackendH*) backend);
                    if (ret != NIXL_SUCCESS) return (uintptr_t) nullptr;
                    else return (uintptr_t) handle;
                }, py::arg("local_descs"),
                   py::arg("remote_descs"), py::arg("remote_agent"),
                   py::arg("notif_msg"), py::arg("operation"),
                   py::arg("backend") = ((uintptr_t) nullptr))
        .def("getXferBackend", [](nixlAgent &agent, uintptr_t reqh) -> uintptr_t {
                    nixlBackendH* handle;
                    nixl_status_t ret = agent.getXferBackend((nixlXferReqH*) reqh, handle);
                    if(ret < 0) return (uintptr_t) nullptr;
                    return (uintptr_t) handle;
            })
        .def("prepXferSide", [](nixlAgent &agent,
                                const nixl_xfer_dlist_t &descs,
                                const std::string &remote_agent,
                                uintptr_t backend) -> uintptr_t {
                    nixlXferSideH* handle;
                    nixl_status_t ret = agent.prepXferSide(descs, remote_agent, (nixlBackendH*) backend, handle);
                    if (ret != NIXL_SUCCESS) return (uintptr_t) nullptr;
                    else return (uintptr_t) handle;
                })
        .def("makeXferReq", [](nixlAgent &agent,
                               uintptr_t local_side,
                               const std::vector<int> &local_indices,
                               uintptr_t remote_side,
                               const std::vector<int> &remote_indices,
                               const std::string &notif_msg,
                               const nixl_xfer_op_t &operation) -> uintptr_t {
                    nixlXferReqH* handle;
                    nixl_status_t ret = agent.makeXferReq((nixlXferSideH*) local_side, local_indices,
                                                          (nixlXferSideH*) remote_side, remote_indices,
                                                          notif_msg, operation, handle);
                    if (ret != NIXL_SUCCESS) return (uintptr_t) nullptr;
                    else return (uintptr_t) handle;
                })
        .def("invalidateXferReq", [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                    return agent.invalidateXferReq((nixlXferReqH*) reqh);
                })
        .def("invalidateXferSide", [](nixlAgent &agent, uintptr_t handle) -> nixl_status_t {
                    return agent.invalidateXferSide((nixlXferSideH*) handle);
                })
        .def("postXferReq", [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                    return agent.postXferReq((nixlXferReqH*) reqh);
                })
        .def("getXferStatus", [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                    return agent.getXferStatus((nixlXferReqH*) reqh);
                })
        .def("getNotifs", [](nixlAgent &agent, nixl_notifs_t notif_map) -> nixl_notifs_t {
                    int n_new;
                    nixl_status_t ret = agent.getNotifs(notif_map, n_new);

                    if (ret != NIXL_SUCCESS || n_new == 0) return notif_map;

                    nixl_notifs_t ret_map;
                    for (const auto& pair : notif_map) {
                        std::vector<std::string> agent_notifs;

                        for(const auto& str : pair.second)  {
                            agent_notifs.push_back(py::bytes(str));
                        }

                        ret_map[pair.first] = agent_notifs;
                    }
                    return ret_map;
                })
        .def("genNotif", [](nixlAgent &agent, const std::string &remote_agent,
                                              const std::string &msg,
                                              uintptr_t backend) {
                    return agent.genNotif(remote_agent, msg, (nixlBackendH*) backend);
                })
        .def("getLocalMD", [](nixlAgent &agent) -> py::bytes {
                    //python can only interpret text strings
                    std::string ret_str;
                    nixl_status_t ret = agent.getLocalMD(ret_str);
                    if(ret != NIXL_SUCCESS) return "";
                    return py::bytes(ret_str);
                })
        .def("loadRemoteMD", [](nixlAgent &agent, const std::string &remote_metadata) -> py::bytes {
                    //python can only interpret text strings
                    std::string remote_name;
                    nixl_status_t ret = agent.loadRemoteMD(remote_metadata, remote_name);
                    if(ret != NIXL_SUCCESS) return "";
                    return py::bytes(remote_name);
                })
        .def("invalidateRemoteMD", &nixlAgent::invalidateRemoteMD);
}
