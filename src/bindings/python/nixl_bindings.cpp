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
#include "serdes/serdes.h"

namespace py = pybind11;

PYBIND11_MODULE(_bindings, m) {

    //TODO: each nixl class and/or function can be documented in place
    m.doc() = "pybind11 NIXL plugin: Implements NIXL descriptors and lists, as well as bindings of NIXL CPP APIs";

    //cast types
    py::enum_<nixl_mem_t>(m, "nixl_mem_t")
        .value("DRAM_SEG", DRAM_SEG)
        .value("VRAM_SEG", VRAM_SEG)
        .value("BLK_SEG", BLK_SEG)
        .value("OBJ_SEG", BLK_SEG)
        .value("FILE_SEG", FILE_SEG)
        .export_values();

    py::enum_<nixl_xfer_op_t>(m, "nixl_xfer_op_t")
        .value("NIXL_READ", NIXL_READ)
        .value("NIXL_WRITE", NIXL_WRITE)
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
                list.addDesc(nixlBasicDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint32_t>()));
            })
        .def("append", [](nixl_xfer_dlist_t &list, const py::tuple &desc) {
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
                    new_list[i] = nixlBlobDesc(descs[i][0].cast<uintptr_t>(), descs[i][1].cast<size_t>(), descs[i][2].cast<uint32_t>(), descs[i][3].cast<std::string>());
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
                    nixlBlobDesc desc = list[i];
                    std::get<0>(ret) = desc.addr;
                    std::get<1>(ret) = desc.len;
                    std::get<2>(ret) = desc.devId;
                    std::get<3>(ret) = desc.metaInfo;
                    return ret;
              })
        .def("__setitem__", [](nixl_reg_dlist_t &list, unsigned int i, const py::tuple &desc) {
                list[i] = nixlBlobDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint32_t>(), desc[3].cast<std::string>());
            })
        .def("addDesc", [](nixl_reg_dlist_t &list, const py::tuple &desc) {
                list.addDesc(nixlBlobDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(),
                                            desc[2].cast<uint32_t>(),desc[3].cast<std::string>()));
            })
        .def("append", [](nixl_reg_dlist_t &list, const py::tuple &desc) {
                list.addDesc(nixlBlobDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(),
                                            desc[2].cast<uint32_t>(),desc[3].cast<std::string>()));
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
        .def("getAvailPlugins", &nixlAgent::getAvailPlugins)
        .def("getPluginParams", [](nixlAgent &agent, const nixl_backend_t type) -> nixl_b_params_t {
                    nixl_b_params_t params;
                    nixl_mem_list_t mems;
                    nixl_status_t ret = agent.getPluginParams(type, mems, params);
                    if(ret < 0); //throw exception
                    // TODO merge the mems
                    return params;
            })
        .def("getBackendParams", [](nixlAgent &agent, uintptr_t backend) -> nixl_b_params_t {
                    nixl_b_params_t params;
                    nixl_mem_list_t mems;
                    nixl_status_t ret = agent.getBackendParams((nixlBackendH*) backend, mems, params);
                    if(ret < 0); //throw exception
                    // TODO merge the mems
                    return params;
            })
        .def("createBackend", [](nixlAgent &agent, const nixl_backend_t &type, const nixl_b_params_t &initParams) -> uintptr_t {
                    nixlBackendH* backend;
                    nixl_status_t ret = agent.createBackend(type, initParams, backend);
                    if(ret < 0) return (uintptr_t) nullptr; //throw exception
                    return (uintptr_t) backend;
            })
        .def("registerMem", [](nixlAgent &agent, nixl_reg_dlist_t descs, uintptr_t backend) -> nixl_status_t {
                    nixl_opt_args_t extra_params;
                    extra_params.backends.push_back((nixlBackendH*) backend);
                    return agent.registerMem(descs, &extra_params);
                })
        .def("deregisterMem", [](nixlAgent &agent, nixl_reg_dlist_t descs, uintptr_t backend) -> nixl_status_t {
                    nixl_opt_args_t extra_params;
                    extra_params.backends.push_back((nixlBackendH*) backend);
                    return agent.deregisterMem(descs, &extra_params);
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
                    nixl_opt_args_t extra_params;
                    if (backend!=0)
                        extra_params.backends.push_back((nixlBackendH*) backend);
                    if (notif_msg.size()>0) {
                        extra_params.notifMsg = notif_msg;
                        extra_params.hasNotif = true;
                    }
                    nixl_status_t ret = agent.createXferReq(operation, local_descs, remote_descs, remote_agent, handle, &extra_params);
                    if (ret != NIXL_SUCCESS) return (uintptr_t) nullptr;
                    else return (uintptr_t) handle;
                }, py::arg("local_descs"),
                   py::arg("remote_descs"), py::arg("remote_agent"),
                   py::arg("notif_msg"), py::arg("operation"),
                   py::arg("backend") = ((uintptr_t) nullptr))
        .def("queryXferBackend", [](nixlAgent &agent, uintptr_t reqh) -> uintptr_t {
                    nixlBackendH* handle;
                    nixl_status_t ret = agent.queryXferBackend((nixlXferReqH*) reqh, handle);
                    if(ret < 0) return (uintptr_t) nullptr;
                    return (uintptr_t) handle;
            })
        .def("prepXferDlist", [](nixlAgent &agent,
                                const nixl_xfer_dlist_t &descs,
                                const std::string &remote_agent,
                                uintptr_t backend) -> uintptr_t {
                    nixlDlistH* handle;
                    nixl_opt_args_t extra_params;
                    extra_params.backends.push_back((nixlBackendH*) backend);
                    nixl_status_t ret = agent.prepXferDlist(descs, remote_agent, handle, &extra_params);
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
                    nixl_opt_args_t extra_params;
                    if (notif_msg.size()>0) {
                        extra_params.notifMsg = notif_msg;
                        extra_params.hasNotif = true;
                    }
                    nixl_status_t ret = agent.makeXferReq(operation,
                                                          (nixlDlistH*) local_side, local_indices,
                                                          (nixlDlistH*) remote_side, remote_indices,
                                                          handle, &extra_params);
                    if (ret != NIXL_SUCCESS) return (uintptr_t) nullptr;
                    else return (uintptr_t) handle;
                })
        .def("releaseXferReq", [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                    return agent.releaseXferReq((nixlXferReqH*) reqh);
                })
        .def("releasedDlistH", [](nixlAgent &agent, uintptr_t handle) -> nixl_status_t {
                    return agent.releasedDlistH((nixlDlistH*) handle);
                })
        .def("postXferReq", [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                    return agent.postXferReq((nixlXferReqH*) reqh);
                })
        .def("getXferStatus", [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                    return agent.getXferStatus((nixlXferReqH*) reqh);
                })
        .def("getNotifs", [](nixlAgent &agent, nixl_notifs_t notif_map) -> nixl_notifs_t {
                    nixl_status_t ret = agent.getNotifs(notif_map);

                    if (ret != NIXL_SUCCESS || notif_map.size() == 0) return notif_map;

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
                    nixl_opt_args_t extra_params;
                    extra_params.backends.push_back((nixlBackendH*) backend);
                    return agent.genNotif(remote_agent, msg, &extra_params);
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
