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

typedef std::map<std::string, std::vector<py::bytes>> nixl_py_notifs_t;

class nixlNotPostedError : public std::runtime_error {
    public:
        nixlNotPostedError(const char* what) : runtime_error(what) {}
};

class nixlInvalidParamError : public std::runtime_error {
    public:
        nixlInvalidParamError(const char* what) : runtime_error(what) {}
};

class nixlBackendError : public std::runtime_error {
    public:
        nixlBackendError(const char* what) : runtime_error(what) {}
};


class nixlNotFoundError : public std::runtime_error {
    public:
        nixlNotFoundError(const char* what) : runtime_error(what) {}
};


class nixlMismatchError : public std::runtime_error {
    public:
        nixlMismatchError(const char* what) : runtime_error(what) {}
};


class nixlNotAllowedError : public std::runtime_error {
    public:
        nixlNotAllowedError(const char* what) : runtime_error(what) {}
};


class nixlRepostActiveError : public std::runtime_error {
    public:
        nixlRepostActiveError(const char* what) : runtime_error(what) {}
};

class nixlNotSupportedError : public std::runtime_error {
    public:
        nixlNotSupportedError(const char* what) : runtime_error(what) {}
};

class nixlUnknownError : public std::runtime_error {
    public:
        nixlUnknownError(const char* what) : runtime_error(what) {}
};

void throw_nixl_exception(const nixl_status_t &status) {
    switch (status) {
        case NIXL_IN_PROG:           return; //not an error
        case NIXL_SUCCESS:           return; //not an error
        case NIXL_ERR_NOT_POSTED:
            throw nixlNotPostedError(nixlEnumStrings::statusStr(status).c_str());
            break;
        case NIXL_ERR_INVALID_PARAM:
            throw nixlInvalidParamError(nixlEnumStrings::statusStr(status).c_str());
            break;
        case NIXL_ERR_BACKEND:
            throw nixlBackendError(nixlEnumStrings::statusStr(status).c_str());
            break;
        case NIXL_ERR_NOT_FOUND:
            throw nixlNotFoundError(nixlEnumStrings::statusStr(status).c_str());
            break;
        case NIXL_ERR_MISMATCH:
            throw nixlMismatchError(nixlEnumStrings::statusStr(status).c_str());
            break;
        case NIXL_ERR_NOT_ALLOWED:
            throw nixlNotAllowedError(nixlEnumStrings::statusStr(status).c_str());
            break;
        case NIXL_ERR_REPOST_ACTIVE:
            throw nixlRepostActiveError(nixlEnumStrings::statusStr(status).c_str());
            break;
        case NIXL_ERR_UNKNOWN:
            throw nixlUnknownError(nixlEnumStrings::statusStr(status).c_str());
            break;
        case NIXL_ERR_NOT_SUPPORTED:
            throw nixlNotSupportedError(nixlEnumStrings::statusStr(status).c_str());
            break;
        default:
            throw std::runtime_error("BAD_STATUS");
    }
}

PYBIND11_MODULE(_bindings, m) {

    //TODO: each nixl class and/or function can be documented in place
    m.doc() = "pybind11 NIXL plugin: Implements NIXL descriptors and lists, as well as bindings of NIXL CPP APIs";

    m.attr("NIXL_INIT_AGENT") = NIXL_INIT_AGENT;

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
        .value("NIXL_ERR_NOT_SUPPORTED", NIXL_ERR_NOT_SUPPORTED)
        .export_values();

    py::register_exception<nixlNotPostedError>(m, "nixlNotPostedError");
    py::register_exception<nixlInvalidParamError>(m, "nixlInvalidParamError");
    py::register_exception<nixlBackendError>(m, "nixlBackendError");
    py::register_exception<nixlNotFoundError>(m, "nixlNotFoundError");
    py::register_exception<nixlMismatchError>(m, "nixlMismatchError");
    py::register_exception<nixlNotAllowedError>(m, "nixlNotAllowedError");
    py::register_exception<nixlRepostActiveError>(m, "nixlRepostActiveError");
    py::register_exception<nixlUnknownError>(m, "nixlUnknownError");
    py::register_exception<nixlNotSupportedError>(m, "nixlNotSupportedError");

    py::class_<nixl_xfer_dlist_t>(m, "nixlXferDList")
        .def(py::init<nixl_mem_t, bool, int>(), py::arg("type"), py::arg("sorted")=false, py::arg("init_size")=0)
        .def(py::init([](nixl_mem_t mem, std::vector<py::tuple> descs, bool sorted) {
                nixl_xfer_dlist_t new_list(mem, sorted, descs.size());
                for(long unsigned int i = 0; i<descs.size(); i++)
                    new_list[i] = nixlBasicDesc(descs[i][0].cast<uintptr_t>(), descs[i][1].cast<size_t>(), descs[i][2].cast<uint32_t>());
                if (sorted) new_list.verifySorted();
                return new_list;
            }), py::arg("type"), py::arg("descs"), py::arg("sorted")=false)
        .def("getType", &nixl_xfer_dlist_t::getType)
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
        .def("index", [](nixl_xfer_dlist_t &list, const py::tuple &desc) {
                int ret = (nixl_status_t) list.getIndex(nixlBasicDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(),
                                                  desc[2].cast<uint32_t>()));
                if(ret < 0) throw_nixl_exception((nixl_status_t) ret);
                return (int) ret;
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
        .def(py::init<nixl_mem_t, bool, int>(), py::arg("type"), py::arg("sorted")=false, py::arg("init_size")=0)
        .def(py::init([](nixl_mem_t mem, std::vector<py::tuple> descs, bool sorted) {
                nixl_reg_dlist_t new_list(mem, sorted, descs.size());
                for(long unsigned int i = 0; i<descs.size(); i++)
                    new_list[i] = nixlBlobDesc(descs[i][0].cast<uintptr_t>(), descs[i][1].cast<size_t>(), descs[i][2].cast<uint32_t>(), descs[i][3].cast<std::string>());
                if (sorted) new_list.verifySorted();
                return new_list;
            }), py::arg("type"), py::arg("descs"), py::arg("sorted")=false)
        .def("getType", &nixl_reg_dlist_t::getType)
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
        .def("index", [](nixl_reg_dlist_t &list, const py::tuple &desc) {
                int ret = list.getIndex(nixlBlobDesc(desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(),
                                                  desc[2].cast<uint32_t>(),desc[3].cast<std::string>()));
                if(ret < 0) throw_nixl_exception((nixl_status_t) ret);
                return ret;
            })
        .def("trim", &nixl_reg_dlist_t::trim)
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
        .def("getAvailPlugins", [](nixlAgent &agent) -> std::vector<nixl_backend_t> {
                    std::vector<nixl_backend_t> backends;
                    throw_nixl_exception(agent.getAvailPlugins(backends));
                    return backends;
            })
        .def("getPluginParams", [](nixlAgent &agent, const nixl_backend_t type) -> std::pair<nixl_b_params_t, std::vector<std::string>> {
                    nixl_b_params_t params;
                    nixl_mem_list_t mems;
                    std::vector<std::string> mems_vec;
                    throw_nixl_exception(agent.getPluginParams(type, mems, params));
                    for (const auto& elm: mems)
                        mems_vec.push_back(nixlEnumStrings::memTypeStr(elm));
                    return std::make_pair(params, mems_vec);
            })
        .def("getBackendParams", [](nixlAgent &agent, uintptr_t backend) -> std::pair<nixl_b_params_t, std::vector<std::string>> {
                    nixl_b_params_t params;
                    nixl_mem_list_t mems;
                    std::vector<std::string> mems_vec;
                    throw_nixl_exception(agent.getBackendParams((nixlBackendH*) backend, mems, params));
                    for (const auto& elm: mems)
                        mems_vec.push_back(nixlEnumStrings::memTypeStr(elm));
                    return std::make_pair(params, mems_vec);
            })
        .def("createBackend", [](nixlAgent &agent, const nixl_backend_t &type, const nixl_b_params_t &initParams) -> uintptr_t {
                    nixlBackendH* backend = nullptr;
                    throw_nixl_exception(agent.createBackend(type, initParams, backend));
                    return (uintptr_t) backend;
            })
        .def("registerMem", [](nixlAgent &agent, nixl_reg_dlist_t descs, std::vector<uintptr_t> backends) -> nixl_status_t {
                    nixl_opt_args_t extra_params;
                    nixl_status_t ret;
                    for(uintptr_t backend: backends)
                        extra_params.backends.push_back((nixlBackendH*) backend);

                    ret = agent.registerMem(descs, &extra_params);
                    throw_nixl_exception(ret);
                    return ret;
                }, py::arg("descs"), py::arg("backends") = std::vector<uintptr_t>({}))
        .def("deregisterMem", [](nixlAgent &agent, nixl_reg_dlist_t descs, std::vector<uintptr_t> backends) -> nixl_status_t {
                    nixl_opt_args_t extra_params;
                    nixl_status_t ret;
                    for(uintptr_t backend: backends)
                        extra_params.backends.push_back((nixlBackendH*) backend);

                    ret = agent.deregisterMem(descs, &extra_params);
                    throw_nixl_exception(ret);
                    return ret;
                }, py::arg("descs"), py::arg("backends") = std::vector<uintptr_t>({}))
        .def("makeConnection", [](nixlAgent &agent, const std::string &remote_agent) {
                    nixl_status_t ret = agent.makeConnection(remote_agent);
                    throw_nixl_exception(ret);
                    return ret;
                })
        .def("createXferReq", [](nixlAgent &agent,
                                 const nixl_xfer_op_t &operation,
                                 const nixl_xfer_dlist_t &local_descs,
                                 const nixl_xfer_dlist_t &remote_descs,
                                 const std::string &remote_agent,
                                 const std::string &notif_msg,
                                 std::vector<uintptr_t> backends) -> uintptr_t {
                    nixlXferReqH* handle = nullptr;
                    nixl_opt_args_t extra_params;

                    for(uintptr_t backend: backends)
                        extra_params.backends.push_back((nixlBackendH*) backend);

                    if (notif_msg.size()>0) {
                        extra_params.notifMsg = notif_msg;
                        extra_params.hasNotif = true;
                    }
                    nixl_status_t ret = agent.createXferReq(operation, local_descs, remote_descs, remote_agent, handle, &extra_params);

                    throw_nixl_exception(ret);
                    return (uintptr_t) handle;
                }, py::arg("operation"), py::arg("local_descs"),
                   py::arg("remote_descs"), py::arg("remote_agent"),
                   py::arg("notif_msg") = std::string(""),
                   py::arg("backend") = std::vector<uintptr_t>({}))
        .def("queryXferBackend", [](nixlAgent &agent, uintptr_t reqh) -> uintptr_t {
                    nixlBackendH* handle = nullptr;
                    throw_nixl_exception(agent.queryXferBackend((nixlXferReqH*) reqh, handle));
                    return (uintptr_t) handle;
            })
        .def("prepXferDlist", [](nixlAgent &agent,
                                 std::string &agent_name,
                                 const nixl_xfer_dlist_t &descs,
                                 std::vector<uintptr_t> backends) -> uintptr_t {
                    nixlDlistH* handle = nullptr;
                    nixl_opt_args_t extra_params;

                    for(uintptr_t backend: backends)
                        extra_params.backends.push_back((nixlBackendH*) backend);

                    throw_nixl_exception(agent.prepXferDlist(agent_name, descs, handle, &extra_params));

                    return (uintptr_t) handle;
                }, py::arg("agent_name"), py::arg("descs"), py::arg("backend") = std::vector<uintptr_t>({}))
        .def("makeXferReq", [](nixlAgent &agent,
                               const nixl_xfer_op_t &operation,
                               uintptr_t local_side,
                               const std::vector<int> &local_indices,
                               uintptr_t remote_side,
                               const std::vector<int> &remote_indices,
                               const std::string &notif_msg,
                               bool skip_desc_merge) -> uintptr_t {
                    nixlXferReqH* handle = nullptr;
                    nixl_opt_args_t extra_params;
                    if (notif_msg.size()>0) {
                        extra_params.notifMsg = notif_msg;
                        extra_params.hasNotif = true;
                    }
                    extra_params.skipDescMerge = skip_desc_merge;
                    throw_nixl_exception(agent.makeXferReq(operation,
                                                           (nixlDlistH*) local_side, local_indices,
                                                           (nixlDlistH*) remote_side, remote_indices,
                                                           handle, &extra_params));

                    return (uintptr_t) handle;
                }, py::arg("operation"), py::arg("local_side"),
                   py::arg("local_indices"), py::arg("remote_side"),
                   py::arg("remote_indices"), py::arg("notif_msg") = std::string(""),
                   py::arg("skip_desc_merg") = false)
        .def("postXferReq", [](nixlAgent &agent, uintptr_t reqh, std::string notif_msg) -> nixl_status_t {
                    nixl_opt_args_t extra_params;
                    nixl_status_t ret;
                    if (notif_msg.size()>0) {
                        extra_params.notifMsg = notif_msg;
                        extra_params.hasNotif = true;
                        ret = agent.postXferReq((nixlXferReqH*) reqh, &extra_params);
                    } else {
                        ret = agent.postXferReq((nixlXferReqH*) reqh);
                    }
                    throw_nixl_exception(ret);
                    return ret;
                }, py::arg("reqh"), py::arg("notif_msg") = std::string(""))
        .def("getXferStatus", [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                    nixl_status_t ret = agent.getXferStatus((nixlXferReqH*) reqh);
                    throw_nixl_exception(ret);
                    return ret;
                })
        .def("queryXferBackend", [](nixlAgent &agent, uintptr_t reqh) -> uintptr_t {
                    nixlBackendH* backend = nullptr;
                    throw_nixl_exception(agent.queryXferBackend((nixlXferReqH*) reqh, backend));
                    return (uintptr_t) backend;
                })
        .def("releaseXferReq", [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                    nixl_status_t ret = agent.releaseXferReq((nixlXferReqH*) reqh);
                    throw_nixl_exception(ret);
                    return ret;
                })
        .def("releasedDlistH", [](nixlAgent &agent, uintptr_t handle) -> nixl_status_t {
                    nixl_status_t ret = agent.releasedDlistH((nixlDlistH*) handle);
                    throw_nixl_exception(ret);
                    return ret;
                })
        .def("getNotifs", [](nixlAgent &agent, nixl_py_notifs_t &notif_map) -> nixl_py_notifs_t {
                    nixl_notifs_t new_notifs;
                    nixl_status_t ret = agent.getNotifs(new_notifs);

                    throw_nixl_exception(ret);

                    for (const auto& pair : new_notifs) {
                        for(const auto& str : pair.second)
                            notif_map[pair.first].push_back(py::bytes(str));
                    }
                    return notif_map;
                })
        .def("genNotif", [](nixlAgent &agent, const std::string &remote_agent,
                                              const std::string &msg,
                                              uintptr_t backend) {
                    nixl_opt_args_t extra_params;
                    nixl_status_t ret;
                    extra_params.backends.push_back((nixlBackendH*) backend);
                    ret = agent.genNotif(remote_agent, msg, &extra_params);

                    throw_nixl_exception(ret);
                    return ret;
                })
        .def("getLocalMD", [](nixlAgent &agent) -> py::bytes {
                    //python can only interpret text strings
                    std::string ret_str("");
                    throw_nixl_exception(agent.getLocalMD(ret_str));
                    return py::bytes(ret_str);
                })
        .def("loadRemoteMD", [](nixlAgent &agent, const std::string &remote_metadata) -> py::bytes {
                    //python can only interpret text strings
                    std::string remote_name("");
                    throw_nixl_exception(agent.loadRemoteMD(remote_metadata, remote_name));
                    return py::bytes(remote_name);
                })
        .def("invalidateRemoteMD", &nixlAgent::invalidateRemoteMD);
}
