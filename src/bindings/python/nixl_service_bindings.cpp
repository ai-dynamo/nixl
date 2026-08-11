/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/**
 * @file nixl_service_bindings.cpp
 * @brief Standalone pybind11 bindings for nixlServiceAgent.
 *
 * This module is layered on top of the core `_bindings` extension (which
 * owns the bindings for nixlAgent, nixlAgentConfig, descriptor lists, enums,
 * and NIXL exception types).  Loading `_service_bindings` requires
 * `_bindings` to have been imported first so that nixlAgent/nixlAgentConfig
 * are already registered; pybind11 then stitches the inheritance edges on
 * nixlServiceAgent -> nixlAgent and nixlServiceAgentConfig -> nixlAgentConfig
 * without duplicating the base class.
 *
 * Exception translation is local to this module: because the `nixl*Error`
 * classes in nixl_bindings.cpp have internal linkage (anonymous-namespace
 * style), this translation unit maintains its own equivalent hierarchy to
 * avoid an ODR hazard.  Python code that catches `RuntimeError` or the
 * subclass by name continues to work; catching by the exact Python class
 * object requires importing from this module.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nixl.h"
#include "nixl_service.h"
#include "nixl_service_types.h"

namespace py = pybind11;

namespace {

using nixl_py_notifs_t = std::map<std::string, std::vector<py::bytes>>;

// ---------------------------------------------------------------------------
// Exception hierarchy (mirrors nixl_bindings.cpp; kept local to this TU)
// ---------------------------------------------------------------------------
class nixlNotPostedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlInvalidParamError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlBackendError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlNotFoundError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlMismatchError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlNotAllowedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlRepostActiveError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlUnknownError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlNotSupportedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlRemoteDisconnectError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlCancelledError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class nixlNoTelemetryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void
throw_nixl_exception(nixl_status_t status) {
    const auto msg = nixlEnumStrings::statusStr(status);
    switch (status) {
    case NIXL_IN_PROG:
    case NIXL_SUCCESS:
        return;
    case NIXL_ERR_NOT_POSTED:
        throw nixlNotPostedError(msg.c_str());
    case NIXL_ERR_INVALID_PARAM:
        throw nixlInvalidParamError(msg.c_str());
    case NIXL_ERR_BACKEND:
        throw nixlBackendError(msg.c_str());
    case NIXL_ERR_NOT_FOUND:
        throw nixlNotFoundError(msg.c_str());
    case NIXL_ERR_MISMATCH:
        throw nixlMismatchError(msg.c_str());
    case NIXL_ERR_NOT_ALLOWED:
        throw nixlNotAllowedError(msg.c_str());
    case NIXL_ERR_REPOST_ACTIVE:
        throw nixlRepostActiveError(msg.c_str());
    case NIXL_ERR_UNKNOWN:
        throw nixlUnknownError(msg.c_str());
    case NIXL_ERR_NOT_SUPPORTED:
        throw nixlNotSupportedError(msg.c_str());
    case NIXL_ERR_REMOTE_DISCONNECT:
        throw nixlRemoteDisconnectError(msg.c_str());
    case NIXL_ERR_CANCELED:
        throw nixlCancelledError(msg.c_str());
    case NIXL_ERR_NO_TELEMETRY:
        throw nixlNoTelemetryError(msg.c_str());
    default:
        throw std::runtime_error("BAD_STATUS");
    }
}

// Collect backend handles from a Python-side list of uintptr_t into a
// nixl_opt_args_t (or a derived type).  Templated so both the base args and
// nixl_service_opt_args_t work without duplication.
template<typename ArgsT>
void
set_backends(ArgsT &args, const std::vector<uintptr_t> &backends) {
    for (uintptr_t backend : backends) {
        args.backends.push_back(reinterpret_cast<nixlBackendH *>(backend));
    }
}

} // namespace

PYBIND11_MODULE(_service_bindings, m) {
    m.doc() = "pybind11 bindings for nixlServiceAgent.  Import nixl._bindings "
              "before this module so the nixlAgent / nixlAgentConfig base "
              "classes and descriptor/enum types are registered.";

    // -----------------------------------------------------------------------
    // Exception types — register with Python so they inherit from RuntimeError.
    // These are distinct Python class objects from the ones in _bindings, even
    // though they share names.  Catching RuntimeError still works.
    // -----------------------------------------------------------------------
    py::register_exception<nixlNotPostedError>(m, "nixlNotPostedError");
    py::register_exception<nixlInvalidParamError>(m, "nixlInvalidParamError");
    py::register_exception<nixlBackendError>(m, "nixlBackendError");
    py::register_exception<nixlNotFoundError>(m, "nixlNotFoundError");
    py::register_exception<nixlMismatchError>(m, "nixlMismatchError");
    py::register_exception<nixlNotAllowedError>(m, "nixlNotAllowedError");
    py::register_exception<nixlRepostActiveError>(m, "nixlRepostActiveError");
    py::register_exception<nixlUnknownError>(m, "nixlUnknownError");
    py::register_exception<nixlNotSupportedError>(m, "nixlNotSupportedError");
    py::register_exception<nixlRemoteDisconnectError>(m, "nixlRemoteDisconnectError");
    py::register_exception<nixlCancelledError>(m, "nixlCancelledError");
    py::register_exception<nixlNoTelemetryError>(m, "nixlNoTelemetryError");

    // -----------------------------------------------------------------------
    // Marshal config types (init-time configuration)
    //
    // nixl_marshal_config_t is std::variant<nixlMarshalDirectConfig,
    // nixlMarshalStagingConfig, nixlMarshalCompressConfig>.  pybind11's
    // <stl.h> variant caster converts any alternative when assigned to the
    // `mode` field on nixlServiceAgentConfig.
    // -----------------------------------------------------------------------
    py::enum_<nixl_marshal_compress_algo_t>(m, "nixl_marshal_compress_algo_t")
        .value("ANS", nixl_marshal_compress_algo_t::ANS)
        .value("ANS_DELTA", nixl_marshal_compress_algo_t::ANS_DELTA)
        .value("BITCOMP", nixl_marshal_compress_algo_t::BITCOMP)
        .export_values();

    py::class_<nixlMarshalDirectConfig>(m, "nixlMarshalDirectConfig")
        .def(py::init<>())
        .def("__repr__", [](const nixlMarshalDirectConfig &) {
            return std::string("nixlMarshalDirectConfig()");
        });

    py::class_<nixlMarshalStagingConfig>(m, "nixlMarshalStagingConfig")
        .def(py::init<>())
        .def("__repr__", [](const nixlMarshalStagingConfig &) {
            return std::string("nixlMarshalStagingConfig()");
        });

    py::class_<nixlMarshalDeltaConfig>(m, "nixlMarshalDeltaConfig")
        .def(py::init<>())
        .def("__repr__", [](const nixlMarshalDeltaConfig &) {
            return std::string("nixlMarshalDeltaConfig()");
        });

    py::class_<nixlMarshalCompressConfig>(m, "nixlMarshalCompressConfig")
        .def(py::init<>())
        .def(py::init([](nixl_marshal_compress_algo_t algo) {
                 nixlMarshalCompressConfig cfg;
                 cfg.algo = algo;
                 return cfg;
             }),
             py::arg("algo") = nixl_marshal_compress_algo_t::ANS)
        .def_readwrite("algo", &nixlMarshalCompressConfig::algo);

    // -----------------------------------------------------------------------
    // Marshal optional args (passed via nixl_service_opt_args_t)
    //
    // nixl_marshal_opt_args_t mirrors nixl_marshal_config_t: the variant
    // alternative must match the configured mode at call time. Placeholder
    // empty structs for now; per-mode knobs are added incrementally.
    // -----------------------------------------------------------------------
    py::class_<nixlMarshalDirectOptArgs>(m, "nixlMarshalDirectOptArgs")
        .def(py::init<>())
        .def("__repr__", [](const nixlMarshalDirectOptArgs &) {
            return std::string("nixlMarshalDirectOptArgs()");
        });

    py::class_<nixlMarshalStagingOptArgs>(m, "nixlMarshalStagingOptArgs")
        .def(py::init<>())
        .def("__repr__", [](const nixlMarshalStagingOptArgs &) {
            return std::string("nixlMarshalStagingOptArgs()");
        });

    py::class_<nixlMarshalDeltaOptArgs>(m, "nixlMarshalDeltaOptArgs")
        .def(py::init([](uintptr_t sender_ref,
                         uintptr_t receiver_ref,
                         nixl_mem_t sender_mem_type,
                         nixl_mem_t receiver_mem_type,
                         size_t element_size) {
                 nixlMarshalDeltaOptArgs args;
                 args.senderRef = reinterpret_cast<std::byte *>(sender_ref);
                 args.receiverRef = reinterpret_cast<std::byte *>(receiver_ref);
                 args.senderMemType = sender_mem_type;
                 args.receiverMemType = receiver_mem_type;
                 args.elementSize = element_size;
                 return args;
             }),
             py::arg("senderRef"),
             py::arg("receiverRef"),
             py::arg("senderMemType"),
             py::arg("receiverMemType"),
             py::arg("elementSize"))
        .def_readwrite("senderMemType", &nixlMarshalDeltaOptArgs::senderMemType)
        .def_readwrite("receiverMemType", &nixlMarshalDeltaOptArgs::receiverMemType)
        .def_readwrite("elementSize", &nixlMarshalDeltaOptArgs::elementSize)
        .def_property(
            "sender_ref",
            [](const nixlMarshalDeltaOptArgs &self) {
                return reinterpret_cast<uintptr_t>(self.senderRef);
            },
            [](nixlMarshalDeltaOptArgs &self, uintptr_t ptr) {
                self.senderRef = reinterpret_cast<std::byte *>(ptr);
            })
        .def_property(
            "receiver_ref",
            [](const nixlMarshalDeltaOptArgs &self) {
                return reinterpret_cast<uintptr_t>(self.receiverRef);
            },
            [](nixlMarshalDeltaOptArgs &self, uintptr_t ptr) {
                self.receiverRef = reinterpret_cast<std::byte *>(ptr);
            })
        .def("__repr__", [](const nixlMarshalDeltaOptArgs &self) {
            return "nixlMarshalDeltaOptArgs(senderMemType=" +
                std::to_string(static_cast<int>(self.senderMemType)) +
                ", receiverMemType=" + std::to_string(static_cast<int>(self.receiverMemType)) +
                ", sender_ref=" + std::to_string(reinterpret_cast<uintptr_t>(self.senderRef)) +
                ", receiver_ref=" + std::to_string(reinterpret_cast<uintptr_t>(self.receiverRef)) +
                ", elementSize=" + std::to_string(self.elementSize) + ")";
        });

    py::class_<nixlMarshalCompressOptArgs>(m, "nixlMarshalCompressOptArgs")
        .def(py::init<>())
        .def(py::init([](std::optional<nixlMarshalDeltaOptArgs> delta) {
                 nixlMarshalCompressOptArgs args;
                 args.delta = delta;
                 return args;
             }),
             py::arg("delta") = std::nullopt)
        .def_readwrite("delta", &nixlMarshalCompressOptArgs::delta)
        .def("__repr__", [](const nixlMarshalCompressOptArgs &) {
            return std::string("nixlMarshalCompressOptArgs()");
        });

    // -----------------------------------------------------------------------
    // Marshal sizing recommendation
    //
    // The ValueError below is pybind's mapping of the C++ std::invalid_argument.
    // -----------------------------------------------------------------------
    m.def(
        "recommendServiceMemSize",
        [](const nixl_marshal_config_t &mode, uint32_t max_concurrent_transfers) {
            return nixlService::recommendServiceMemSize(mode, max_concurrent_transfers);
        },
        py::arg("mode"),
        py::arg("maxConcurrentTransfers") = 1,
        "Recommend the service memory, in bytes per descriptor, to register for the "
        "given marshal mode.  Raises ValueError for nixlMarshalDirectConfig.");

    // -----------------------------------------------------------------------
    // nixlServiceAgentConfig
    //
    // Inherits from nixlAgentConfig, which must already be registered by
    // _bindings (import nixl._bindings before _service_bindings).
    // -----------------------------------------------------------------------
    py::class_<nixlServiceAgentConfig, nixlAgentConfig>(m, "nixlServiceAgentConfig")
        .def(py::init<>())
        .def_readwrite("mode", &nixlServiceAgentConfig::mode);

    // -----------------------------------------------------------------------
    // nixl_service_opt_args_t — optional args carrier. The configured agent
    // marshal mode is used by default; explicit per-transfer override must be
    // direct or match the configured marshal mode.
    // -----------------------------------------------------------------------
    py::class_<nixl_service_opt_args_t>(m, "nixl_service_opt_args_t")
        .def(py::init<>())
        .def_readwrite("marshalOptArgs", &nixl_service_opt_args_t::marshalOptArgs);

    // -----------------------------------------------------------------------
    // nixlServiceAgent
    //
    // IS-A nixlAgent — every method on nixlAgent is inherited.  We rebind
    // the transfer-handle methods here because:
    //   * they take nixlServiceXferReqH* instead of nixlXferReqH*; and
    //   * the C++ `= delete`d base-type overloads in nixl_service.h do not
    //     propagate through pybind11 — without an explicit rebind, Python
    //     users of a nixlServiceAgent instance would silently get the base
    //     implementation back (and no service features).
    //
    // Handles are passed across the Python boundary as uintptr_t, matching
    // the convention for nixlXferReqH / nixlDlistH / nixlBackendH.
    // -----------------------------------------------------------------------
    py::class_<nixlServiceAgent, nixlAgent>(m, "nixlServiceAgent")
        .def(py::init<std::string, nixlServiceAgentConfig>(), py::arg("name"), py::arg("cfg"))

        .def("getLocalMD",
             [](const nixlServiceAgent &agent) -> py::bytes {
                 std::string md;
                 throw_nixl_exception(agent.getLocalMD(md));
                 return py::bytes(md);
             })

        .def(
            "loadRemoteMD",
            [](nixlServiceAgent &agent, const std::string &remote_metadata) -> py::bytes {
                std::string remote_name;
                {
                    py::gil_scoped_release release;
                    throw_nixl_exception(agent.loadRemoteMD(remote_metadata, remote_name));
                }
                return py::bytes(remote_name);
            },
            py::arg("remote_metadata"))

        .def(
            "registerServiceMem",
            [](nixlServiceAgent &agent,
               const nixl_reg_dlist_t &descs,
               const std::vector<uintptr_t> &backends) -> nixl_status_t {
                nixl_opt_args_t extra_params;
                set_backends(extra_params, backends);

                const nixl_status_t ret = agent.registerServiceMem(descs, &extra_params);
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("descs"),
            py::arg("backends") = std::vector<uintptr_t>{},
            py::call_guard<py::gil_scoped_release>())

        .def(
            "deregisterServiceMem",
            [](nixlServiceAgent &agent,
               const nixl_reg_dlist_t &descs,
               const std::vector<uintptr_t> &backends) -> nixl_status_t {
                nixl_opt_args_t extra_params;
                set_backends(extra_params, backends);

                const nixl_status_t ret = agent.deregisterServiceMem(descs, &extra_params);
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("descs"),
            py::arg("backends") = std::vector<uintptr_t>{},
            py::call_guard<py::gil_scoped_release>())

        .def(
            "createXferReq",
            [](nixlServiceAgent &agent,
               const nixl_xfer_op_t &operation,
               const nixl_xfer_dlist_t &local_descs,
               const nixl_xfer_dlist_t &remote_descs,
               const std::string &remote_agent,
               const std::string &notif_msg,
               const std::vector<uintptr_t> &backends,
               const std::optional<nixl_marshal_opt_args_t> &marshal_opt_args) -> uintptr_t {
                nixlServiceXferReqH *handle = nullptr;
                nixl_service_opt_args_t extra_params{};
                set_backends(extra_params, backends);
                if (!notif_msg.empty()) {
                    extra_params.notif = notif_msg;
                }
                if (marshal_opt_args.has_value()) {
                    extra_params.marshalOptArgs = *marshal_opt_args;
                }

                throw_nixl_exception(agent.createXferReq(
                    operation, local_descs, remote_descs, remote_agent, handle, &extra_params));
                return reinterpret_cast<uintptr_t>(handle);
            },
            py::arg("operation"),
            py::arg("local_descs"),
            py::arg("remote_descs"),
            py::arg("remote_agent"),
            py::arg("notif_msg") = std::string{},
            py::arg("backends") = std::vector<uintptr_t>{},
            py::arg("marshal_opt_args") = std::nullopt,
            py::call_guard<py::gil_scoped_release>())

        .def(
            "makeXferReq",
            [](nixlServiceAgent &agent,
               const nixl_xfer_op_t &operation,
               uintptr_t local_side,
               const std::vector<int> &local_indices,
               uintptr_t remote_side,
               const std::vector<int> &remote_indices,
               const std::string &notif_msg,
               const std::vector<uintptr_t> &backends,
               bool skip_desc_merge,
               const std::optional<nixl_marshal_opt_args_t> &marshal_opt_args) -> uintptr_t {
                nixlServiceXferReqH *handle = nullptr;
                nixl_service_opt_args_t extra_params{};
                set_backends(extra_params, backends);
                if (!notif_msg.empty()) {
                    extra_params.notif = notif_msg;
                }
                extra_params.skipDescMerge = skip_desc_merge;
                if (marshal_opt_args.has_value()) {
                    extra_params.marshalOptArgs = *marshal_opt_args;
                }

                throw_nixl_exception(agent.makeXferReq(operation,
                                                       reinterpret_cast<nixlDlistH *>(local_side),
                                                       local_indices,
                                                       reinterpret_cast<nixlDlistH *>(remote_side),
                                                       remote_indices,
                                                       handle,
                                                       &extra_params));
                return reinterpret_cast<uintptr_t>(handle);
            },
            py::arg("operation"),
            py::arg("local_side"),
            py::arg("local_indices"),
            py::arg("remote_side"),
            py::arg("remote_indices"),
            py::arg("notif_msg") = std::string{},
            py::arg("backends") = std::vector<uintptr_t>{},
            py::arg("skip_desc_merge") = false,
            py::arg("marshal_opt_args") = std::nullopt)

        .def(
            "postXferReq",
            [](nixlServiceAgent &agent,
               uintptr_t reqh,
               const std::string &notif_msg,
               const std::optional<nixl_marshal_opt_args_t> &marshal_opt_args) -> nixl_status_t {
                nixl_status_t ret;
                auto *h = reinterpret_cast<nixlServiceXferReqH *>(reqh);
                if (!notif_msg.empty() || marshal_opt_args.has_value()) {
                    nixl_service_opt_args_t extra_params{};
                    if (!notif_msg.empty()) {
                        extra_params.notif = notif_msg;
                    }
                    if (marshal_opt_args.has_value()) {
                        extra_params.marshalOptArgs = *marshal_opt_args;
                    }
                    ret = agent.postXferReq(h, &extra_params);
                } else {
                    ret = agent.postXferReq(h);
                }
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("reqh"),
            py::arg("notif_msg") = std::string{},
            py::arg("marshal_opt_args") = std::nullopt,
            py::call_guard<py::gil_scoped_release>())

        .def(
            "getXferStatus",
            [](nixlServiceAgent &agent, uintptr_t reqh) -> nixl_status_t {
                const nixl_status_t ret =
                    agent.getXferStatus(reinterpret_cast<nixlServiceXferReqH *>(reqh));
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("reqh"),
            py::call_guard<py::gil_scoped_release>())

        .def(
            "releaseXferReq",
            [](nixlServiceAgent &agent, uintptr_t reqh) -> nixl_status_t {
                const nixl_status_t ret =
                    agent.releaseXferReq(reinterpret_cast<nixlServiceXferReqH *>(reqh));
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("reqh"))

        .def(
            "getNotifs",
            [](nixlServiceAgent &agent,
               nixl_py_notifs_t &notif_map,
               const std::vector<uintptr_t> &backends) -> nixl_py_notifs_t {
                nixl_notifs_t new_notifs;
                nixl_opt_args_t extra_params;
                {
                    py::gil_scoped_release release;
                    set_backends(extra_params, backends);
                    throw_nixl_exception(agent.getNotifs(new_notifs, &extra_params));
                }

                for (const auto &pair : new_notifs) {
                    for (const auto &str : pair.second) {
                        notif_map[pair.first].push_back(py::bytes(str));
                    }
                }
                return notif_map;
            },
            py::arg("notif_map"),
            py::arg("backends") = std::vector<uintptr_t>{});
}
