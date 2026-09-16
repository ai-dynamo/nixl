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

#include "backend_engine_forwarder.h"

nixlBackendEngineForwarder::nixlBackendEngineForwarder(const nixlBackendInitParams *init_params,
                                                       nixlBackendEngine &inner)
    : nixlBackendEngine(init_params),
      inner_(inner) {
    initErr = inner_.getInitErr();
}

bool
nixlBackendEngineForwarder::supportsRemote() const {
    return inner_.supportsRemote();
}

bool
nixlBackendEngineForwarder::supportsLocal() const {
    return inner_.supportsLocal();
}

bool
nixlBackendEngineForwarder::supportsNotif() const {
    return inner_.supportsNotif();
}

nixl_mem_list_t
nixlBackendEngineForwarder::getSupportedMems() const {
    return inner_.getSupportedMems();
}

nixl_status_t
nixlBackendEngineForwarder::registerMem(const nixlBlobDesc &mem,
                                        const nixl_mem_t &nixl_mem,
                                        nixlBackendMD *&out) {
    return inner_.registerMem(mem, nixl_mem, out);
}

nixl_status_t
nixlBackendEngineForwarder::deregisterMem(nixlBackendMD *meta) {
    return inner_.deregisterMem(meta);
}

nixl_status_t
nixlBackendEngineForwarder::connect(const std::string &remote_agent) {
    return inner_.connect(remote_agent);
}

nixl_status_t
nixlBackendEngineForwarder::disconnect(const std::string &remote_agent) {
    return inner_.disconnect(remote_agent);
}

nixl_status_t
nixlBackendEngineForwarder::unloadMD(nixlBackendMD *input) {
    return inner_.unloadMD(input);
}

nixl_status_t
nixlBackendEngineForwarder::prepXfer(const nixl_xfer_op_t &operation,
                                     const nixl_meta_dlist_t &local,
                                     const nixl_meta_dlist_t &remote,
                                     const std::string &remote_agent,
                                     nixlBackendReqH *&handle,
                                     const nixl_opt_b_args_t *opt_args) const {
    return inner_.prepXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixl_status_t
nixlBackendEngineForwarder::postXfer(const nixl_xfer_op_t &operation,
                                     const nixl_meta_dlist_t &local,
                                     const nixl_meta_dlist_t &remote,
                                     const std::string &remote_agent,
                                     nixlBackendReqH *&handle,
                                     const nixl_opt_b_args_t *opt_args) const {
    return inner_.postXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixl_status_t
nixlBackendEngineForwarder::checkXfer(nixlBackendReqH *handle) const {
    return inner_.checkXfer(handle);
}

nixl_status_t
nixlBackendEngineForwarder::releaseReqH(nixlBackendReqH *handle) const {
    return inner_.releaseReqH(handle);
}

nixl_status_t
nixlBackendEngineForwarder::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                                        nixlMemViewH &mvh,
                                        const nixl_opt_b_args_t *opt_args) const {
    return inner_.prepMemView(dlist, mvh, opt_args);
}

nixl_status_t
nixlBackendEngineForwarder::prepMemView(const nixl_meta_dlist_t &dlist,
                                        nixlMemViewH &mvh,
                                        const nixl_opt_b_args_t *opt_args) const {
    return inner_.prepMemView(dlist, mvh, opt_args);
}

void
nixlBackendEngineForwarder::releaseMemView(nixlMemViewH mvh) const {
    inner_.releaseMemView(mvh);
}

nixl_status_t
nixlBackendEngineForwarder::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    return inner_.getPublicData(meta, str);
}

nixl_status_t
nixlBackendEngineForwarder::getConnInfo(std::string &str) const {
    return inner_.getConnInfo(str);
}

nixl_status_t
nixlBackendEngineForwarder::loadRemoteConnInfo(const std::string &remote_agent,
                                               const std::string &remote_conn_info) {
    return inner_.loadRemoteConnInfo(remote_agent, remote_conn_info);
}

nixl_status_t
nixlBackendEngineForwarder::loadRemoteMD(const nixlBlobDesc &input,
                                         const nixl_mem_t &nixl_mem,
                                         const std::string &remote_agent,
                                         nixlBackendMD *&output) {
    return inner_.loadRemoteMD(input, nixl_mem, remote_agent, output);
}

nixl_status_t
nixlBackendEngineForwarder::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    return inner_.loadLocalMD(input, output);
}

nixl_status_t
nixlBackendEngineForwarder::getNotifs(notif_list_t &notif_list) {
    return inner_.getNotifs(notif_list);
}

nixl_status_t
nixlBackendEngineForwarder::genNotif(const std::string &remote_agent,
                                     const std::string &msg) const {
    return inner_.genNotif(remote_agent, msg);
}

nixl_status_t
nixlBackendEngineForwarder::queryMem(const nixl_reg_dlist_t &descs,
                                     std::vector<nixl_query_resp_t> &resp) const {
    return inner_.queryMem(descs, resp);
}

nixl_status_t
nixlBackendEngineForwarder::estimateXferCost(const nixl_xfer_op_t &operation,
                                             const nixl_meta_dlist_t &local,
                                             const nixl_meta_dlist_t &remote,
                                             const std::string &remote_agent,
                                             nixlBackendReqH *const &handle,
                                             std::chrono::microseconds &duration,
                                             std::chrono::microseconds &err_margin,
                                             nixl_cost_t &method,
                                             const nixl_opt_args_t *extra_params) const {
    return inner_.estimateXferCost(
        operation, local, remote, remote_agent, handle, duration, err_margin, method, extra_params);
}
