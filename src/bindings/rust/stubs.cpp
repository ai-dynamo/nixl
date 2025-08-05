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
#include "wrapper.h"

#include <cstdlib>
#include <iostream>

// NOTE: The original includes from nixl.h, nixl_types.h, cstring, exception, etc. are removed here.
// The original blank lines around includes are also implicitly handled by this replacement.

extern "C" {

// Internal struct definitions to match our opaque types
// These are now stubs as their internal details are no longer used.
struct nixl_capi_agent_s { /* empty */ };
struct nixl_capi_string_list_s { /* empty */ };
struct nixl_capi_params_s { /* empty */ };
struct nixl_capi_mem_list_s { /* empty */ };
struct nixl_capi_backend_s { /* empty */ };
struct nixl_capi_opt_args_s { /* empty */ };
struct nixl_capi_param_iter_s { /* empty */ };
struct nixl_capi_xfer_dlist_s { /* empty */ };
struct nixl_capi_reg_dlist_s { /* empty */ };
struct nixl_capi_xfer_req_s { /* empty */ };
struct nixl_capi_notif_map_s { /* empty */ };

nixl_capi_status_t
nixl_capi_stub_abort()
{
  std::cerr << "nixl error: detected use of the NIXL C API's stub; if you want to use NIXL, don't use the stub-api feature.\n";
  std::abort();
  return NIXL_CAPI_ERROR_EXCEPTION;
}

nixl_capi_status_t
nixl_capi_create_agent(const char* name, nixl_capi_agent_t* agent)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_agent(nixl_capi_agent_t agent)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_get_local_md(nixl_capi_agent_t agent, void** data, size_t* len)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_load_remote_md(nixl_capi_agent_t agent, const void* data, size_t len, char** agent_name)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_invalidate_remote_md(nixl_capi_agent_t agent, const char* remote_agent)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_get_available_plugins(nixl_capi_agent_t agent, nixl_capi_string_list_t* plugins)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_string_list(nixl_capi_string_list_t list)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_string_list_size(nixl_capi_string_list_t list, size_t* size)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_string_list_get(nixl_capi_string_list_t list, size_t index, const char** str)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_get_plugin_params(
    nixl_capi_agent_t agent, const char* plugin_name, nixl_capi_mem_list_t* mems, nixl_capi_params_t* params)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_mem_list(nixl_capi_mem_list_t list)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_params(nixl_capi_params_t params)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_create_backend(
    nixl_capi_agent_t agent, const char* plugin_name, nixl_capi_params_t params, nixl_capi_backend_t* backend)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_backend(nixl_capi_backend_t backend)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_create_opt_args(nixl_capi_opt_args_t* args)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_opt_args(nixl_capi_opt_args_t args)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_opt_args_add_backend(nixl_capi_opt_args_t args, nixl_capi_backend_t backend)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_opt_args_set_notif_msg(nixl_capi_opt_args_t args, const void* data, size_t len)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_opt_args_get_notif_msg(nixl_capi_opt_args_t args, void** data, size_t* len)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_opt_args_set_has_notif(nixl_capi_opt_args_t args, bool has_notif)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_opt_args_get_has_notif(nixl_capi_opt_args_t args, bool* has_notif)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_opt_args_set_skip_desc_merge(nixl_capi_opt_args_t args, bool skip_merge)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_opt_args_get_skip_desc_merge(nixl_capi_opt_args_t args, bool* skip_merge)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_params_is_empty(nixl_capi_params_t params, bool* is_empty)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_params_create_iterator(nixl_capi_params_t params, nixl_capi_param_iter_t* iter)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_params_iterator_next(nixl_capi_param_iter_t iter, const char** key, const char** value, bool* has_next)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_params_destroy_iterator(nixl_capi_param_iter_t iter)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_mem_list_is_empty(nixl_capi_mem_list_t list, bool* is_empty)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_mem_list_size(nixl_capi_mem_list_t list, size_t* size)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_mem_list_get(nixl_capi_mem_list_t list, size_t index, nixl_capi_mem_type_t* mem_type)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_mem_type_to_string(nixl_capi_mem_type_t mem_type, const char** str)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_get_backend_params(
    nixl_capi_agent_t agent, nixl_capi_backend_t backend, nixl_capi_mem_list_t* mems, nixl_capi_params_t* params)
{
  return nixl_capi_stub_abort();
}

// Transfer descriptor list functions
nixl_capi_status_t
nixl_capi_create_xfer_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_xfer_dlist_t* dlist, bool sorted)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_xfer_dlist(nixl_capi_xfer_dlist_t dlist)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_xfer_dlist_add_desc(nixl_capi_xfer_dlist_t dlist, uintptr_t addr, size_t len, uint64_t dev_id)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_xfer_dlist_len(nixl_capi_xfer_dlist_t dlist, size_t* len)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_xfer_dlist_has_overlaps(nixl_capi_xfer_dlist_t dlist, bool* has_overlaps)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_xfer_dlist_clear(nixl_capi_xfer_dlist_t dlist)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_xfer_dlist_resize(nixl_capi_xfer_dlist_t dlist, size_t new_size)
{
  return nixl_capi_stub_abort();
}

// Registration descriptor list functions
nixl_capi_status_t
nixl_capi_create_reg_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_reg_dlist_t* dlist, bool sorted)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_reg_dlist(nixl_capi_reg_dlist_t dlist)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_reg_dlist_add_desc(nixl_capi_reg_dlist_t dlist,
                             uintptr_t addr,
                             size_t len,
                             uint64_t dev_id,
                             const void *metadata,
                             size_t metadata_len) {
    return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_reg_dlist_len(nixl_capi_reg_dlist_t dlist, size_t* len)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_reg_dlist_has_overlaps(nixl_capi_reg_dlist_t dlist, bool* has_overlaps)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_reg_dlist_clear(nixl_capi_reg_dlist_t dlist)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_reg_dlist_resize(nixl_capi_reg_dlist_t dlist, size_t new_size)
{
  return nixl_capi_stub_abort();
}

// Memory registration functions
nixl_capi_status_t
nixl_capi_register_mem(nixl_capi_agent_t agent, nixl_capi_reg_dlist_t dlist, nixl_capi_opt_args_t opt_args)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_deregister_mem(nixl_capi_agent_t agent, nixl_capi_reg_dlist_t dlist, nixl_capi_opt_args_t opt_args)
{
  return nixl_capi_stub_abort();
}


nixl_capi_status_t
nixl_capi_create_xfer_req(
    nixl_capi_agent_t agent, nixl_capi_xfer_op_t operation, nixl_capi_xfer_dlist_t local_descs,
    nixl_capi_xfer_dlist_t remote_descs, const char* remote_agent, nixl_capi_xfer_req_t* req_hndl,
    nixl_capi_opt_args_t opt_args)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_post_xfer_req(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl, nixl_capi_opt_args_t opt_args)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_get_xfer_status(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_xfer_req(nixl_capi_xfer_req_t req)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_release_xfer_req(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_get_notifs(nixl_capi_agent_t agent, nixl_capi_notif_map_t notif_map, nixl_capi_opt_args_t opt_args)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_create_notif_map(nixl_capi_notif_map_t* notif_map)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_notif_map(nixl_capi_notif_map_t notif_map)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_notif_map_size(nixl_capi_notif_map_t map, size_t* size)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_notif_map_get_agent_at(nixl_capi_notif_map_t map, size_t index, const char** agent_name)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_notif_map_get_notifs_size(nixl_capi_notif_map_t map, const char* agent_name, size_t* size)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_notif_map_get_notif(
    nixl_capi_notif_map_t map, const char* agent_name, size_t index, const void** data, size_t* len)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_notif_map_clear(nixl_capi_notif_map_t map)
{
  return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_create_query_resp_list(nixl_capi_query_resp_list_t *list) {
    return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_destroy_query_resp_list(nixl_capi_query_resp_list_t list) {
    return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_query_resp_list_size(nixl_capi_query_resp_list_t list, size_t *size) {
    return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_query_resp_list_has_value(nixl_capi_query_resp_list_t list,
                                    size_t index,
                                    bool *has_value) {
    return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_query_resp_list_get_params(nixl_capi_query_resp_list_t list,
                                     size_t index,
                                     nixl_capi_params_t *params) {
    return nixl_capi_stub_abort();
}

nixl_capi_status_t
nixl_capi_query_mem(nixl_capi_agent_t agent,
                    nixl_capi_reg_dlist_t descs,
                    nixl_capi_query_resp_list_t resp,
                    nixl_capi_opt_args_t opt_args) {
    return nixl_capi_stub_abort();
}

}  // extern "C"
