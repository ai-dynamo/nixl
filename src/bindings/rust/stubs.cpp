/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Lazy-loading stubs for the NIXL C API.
//
// When nixl is not available at build time, these stubs are compiled in.
// At runtime, they attempt to dlopen the nixl C API shared library (by
// soname first, then the dev symlink) and forward all calls to the real
// implementation. If the library cannot be loaded, or its major version
// does not match the header these stubs were compiled against,
// status-returning functions fail with NIXL_CAPI_ERROR_INVALID_STATE and
// nixl_capi_last_error_message() reports why the load failed. When no
// library is loaded, destroy and release functions succeed instead: no
// library-created handle can exist, so there is nothing to destroy.
//
// This allows building without nixl present while still using nixl at
// runtime when the shared library is installed.

#include "nixl_capi.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <iostream>

namespace {

// Result of the one-shot dlopen attempt, capturing both the handle and any
// error string so the diagnostic is not lost between get_nixl_handle() and
// resolve().
struct NixlHandle {
    void *handle;
    const char *error; // load failure reason; nullptr on success
};

// Thread-safe lazy initialization of the nixl C API shared library handle.
// C++11 guarantees thread-safe initialization of function-local static variables.
const NixlHandle &
get_nixl_handle() {
    static NixlHandle h = []() -> NixlHandle {
        static const char *const names[] = {
            "libnixl_capi.so.0",
            "libnixl_capi.so",
        };
        void *hdl = nullptr;
        const char *err = nullptr;
        for (const char *name : names) {
            hdl = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (hdl) {
                break;
            }
            // Keep the first (soname) failure: an installed runtime package
            // provides libnixl_capi.so.0 and usually omits the dev symlink,
            // so the later candidates' errors are less informative.
            if (!err) {
                static char dlerr[256];
                const char *e = dlerror();
                if (e) {
                    snprintf(dlerr, sizeof(dlerr), "%s", e);
                    err = dlerr;
                } else {
                    err = "dlopen failed";
                }
            }
        }
        if (!hdl) {
            return {nullptr, err};
        }
        // Version handshake: reject a library whose major version differs
        // from the header these stubs were compiled against.
        using ver_fn_t = void (*)(int *, int *, int *);
        auto ver = (ver_fn_t)dlsym(hdl, "nixl_capi_get_version");
        if (!ver) {
            dlclose(hdl);
            return {nullptr, "loaded nixl C API library has no nixl_capi_get_version symbol"};
        }
        int major = -1, minor = -1, patch = -1;
        ver(&major, &minor, &patch);
        if (major != NIXL_CAPI_API_VERSION_MAJOR) {
            dlclose(hdl);
            static char msg[128];
            snprintf(msg,
                     sizeof(msg),
                     "nixl C API major version mismatch: library is %d.%d.%d, "
                     "bindings expect major %d",
                     major,
                     minor,
                     patch,
                     NIXL_CAPI_API_VERSION_MAJOR);
            return {nullptr, msg};
        }
        return {hdl, nullptr};
    }();
    return h;
}

// Resolve a symbol from the nixl C API shared library. Returns nullptr (after
// printing a one-time diagnostic) if the library is not usable or the symbol
// is missing; callers then fail with NIXL_CAPI_ERROR_INVALID_STATE.
void *
resolve(const char *name) {
    const auto &h = get_nixl_handle();
    if (!h.handle) {
        static bool warned = [](const char *error) {
            std::cerr << "nixl error: nixl C API library not usable: "
                      << (error ? error : "unknown error")
                      << ". Install nixl or ensure the nixl library directory "
                      << "is in LD_LIBRARY_PATH.\n";
            return true;
        }(h.error);
        (void)warned;
        return nullptr;
    }
    dlerror(); // clear any stale error
    void *sym = dlsym(h.handle, name);
    const char *err = dlerror();
    if (err) {
        std::cerr << "nixl error: symbol '" << name << "' not found in nixl C API library: " << err
                  << "\n";
        return nullptr;
    }
    return sym;
}

} // anonymous namespace

extern "C" {

// clang-format off
// Opaque struct definitions (never dereferenced by stubs; needed for type completeness)
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
struct nixl_capi_xfer_dlist_handle_s { /* empty */ };
struct nixl_capi_query_resp_list_s { /* empty */ };
// clang-format on

// Every status-returning function forwards identically; NIXL_CAPI_STUB
// expands to a definition that resolves the real symbol once and fails with
// NIXL_CAPI_ERROR_INVALID_STATE when the library is unusable. params is the
// parenthesized parameter list, args the matching parenthesized call.
#define NIXL_CAPI_STUB(name, params, args)                                     \
    nixl_capi_status_t name params {                                           \
        static const auto real = (nixl_capi_status_t(*) params)resolve(#name); \
        if (!real) {                                                           \
            return NIXL_CAPI_ERROR_INVALID_STATE;                              \
        }                                                                      \
        return real args;                                                      \
    }

// Destroy/release variant: when no library is loaded, no library-created
// handle can exist, so there is nothing to destroy and the call succeeds.
// A loaded library that lacks the symbol is a real mismatch and fails like
// any other call.
#define NIXL_CAPI_STUB_DESTROY(name, params, args)                                               \
    nixl_capi_status_t name params {                                                             \
        static const auto real = (nixl_capi_status_t(*) params)resolve(#name);                   \
        if (!real) {                                                                             \
            return get_nixl_handle().handle ? NIXL_CAPI_ERROR_INVALID_STATE : NIXL_CAPI_SUCCESS; \
        }                                                                                        \
        return real args;                                                                        \
    }

// clang-format off
NIXL_CAPI_STUB(nixl_capi_create_agent, (const char *name, nixl_capi_agent_t *agent), (name, agent))
NIXL_CAPI_STUB(nixl_capi_create_configured_agent, (const char *name, const nixl_capi_agent_config_t *cfg, nixl_capi_agent_t *agent), (name, cfg, agent))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_agent, (nixl_capi_agent_t agent), (agent))
NIXL_CAPI_STUB(nixl_capi_get_local_md, (nixl_capi_agent_t agent, void **data, size_t *len), (agent, data, len))
NIXL_CAPI_STUB(nixl_capi_get_local_partial_md, (nixl_capi_agent_t agent, nixl_capi_reg_dlist_t descs, void **data, size_t *len, nixl_capi_opt_args_t opt_args), (agent, descs, data, len, opt_args))
NIXL_CAPI_STUB(nixl_capi_load_remote_md, (nixl_capi_agent_t agent, const void *data, size_t len, char **agent_name), (agent, data, len, agent_name))
NIXL_CAPI_STUB(nixl_capi_send_local_md, (nixl_capi_agent_t agent, nixl_capi_opt_args_t opt_args), (agent, opt_args))
NIXL_CAPI_STUB(nixl_capi_send_local_partial_md, (nixl_capi_agent_t agent, nixl_capi_reg_dlist_t descs, nixl_capi_opt_args_t opt_args), (agent, descs, opt_args))
NIXL_CAPI_STUB(nixl_capi_invalidate_remote_md, (nixl_capi_agent_t agent, const char *remote_agent), (agent, remote_agent))
NIXL_CAPI_STUB(nixl_capi_invalidate_local_md, (nixl_capi_agent_t agent, nixl_capi_opt_args_t opt_args), (agent, opt_args))
NIXL_CAPI_STUB(nixl_capi_check_remote_md, (nixl_capi_agent_t agent, const char *remote_name, nixl_capi_xfer_dlist_t descs), (agent, remote_name, descs))
NIXL_CAPI_STUB(nixl_capi_fetch_remote_md, (nixl_capi_agent_t agent, const char *remote_name, nixl_capi_opt_args_t opt_args), (agent, remote_name, opt_args))
NIXL_CAPI_STUB(nixl_capi_prep_xfer_dlist, (nixl_capi_agent_t agent, const char *agent_name, nixl_capi_xfer_dlist_t descs, nixl_capi_xfer_dlist_handle_t *dlist_handle, nixl_capi_opt_args_t opt_args), (agent, agent_name, descs, dlist_handle, opt_args))
NIXL_CAPI_STUB_DESTROY(nixl_capi_release_xfer_dlist_handle, (nixl_capi_agent_t agent, nixl_capi_xfer_dlist_handle_t dlist_handle), (agent, dlist_handle))
NIXL_CAPI_STUB(nixl_capi_make_xfer_req, (nixl_capi_agent_t agent, nixl_capi_xfer_op_t operation, nixl_capi_xfer_dlist_handle_t local_descs, const int *local_indices, size_t local_indices_count, nixl_capi_xfer_dlist_handle_t remote_descs, const int *remote_indices, size_t remote_indices_count, nixl_capi_xfer_req_t *req_hndl, nixl_capi_opt_args_t opt_args), (agent, operation, local_descs, local_indices, local_indices_count, remote_descs, remote_indices, remote_indices_count, req_hndl, opt_args))
NIXL_CAPI_STUB(nixl_capi_agent_make_connection, (nixl_capi_agent_t agent, const char *remote_agent, nixl_capi_opt_args_t opt_args), (agent, remote_agent, opt_args))
NIXL_CAPI_STUB(nixl_capi_get_available_plugins, (nixl_capi_agent_t agent, nixl_capi_string_list_t *plugins), (agent, plugins))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_string_list, (nixl_capi_string_list_t list), (list))
NIXL_CAPI_STUB(nixl_capi_string_list_size, (nixl_capi_string_list_t list, size_t *size), (list, size))
NIXL_CAPI_STUB(nixl_capi_string_list_get, (nixl_capi_string_list_t list, size_t index, const char **str), (list, index, str))
NIXL_CAPI_STUB(nixl_capi_get_plugin_params, (nixl_capi_agent_t agent, const char *plugin_name, nixl_capi_mem_list_t *mems, nixl_capi_params_t *params), (agent, plugin_name, mems, params))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_mem_list, (nixl_capi_mem_list_t list), (list))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_params, (nixl_capi_params_t params), (params))
NIXL_CAPI_STUB(nixl_capi_create_backend, (nixl_capi_agent_t agent, const char *plugin_name, nixl_capi_params_t params, nixl_capi_backend_t *backend), (agent, plugin_name, params, backend))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_backend, (nixl_capi_backend_t backend), (backend))
NIXL_CAPI_STUB(nixl_capi_get_backend_params, (nixl_capi_agent_t agent, nixl_capi_backend_t backend, nixl_capi_mem_list_t *mems, nixl_capi_params_t *params), (agent, backend, mems, params))
NIXL_CAPI_STUB(nixl_capi_create_opt_args, (nixl_capi_opt_args_t *args), (args))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_opt_args, (nixl_capi_opt_args_t args), (args))
NIXL_CAPI_STUB(nixl_capi_opt_args_add_backend, (nixl_capi_opt_args_t args, nixl_capi_backend_t backend), (args, backend))
NIXL_CAPI_STUB(nixl_capi_opt_args_set_notif_msg, (nixl_capi_opt_args_t args, const void *data, size_t len), (args, data, len))
NIXL_CAPI_STUB(nixl_capi_opt_args_get_notif_msg, (nixl_capi_opt_args_t args, void **data, size_t *len), (args, data, len))
NIXL_CAPI_STUB(nixl_capi_opt_args_set_custom_param, (nixl_capi_opt_args_t args, const void *data, size_t len), (args, data, len))
NIXL_CAPI_STUB(nixl_capi_opt_args_get_custom_param, (nixl_capi_opt_args_t args, void **data, size_t *len), (args, data, len))
NIXL_CAPI_STUB(nixl_capi_opt_args_set_has_notif, (nixl_capi_opt_args_t args, bool has_notif), (args, has_notif))
NIXL_CAPI_STUB(nixl_capi_opt_args_get_has_notif, (nixl_capi_opt_args_t args, bool *has_notif), (args, has_notif))
NIXL_CAPI_STUB(nixl_capi_opt_args_set_skip_desc_merge, (nixl_capi_opt_args_t args, bool skip_merge), (args, skip_merge))
NIXL_CAPI_STUB(nixl_capi_opt_args_get_skip_desc_merge, (nixl_capi_opt_args_t args, bool *skip_merge), (args, skip_merge))
NIXL_CAPI_STUB(nixl_capi_opt_args_set_ip_addr, (nixl_capi_opt_args_t args, const char *ip_addr), (args, ip_addr))
NIXL_CAPI_STUB(nixl_capi_opt_args_set_port, (nixl_capi_opt_args_t args, uint16_t port), (args, port))
NIXL_CAPI_STUB(nixl_capi_create_params, (nixl_capi_params_t *params), (params))
NIXL_CAPI_STUB(nixl_capi_params_add, (nixl_capi_params_t params, const char *key, const char *value), (params, key, value))
NIXL_CAPI_STUB(nixl_capi_params_is_empty, (nixl_capi_params_t params, bool *is_empty), (params, is_empty))
NIXL_CAPI_STUB(nixl_capi_params_create_iterator, (nixl_capi_params_t params, nixl_capi_param_iter_t *iter), (params, iter))
NIXL_CAPI_STUB(nixl_capi_params_iterator_next, (nixl_capi_param_iter_t iter, const char **key, const char **value, bool *has_next), (iter, key, value, has_next))
NIXL_CAPI_STUB_DESTROY(nixl_capi_params_destroy_iterator, (nixl_capi_param_iter_t iter), (iter))
NIXL_CAPI_STUB(nixl_capi_mem_list_is_empty, (nixl_capi_mem_list_t list, bool *is_empty), (list, is_empty))
NIXL_CAPI_STUB(nixl_capi_mem_list_size, (nixl_capi_mem_list_t list, size_t *size), (list, size))
NIXL_CAPI_STUB(nixl_capi_mem_list_get, (nixl_capi_mem_list_t list, size_t index, nixl_capi_mem_type_t *mem_type), (list, index, mem_type))
NIXL_CAPI_STUB(nixl_capi_mem_type_to_string, (nixl_capi_mem_type_t mem_type, const char **str), (mem_type, str))
NIXL_CAPI_STUB(nixl_capi_create_xfer_dlist, (nixl_capi_mem_type_t mem_type, nixl_capi_xfer_dlist_t *dlist), (mem_type, dlist))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_xfer_dlist, (nixl_capi_xfer_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_add_desc, (nixl_capi_xfer_dlist_t dlist, uintptr_t addr, size_t len, uint64_t dev_id), (dlist, addr, len, dev_id))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_len, (nixl_capi_xfer_dlist_t dlist, size_t *len), (dlist, len))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_clear, (nixl_capi_xfer_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_resize, (nixl_capi_xfer_dlist_t dlist, size_t new_size), (dlist, new_size))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_get_type, (nixl_capi_xfer_dlist_t dlist, nixl_capi_mem_type_t *mem_type), (dlist, mem_type))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_desc_count, (nixl_capi_xfer_dlist_t dlist, size_t *count), (dlist, count))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_is_empty, (nixl_capi_xfer_dlist_t dlist, bool *is_empty), (dlist, is_empty))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_trim, (nixl_capi_xfer_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_rem_desc, (nixl_capi_xfer_dlist_t dlist, int index), (dlist, index))
NIXL_CAPI_STUB(nixl_capi_xfer_dlist_print, (nixl_capi_xfer_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_create_reg_dlist, (nixl_capi_mem_type_t mem_type, nixl_capi_reg_dlist_t *dlist), (mem_type, dlist))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_reg_dlist, (nixl_capi_reg_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_add_desc, (nixl_capi_reg_dlist_t dlist, uintptr_t addr, size_t len, uint64_t dev_id, const void *metadata, size_t metadata_len), (dlist, addr, len, dev_id, metadata, metadata_len))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_clear, (nixl_capi_reg_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_resize, (nixl_capi_reg_dlist_t dlist, size_t new_size), (dlist, new_size))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_get_type, (nixl_capi_reg_dlist_t dlist, nixl_capi_mem_type_t *mem_type), (dlist, mem_type))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_desc_count, (nixl_capi_reg_dlist_t dlist, size_t *count), (dlist, count))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_is_empty, (nixl_capi_reg_dlist_t dlist, bool *is_empty), (dlist, is_empty))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_trim, (nixl_capi_reg_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_rem_desc, (nixl_capi_reg_dlist_t dlist, int index), (dlist, index))
NIXL_CAPI_STUB(nixl_capi_reg_dlist_print, (nixl_capi_reg_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_register_mem, (nixl_capi_agent_t agent, nixl_capi_reg_dlist_t dlist, nixl_capi_opt_args_t opt_args), (agent, dlist, opt_args))
NIXL_CAPI_STUB(nixl_capi_deregister_mem, (nixl_capi_agent_t agent, nixl_capi_reg_dlist_t dlist, nixl_capi_opt_args_t opt_args), (agent, dlist, opt_args))
NIXL_CAPI_STUB(nixl_capi_create_xfer_req, (nixl_capi_agent_t agent, nixl_capi_xfer_op_t operation, nixl_capi_xfer_dlist_t local_descs, nixl_capi_xfer_dlist_t remote_descs, const char *remote_agent, nixl_capi_xfer_req_t *req_hndl, nixl_capi_opt_args_t opt_args), (agent, operation, local_descs, remote_descs, remote_agent, req_hndl, opt_args))
NIXL_CAPI_STUB(nixl_capi_post_xfer_req, (nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl, nixl_capi_opt_args_t opt_args), (agent, req_hndl, opt_args))
NIXL_CAPI_STUB(nixl_capi_get_xfer_status, (nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl), (agent, req_hndl))
NIXL_CAPI_STUB(nixl_capi_query_xfer_backend, (nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl, nixl_capi_backend_t *backend), (agent, req_hndl, backend))
NIXL_CAPI_STUB(nixl_capi_estimate_xfer_cost, (nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl, nixl_capi_opt_args_t opt_args, int64_t *duration_us, int64_t *err_margin_us, nixl_capi_cost_t *method), (agent, req_hndl, opt_args, duration_us, err_margin_us, method))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_xfer_req, (nixl_capi_xfer_req_t req), (req))
NIXL_CAPI_STUB_DESTROY(nixl_capi_release_xfer_req, (nixl_capi_agent_t agent, nixl_capi_xfer_req_t req), (agent, req))
NIXL_CAPI_STUB(nixl_capi_get_notifs, (nixl_capi_agent_t agent, nixl_capi_notif_map_t notif_map, nixl_capi_opt_args_t opt_args), (agent, notif_map, opt_args))
NIXL_CAPI_STUB(nixl_capi_gen_notif, (nixl_capi_agent_t agent, const char *remote_agent, const void *data, size_t len, nixl_capi_opt_args_t opt_args), (agent, remote_agent, data, len, opt_args))
NIXL_CAPI_STUB(nixl_capi_create_notif_map, (nixl_capi_notif_map_t *notif_map), (notif_map))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_notif_map, (nixl_capi_notif_map_t notif_map), (notif_map))
NIXL_CAPI_STUB(nixl_capi_notif_map_size, (nixl_capi_notif_map_t map, size_t *size), (map, size))
NIXL_CAPI_STUB(nixl_capi_notif_map_get_agent_at, (nixl_capi_notif_map_t map, size_t index, const char **agent_name), (map, index, agent_name))
NIXL_CAPI_STUB(nixl_capi_notif_map_get_notifs_size, (nixl_capi_notif_map_t map, const char *agent_name, size_t *size), (map, agent_name, size))
NIXL_CAPI_STUB(nixl_capi_notif_map_get_notif, (nixl_capi_notif_map_t map, const char *agent_name, size_t index, const void **data, size_t *len), (map, agent_name, index, data, len))
NIXL_CAPI_STUB(nixl_capi_notif_map_clear, (nixl_capi_notif_map_t map), (map))
NIXL_CAPI_STUB(nixl_capi_create_query_resp_list, (nixl_capi_query_resp_list_t *list), (list))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_query_resp_list, (nixl_capi_query_resp_list_t list), (list))
NIXL_CAPI_STUB(nixl_capi_query_resp_list_size, (nixl_capi_query_resp_list_t list, size_t *size), (list, size))
NIXL_CAPI_STUB(nixl_capi_query_resp_list_has_value, (nixl_capi_query_resp_list_t list, size_t index, bool *has_value), (list, index, has_value))
NIXL_CAPI_STUB(nixl_capi_query_resp_list_get_params, (nixl_capi_query_resp_list_t list, size_t index, nixl_capi_params_t *params), (list, index, params))
NIXL_CAPI_STUB(nixl_capi_query_mem, (nixl_capi_agent_t agent, nixl_capi_reg_dlist_t descs, nixl_capi_query_resp_list_t resp, nixl_capi_opt_args_t opt_args), (agent, descs, resp, opt_args))
NIXL_CAPI_STUB(nixl_capi_prep_mem_view_local, (nixl_capi_agent_t agent, nixl_capi_xfer_dlist_t descs, nixl_capi_mem_view_t *mvh, nixl_capi_opt_args_t opt_args), (agent, descs, mvh, opt_args))
NIXL_CAPI_STUB(nixl_capi_prep_mem_view_remote, (nixl_capi_agent_t agent, nixl_capi_remote_dlist_t descs, nixl_capi_mem_view_t *mvh, nixl_capi_opt_args_t opt_args), (agent, descs, mvh, opt_args))
NIXL_CAPI_STUB(nixl_capi_create_remote_dlist, (nixl_capi_mem_type_t mem_type, nixl_capi_remote_dlist_t *dlist), (mem_type, dlist))
NIXL_CAPI_STUB_DESTROY(nixl_capi_destroy_remote_dlist, (nixl_capi_remote_dlist_t dlist), (dlist))
NIXL_CAPI_STUB(nixl_capi_remote_dlist_add_desc, (nixl_capi_remote_dlist_t dlist, uintptr_t addr, size_t len, uint64_t dev_id, const char *remote_agent), (dlist, addr, len, dev_id, remote_agent))
NIXL_CAPI_STUB_DESTROY(nixl_capi_release_mem_view, (nixl_capi_agent_t agent, nixl_capi_mem_view_t mvh), (agent, mvh))
NIXL_CAPI_STUB(nixl_capi_get_xfer_telemetry, (nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl, nixl_capi_xfer_telemetry_t telemetry), (agent, req_hndl, telemetry))
// clang-format on
#undef NIXL_CAPI_STUB
#undef NIXL_CAPI_STUB_DESTROY

void
nixl_capi_get_version(int *major, int *minor, int *patch) {
    using fn_t = void (*)(int *, int *, int *);
    static fn_t real = (fn_t)resolve("nixl_capi_get_version");
    if (!real) {
        // No usable library; report 0.0.0 rather than the header's version.
        if (major) {
            *major = 0;
        }
        if (minor) {
            *minor = 0;
        }
        if (patch) {
            *patch = 0;
        }
        return;
    }
    real(major, minor, patch);
}

const char *
nixl_capi_status_string(nixl_capi_status_t status) {
    using fn_t = const char *(*)(nixl_capi_status_t);
    static fn_t real = (fn_t)resolve("nixl_capi_status_string");
    if (!real) {
        (void)status;
        return "NIXL_CAPI_UNKNOWN_STATUS (nixl C API library not loaded)";
    }
    return real(status);
}

const char *
nixl_capi_last_error_message(void) {
    using fn_t = const char *(*)(void);
    static fn_t real = (fn_t)resolve("nixl_capi_last_error_message");
    if (!real) {
        // Report why the library could not be loaded, if known.
        const char *err = get_nixl_handle().error;
        return err ? err : "";
    }
    return real();
}

void
nixl_capi_mem_free(void *ptr) {
    using fn_t = void (*)(void *);
    static fn_t real = (fn_t)resolve("nixl_capi_mem_free");
    if (!real) {
        // No library: nothing the library allocated can exist, so ignore.
        // A loaded library missing the symbol is a real allocation that
        // cannot be freed here; warn once since void gives no error path.
        if (get_nixl_handle().handle) {
            static bool warned = []() {
                std::cerr << "nixl error: loaded nixl C API library has no "
                          << "nixl_capi_mem_free symbol; library allocations leak.\n";
                return true;
            }();
            (void)warned;
        }
        return;
    }
    real(ptr);
}

// ---- Stub detection ----
// Returns true if the real nixl library is NOT available at runtime.
// Unlike other functions, this does NOT abort when the library is missing.
bool
nixl_capi_is_stub() {
    return (get_nixl_handle().handle == nullptr);
}

} // extern "C"
