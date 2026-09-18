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
#ifndef NIXL_SRC_API_C_NIXL_CAPI_H
#define NIXL_SRC_API_C_NIXL_CAPI_H

/**
 * @file nixl_capi.h
 * @brief C API for the NIXL library.
 *
 * EXPERIMENTAL: the nixl C API and ABI are not yet stable (soversion 0).
 * Symbols, types, and enum values may change between releases without notice.
 *
 * Ownership rules:
 *   - Every nixl_capi_create_* (or *_prep_*) call has exactly one matching
 *     destroy/release call; destroy child objects (backends, descriptor lists,
 *     transfer requests, memory views, ...) before destroying the agent that
 *     created them.
 *   - Every buffer the library returns through an out-pointer (local metadata,
 *     partial metadata, remote agent names, notification messages, custom
 *     parameters, ...) is owned by the caller and must be released with
 *     nixl_capi_mem_free().
 *   - All destroy functions accept NULL and return NIXL_CAPI_SUCCESS without
 *     doing anything.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Marks the public entry points of the nixl C API. The library is built with
// hidden symbol visibility, so only functions carrying this attribute are
// exported from the shared library.
#ifndef NIXL_CAPI_EXPORT
#if defined(__GNUC__) || defined(__clang__)
#define NIXL_CAPI_EXPORT __attribute__((visibility("default")))
#else
#define NIXL_CAPI_EXPORT
#endif
#endif

// Version of the nixl C API described by this header. Query the version of
// the library actually loaded at runtime with nixl_capi_get_version().
#define NIXL_CAPI_API_VERSION_MAJOR 0
#define NIXL_CAPI_API_VERSION_MINOR 1
#define NIXL_CAPI_API_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes for the nixl C API.
 *
 * Each NIXL_ERR_* value of nixl_status_t (nixl_types.h) has a C mirror; values
 * that predate the 1:1 mapping keep their original numbers, newer ones are
 * appended:
 *   NIXL_IN_PROG               -> NIXL_CAPI_IN_PROG
 *   NIXL_ERR_INVALID_PARAM     -> NIXL_CAPI_ERROR_INVALID_PARAM
 *   NIXL_ERR_BACKEND           -> NIXL_CAPI_ERROR_BACKEND
 *   NIXL_ERR_NO_TELEMETRY      -> NIXL_CAPI_ERROR_NO_TELEMETRY
 *   NIXL_ERR_NOT_FOUND         -> NIXL_CAPI_ERROR_NOT_FOUND
 *   NIXL_ERR_NOT_POSTED        -> NIXL_CAPI_ERROR_NOT_POSTED
 *   NIXL_ERR_MISMATCH          -> NIXL_CAPI_ERROR_MISMATCH
 *   NIXL_ERR_NOT_ALLOWED       -> NIXL_CAPI_ERROR_NOT_ALLOWED
 *   NIXL_ERR_REPOST_ACTIVE     -> NIXL_CAPI_ERROR_REPOST_ACTIVE
 *   NIXL_ERR_UNKNOWN           -> NIXL_CAPI_ERROR_UNKNOWN
 *   NIXL_ERR_NOT_SUPPORTED     -> NIXL_CAPI_ERROR_NOT_SUPPORTED
 *   NIXL_ERR_REMOTE_DISCONNECT -> NIXL_CAPI_ERROR_REMOTE_DISCONNECT
 *   NIXL_ERR_CANCELED          -> NIXL_CAPI_ERROR_CANCELED
 * NIXL_CAPI_ERROR_INVALID_STATE and NIXL_CAPI_ERROR_EXCEPTION are C-API-only:
 * the former reports misuse of a C wrapper object, the latter an unexpected
 * C++ exception caught at the API boundary (see
 * nixl_capi_last_error_message()).
 */
typedef enum {
    NIXL_CAPI_SUCCESS = 0,
    NIXL_CAPI_ERROR_INVALID_PARAM = -1,
    NIXL_CAPI_ERROR_BACKEND = -2,
    NIXL_CAPI_ERROR_INVALID_STATE = -3, ///< Misuse of a C wrapper object (C-API-only)
    NIXL_CAPI_ERROR_EXCEPTION = -4, ///< Unexpected C++ exception at the API boundary (C-API-only)
    NIXL_CAPI_IN_PROG = 1, ///< Operation still in progress (not an error)
    NIXL_CAPI_ERROR_NO_TELEMETRY = -5,
    NIXL_CAPI_ERROR_NOT_FOUND = -6, ///< Agent metadata not loaded, or no backend serves the request
    NIXL_CAPI_ERROR_NOT_POSTED = -7,
    NIXL_CAPI_ERROR_MISMATCH = -8,
    NIXL_CAPI_ERROR_NOT_ALLOWED = -9,
    NIXL_CAPI_ERROR_REPOST_ACTIVE = -10,
    NIXL_CAPI_ERROR_UNKNOWN = -11,
    NIXL_CAPI_ERROR_NOT_SUPPORTED = -12,
    NIXL_CAPI_ERROR_REMOTE_DISCONNECT = -13,
    NIXL_CAPI_ERROR_CANCELED = -14,
} nixl_capi_status_t;

/**
 * @brief Store the version of the loaded library in *major, *minor and *patch.
 *
 * NULL out-pointers are skipped.
 *
 * @param  major  [out] Major version, if non-NULL
 * @param  minor  [out] Minor version, if non-NULL
 * @param  patch  [out] Patch version, if non-NULL
 */
NIXL_CAPI_EXPORT void
nixl_capi_get_version(int *major, int *minor, int *patch);

/**
 * @brief Return a static, human-readable name for a status code.
 *
 * @param  status  [in] Status code to name
 * @return Static string naming the status code
 */
NIXL_CAPI_EXPORT const char *
nixl_capi_status_string(nixl_capi_status_t status);

/**
 * @brief Return the message of the last C++ exception that failed a nixl C API
 *        call on the calling thread, or "" if no such failure occurred.
 *
 * @return Message string, valid until the next failing call on this thread
 */
NIXL_CAPI_EXPORT const char *
nixl_capi_last_error_message(void);

/**
 * @brief Release a buffer returned by the library (see the ownership rules
 *        in the file-level documentation).
 *
 * @param  ptr  [in] Buffer to release; NULL is accepted and ignored
 */
NIXL_CAPI_EXPORT void
nixl_capi_mem_free(void *ptr);

// Memory types enum (matching nixl's memory types)
typedef enum {
    NIXL_CAPI_MEM_DRAM = 0,
    NIXL_CAPI_MEM_VRAM = 1,
    NIXL_CAPI_MEM_BLOCK = 2,
    NIXL_CAPI_MEM_OBJECT = 3,
    NIXL_CAPI_MEM_FILE = 4,
    NIXL_CAPI_MEM_UNKNOWN = 5
} nixl_capi_mem_type_t;

struct nixl_capi_agent_s;
struct nixl_capi_params_s;
struct nixl_capi_mem_list_s;
struct nixl_capi_string_list_s;
struct nixl_capi_backend_s;
struct nixl_capi_opt_args_s;
struct nixl_capi_param_iter_s;
struct nixl_capi_xfer_dlist_s;
struct nixl_capi_xfer_dlist_handle_s;
struct nixl_capi_reg_dlist_s;
struct nixl_capi_xfer_req_s;
struct nixl_capi_notif_map_s;
struct nixl_capi_query_resp_list_s;
struct nixl_capi_remote_dlist_s;
struct nixl_capi_mem_view_s;

struct nixl_capi_xfer_telemetry_s {
    uint64_t start_time_us; ///< Start time in microseconds since an unspecified steady-clock
                            ///< epoch; compare across transfers, not against wall-clock time
    uint64_t post_duration_us; ///< Post operation duration in microseconds
    uint64_t xfer_duration_us; ///< Transfer duration in microseconds
    uint64_t total_bytes; ///< Total bytes transferred
    uint64_t desc_count; ///< Number of descriptors
};

// Opaque handle types for C++ objects
typedef struct nixl_capi_agent_s *nixl_capi_agent_t;
typedef struct nixl_capi_params_s *nixl_capi_params_t;
typedef struct nixl_capi_mem_list_s *nixl_capi_mem_list_t;
typedef struct nixl_capi_string_list_s *nixl_capi_string_list_t;
typedef struct nixl_capi_backend_s *nixl_capi_backend_t;
typedef struct nixl_capi_opt_args_s *nixl_capi_opt_args_t;
typedef struct nixl_capi_param_iter_s *nixl_capi_param_iter_t;
typedef struct nixl_capi_xfer_dlist_s *nixl_capi_xfer_dlist_t;
typedef struct nixl_capi_xfer_dlist_handle_s *nixl_capi_xfer_dlist_handle_t;
typedef struct nixl_capi_reg_dlist_s *nixl_capi_reg_dlist_t;
typedef struct nixl_capi_xfer_req_s *nixl_capi_xfer_req_t;
typedef struct nixl_capi_notif_map_s *nixl_capi_notif_map_t;
typedef struct nixl_capi_query_resp_list_s *nixl_capi_query_resp_list_t;
typedef struct nixl_capi_remote_dlist_s *nixl_capi_remote_dlist_t;
typedef struct nixl_capi_mem_view_s *nixl_capi_mem_view_t;

// Thread sync enum matching nixl_thread_sync_t
typedef enum {
    NIXL_CAPI_THREAD_SYNC_NONE = 0,
    NIXL_CAPI_THREAD_SYNC_STRICT = 1,
    NIXL_CAPI_THREAD_SYNC_RW = 2,
    NIXL_CAPI_THREAD_SYNC_DEFAULT = NIXL_CAPI_THREAD_SYNC_NONE,
} nixl_capi_thread_sync_t;

// Agent configuration struct mirroring nixlAgentConfig constructor args
typedef struct nixl_capi_agent_config_s {
    bool enable_prog_thread;
    bool enable_listen_thread;
    uint16_t listen_port;
    nixl_capi_thread_sync_t thread_sync;
    unsigned int num_workers;
    uint64_t pthr_delay_us;
    uint64_t lthr_delay_us;
    bool capture_telemetry;
} nixl_capi_agent_config_t;

// Transfer operation enum
typedef enum {
    NIXL_CAPI_XFER_OP_READ = 0,
    NIXL_CAPI_XFER_OP_WRITE = 1,
} nixl_capi_xfer_op_t;

// Core API functions

/**
 * @brief Create an agent with a provided config.
 *
 * @param  name   [in]  Name of the agent
 * @param  cfg    [in]  Agent configuration
 * @param  agent  [out] Created agent
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_configured_agent(const char *name,
                                  const nixl_capi_agent_config_t *cfg,
                                  nixl_capi_agent_t *agent);

/**
 * @brief Create an agent with default config.
 *
 * @param  name   [in]  Name of the agent
 * @param  agent  [out] Created agent
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_agent(const char *name, nixl_capi_agent_t *agent);

/**
 * @brief Destroy an agent.
 *
 * @param  agent  [in] Agent to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_agent(nixl_capi_agent_t agent);

/**
 * @brief Get local metadata as a byte array.
 *
 * @param  agent  [in]  Agent to query
 * @param  data   [out] Metadata bytes, released with nixl_capi_mem_free(); NULL when empty
 * @param  len    [out] Byte count, 0 when empty
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_get_local_md(nixl_capi_agent_t agent, void **data, size_t *len);

/**
 * @brief Get local partial metadata as a byte array.
 *
 * @param  agent     [in]  Agent to query
 * @param  descs     [in]  Descriptor list selecting the memory to describe
 * @param  data      [out] Metadata bytes, released with nixl_capi_mem_free(); NULL when empty
 * @param  len       [out] Byte count, 0 when empty
 * @param  opt_args  [in]  Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_get_local_partial_md(nixl_capi_agent_t agent,
                               nixl_capi_reg_dlist_t descs,
                               void **data,
                               size_t *len,
                               nixl_capi_opt_args_t opt_args);

/**
 * @brief Load remote metadata from a byte array.
 *
 * @param  agent       [in]  Agent to load the metadata into
 * @param  data        [in]  Metadata bytes
 * @param  len         [in]  Byte count
 * @param  agent_name  [out] Name of the remote agent, released with nixl_capi_mem_free()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_load_remote_md(nixl_capi_agent_t agent, const void *data, size_t len, char **agent_name);

/**
 * @brief Invalidate remote agent metadata.
 *
 * @param  agent         [in] Agent holding the metadata
 * @param  remote_agent  [in] Name of the remote agent to invalidate
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_invalidate_remote_md(nixl_capi_agent_t agent, const char *remote_agent);

/**
 * @brief Invalidate local metadata in etcd.
 *
 * @param  agent     [in] Agent whose metadata is invalidated
 * @param  opt_args  [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_invalidate_local_md(nixl_capi_agent_t agent, nixl_capi_opt_args_t opt_args);

/**
 * @brief Check if remote metadata is available.
 *
 * @param  agent        [in] Agent to query
 * @param  remote_name  [in] Name of the remote agent
 * @param  descs        [in] Descriptor list the metadata must cover
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_check_remote_md(nixl_capi_agent_t agent,
                          const char *remote_name,
                          nixl_capi_xfer_dlist_t descs);

/**
 * @brief Send local metadata to etcd.
 *
 * @param  agent     [in] Agent whose metadata is sent
 * @param  opt_args  [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_send_local_md(nixl_capi_agent_t agent, nixl_capi_opt_args_t opt_args);

/**
 * @brief Send local partial metadata to etcd.
 *
 * @param  agent     [in] Agent whose metadata is sent
 * @param  descs     [in] Descriptor list selecting the memory to describe
 * @param  opt_args  [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_send_local_partial_md(nixl_capi_agent_t agent,
                                nixl_capi_reg_dlist_t descs,
                                nixl_capi_opt_args_t opt_args);

/**
 * @brief Fetch remote metadata from etcd.
 *
 * @param  agent        [in] Agent to load the metadata into
 * @param  remote_name  [in] Name of the remote agent to fetch
 * @param  opt_args     [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_fetch_remote_md(nixl_capi_agent_t agent,
                          const char *remote_name,
                          nixl_capi_opt_args_t opt_args);

// Plugin and parameter functions

/**
 * @brief Get the list of available backend plugins.
 *
 * @param  agent    [in]  Agent to query
 * @param  plugins  [out] Plugin name list, destroyed with nixl_capi_destroy_string_list()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_get_available_plugins(nixl_capi_agent_t agent, nixl_capi_string_list_t *plugins);

/**
 * @brief Destroy a string list.
 *
 * @param  list  [in] String list to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_string_list(nixl_capi_string_list_t list);

/**
 * @brief Get the number of strings in a string list.
 *
 * @param  list  [in]  String list to query
 * @param  size  [out] Number of strings
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_string_list_size(nixl_capi_string_list_t list, size_t *size);

/**
 * @brief Get the string at an index of a string list.
 *
 * @param  list   [in]  String list to query
 * @param  index  [in]  Index of the string
 * @param  str    [out] String pointer, valid while the list exists
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_string_list_get(nixl_capi_string_list_t list, size_t index, const char **str);

/**
 * @brief Get the supported memory types and default parameters of a plugin.
 *
 * @param  agent        [in]  Agent to query
 * @param  plugin_name  [in]  Name of the plugin
 * @param  mems         [out] Memory type list, destroyed with nixl_capi_destroy_mem_list()
 * @param  params       [out] Parameter map, destroyed with nixl_capi_destroy_params()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_get_plugin_params(nixl_capi_agent_t agent,
                            const char *plugin_name,
                            nixl_capi_mem_list_t *mems,
                            nixl_capi_params_t *params);

/**
 * @brief Destroy a memory type list.
 *
 * @param  list  [in] Memory type list to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_mem_list(nixl_capi_mem_list_t list);

/**
 * @brief Destroy a parameter map.
 *
 * @param  params  [in] Parameter map to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_params(nixl_capi_params_t params);

// Backend creation and management

/**
 * @brief Create a backend for a plugin with the given parameters.
 *
 * @param  agent        [in]  Agent that owns the backend
 * @param  plugin_name  [in]  Name of the plugin
 * @param  params       [in]  Backend initialization parameters
 * @param  backend      [out] Created backend
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_backend(nixl_capi_agent_t agent,
                         const char *plugin_name,
                         nixl_capi_params_t params,
                         nixl_capi_backend_t *backend);

/**
 * @brief Destroy a backend handle.
 *
 * @param  backend  [in] Backend to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_backend(nixl_capi_backend_t backend);

/**
 * @brief Get backend parameters after initialization.
 *
 * @param  agent    [in]  Agent that owns the backend
 * @param  backend  [in]  Backend to query
 * @param  mems     [out] Memory type list, destroyed with nixl_capi_destroy_mem_list()
 * @param  params   [out] Parameter map, destroyed with nixl_capi_destroy_params()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_get_backend_params(nixl_capi_agent_t agent,
                             nixl_capi_backend_t backend,
                             nixl_capi_mem_list_t *mems,
                             nixl_capi_params_t *params);

// Optional arguments management

/**
 * @brief Create an empty optional-arguments object.
 *
 * @param  args  [out] Created optional arguments
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_opt_args(nixl_capi_opt_args_t *args);

/**
 * @brief Destroy an optional-arguments object.
 *
 * @param  args  [in] Optional arguments to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_opt_args(nixl_capi_opt_args_t args);

/**
 * @brief Append a backend to the backend list of an optional-arguments object.
 *
 * @param  args     [in] Optional arguments
 * @param  backend  [in] Backend to append
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_add_backend(nixl_capi_opt_args_t args, nixl_capi_backend_t backend);

// OptArgs notification and merge control

/**
 * @brief Set the notification message of an optional-arguments object.
 *
 * @param  args  [in] Optional arguments
 * @param  data  [in] Message bytes, copied
 * @param  len   [in] Byte count
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_set_notif_msg(nixl_capi_opt_args_t args, const void *data, size_t len);

/**
 * @brief Get the notification message of an optional-arguments object.
 *
 * @param  args  [in]  Optional arguments
 * @param  data  [out] Message bytes, released with nixl_capi_mem_free(); NULL when empty
 * @param  len   [out] Byte count, 0 when empty
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_get_notif_msg(nixl_capi_opt_args_t args, void **data, size_t *len);

/**
 * @brief Set the backend custom parameter, a blob whose contents are backend-defined.
 *
 * Copies @a len bytes out of @a data, which may be freed once this returns. @a data
 * may be NULL only when @a len is 0.
 *
 * @param  args  [in]  Optional arguments
 * @param  data  [in]  Parameter bytes
 * @param  len   [in]  Byte count
 * @return nixl_capi_status_t Error code if call was not successful
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_set_custom_param(nixl_capi_opt_args_t args, const void *data, size_t len);

/**
 * @brief Get the backend custom parameter.
 *
 * On success the caller owns @a data and must release it with
 * @ref nixl_capi_mem_free. An unset
 * parameter yields NULL and a length of 0.
 *
 * @param  args  [in]  Optional arguments
 * @param  data  [out] Newly allocated copy of the parameter bytes
 * @param  len   [out] Byte count
 * @return nixl_capi_status_t Error code if call was not successful
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_get_custom_param(nixl_capi_opt_args_t args, void **data, size_t *len);

/**
 * @brief Set the has-notification flag of an optional-arguments object.
 *
 * @param  args       [in] Optional arguments
 * @param  has_notif  [in] Whether a notification is attached
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_set_has_notif(nixl_capi_opt_args_t args, bool has_notif);

/**
 * @brief Get the has-notification flag of an optional-arguments object.
 *
 * @param  args       [in]  Optional arguments
 * @param  has_notif  [out] Whether a notification is attached
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_get_has_notif(nixl_capi_opt_args_t args, bool *has_notif);

/**
 * @brief Set the skip-descriptor-merge flag of an optional-arguments object.
 *
 * @param  args        [in] Optional arguments
 * @param  skip_merge  [in] Whether to skip descriptor merging
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_set_skip_desc_merge(nixl_capi_opt_args_t args, bool skip_merge);

/**
 * @brief Get the skip-descriptor-merge flag of an optional-arguments object.
 *
 * @param  args        [in]  Optional arguments
 * @param  skip_merge  [out] Whether descriptor merging is skipped
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_get_skip_desc_merge(nixl_capi_opt_args_t args, bool *skip_merge);

/**
 * @brief Set the IP address of an optional-arguments object.
 *
 * @param  args     [in] Optional arguments
 * @param  ip_addr  [in] IP address string, copied
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_set_ip_addr(nixl_capi_opt_args_t args, const char *ip_addr);

/**
 * @brief Set the port of an optional-arguments object.
 *
 * @param  args  [in] Optional arguments
 * @param  port  [in] Port number
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_opt_args_set_port(nixl_capi_opt_args_t args, uint16_t port);

// Parameter access functions

/**
 * @brief Create an empty parameter map.
 *
 * @param  params  [out] Created parameter map
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_params(nixl_capi_params_t *params);

/**
 * @brief Add a key/value pair to a parameter map.
 *
 * @param  params  [in] Parameter map
 * @param  key     [in] Parameter key
 * @param  value   [in] Parameter value
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_params_add(nixl_capi_params_t params, const char *key, const char *value);

/**
 * @brief Check whether a parameter map is empty.
 *
 * @param  params    [in]  Parameter map
 * @param  is_empty  [out] Whether the map is empty
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_params_is_empty(nixl_capi_params_t params, bool *is_empty);

/**
 * @brief Create an iterator over a parameter map.
 *
 * @param  params  [in]  Parameter map
 * @param  iter    [out] Created iterator, destroyed with nixl_capi_params_destroy_iterator()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_params_create_iterator(nixl_capi_params_t params, nixl_capi_param_iter_t *iter);

/**
 * @brief Get the current key/value pair and advance the iterator.
 *
 * @param  iter      [in]  Parameter iterator
 * @param  key       [out] Current key, valid while the map exists
 * @param  value     [out] Current value, valid while the map exists
 * @param  has_next  [out] Whether another pair follows
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_params_iterator_next(nixl_capi_param_iter_t iter,
                               const char **key,
                               const char **value,
                               bool *has_next);

/**
 * @brief Destroy a parameter iterator.
 *
 * @param  iter  [in] Iterator to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_params_destroy_iterator(nixl_capi_param_iter_t iter);

// Memory list access functions

/**
 * @brief Check whether a memory type list is empty.
 *
 * @param  list      [in]  Memory type list
 * @param  is_empty  [out] Whether the list is empty
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_mem_list_is_empty(nixl_capi_mem_list_t list, bool *is_empty);

/**
 * @brief Get the number of entries in a memory type list.
 *
 * @param  list  [in]  Memory type list
 * @param  size  [out] Number of entries
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_mem_list_size(nixl_capi_mem_list_t list, size_t *size);

/**
 * @brief Get the memory type at an index of a memory type list.
 *
 * @param  list      [in]  Memory type list
 * @param  index     [in]  Index of the entry
 * @param  mem_type  [out] Memory type at that index
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_mem_list_get(nixl_capi_mem_list_t list, size_t index, nixl_capi_mem_type_t *mem_type);

/**
 * @brief Get a static, human-readable name for a memory type.
 *
 * @param  mem_type  [in]  Memory type
 * @param  str       [out] Static name string
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_mem_type_to_string(nixl_capi_mem_type_t mem_type, const char **str);

// Memory registration functions

/**
 * @brief Register memory described by a registration descriptor list.
 *
 * @param  agent     [in] Agent to register with
 * @param  dlist     [in] Registration descriptor list
 * @param  opt_args  [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_register_mem(nixl_capi_agent_t agent,
                       nixl_capi_reg_dlist_t dlist,
                       nixl_capi_opt_args_t opt_args);

/**
 * @brief Deregister memory described by a registration descriptor list.
 *
 * @param  agent     [in] Agent to deregister from
 * @param  dlist     [in] Registration descriptor list
 * @param  opt_args  [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_deregister_mem(nixl_capi_agent_t agent,
                         nixl_capi_reg_dlist_t dlist,
                         nixl_capi_opt_args_t opt_args);

/**
 * @brief Proactively establish a connection to a remote agent.
 *
 * @param  agent         [in] Local agent
 * @param  remote_agent  [in] Name of the remote agent
 * @param  opt_args      [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_agent_make_connection(nixl_capi_agent_t agent,
                                const char *remote_agent,
                                nixl_capi_opt_args_t opt_args);

/**
 * @brief Prepare a transfer descriptor list handle for use with
 *        nixl_capi_make_xfer_req().
 *
 * @param  agent       [in]  Local agent
 * @param  agent_name  [in]  Name of the agent the descriptors belong to
 * @param  descs       [in]  Transfer descriptor list
 * @param  dlist_hndl  [out] Prepared handle, released with
 *                           nixl_capi_release_xfer_dlist_handle()
 * @param  opt_args    [in]  Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_prep_xfer_dlist(nixl_capi_agent_t agent,
                          const char *agent_name,
                          nixl_capi_xfer_dlist_t descs,
                          nixl_capi_xfer_dlist_handle_t *dlist_hndl,
                          nixl_capi_opt_args_t opt_args);

/**
 * @brief Release a prepared transfer descriptor list handle.
 *
 * @param  agent         [in] Agent that prepared the handle
 * @param  dlist_handle  [in] Handle to release
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_release_xfer_dlist_handle(nixl_capi_agent_t agent,
                                    nixl_capi_xfer_dlist_handle_t dlist_handle);

/**
 * @brief Make a transfer request from prepared descriptor list handles,
 *        selecting descriptors by index.
 *
 * @param  agent                 [in]  Local agent
 * @param  operation             [in]  Transfer operation (read or write)
 * @param  local_descs           [in]  Prepared local descriptor list handle
 * @param  local_indices         [in]  Indices into the local list
 * @param  local_indices_count   [in]  Number of local indices
 * @param  remote_descs          [in]  Prepared remote descriptor list handle
 * @param  remote_indices        [in]  Indices into the remote list
 * @param  remote_indices_count  [in]  Number of remote indices
 * @param  req_hndl              [out] Created transfer request
 * @param  opt_args              [in]  Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_make_xfer_req(nixl_capi_agent_t agent,
                        nixl_capi_xfer_op_t operation,
                        nixl_capi_xfer_dlist_handle_t local_descs,
                        const int *local_indices,
                        size_t local_indices_count,
                        nixl_capi_xfer_dlist_handle_t remote_descs,
                        const int *remote_indices,
                        size_t remote_indices_count,
                        nixl_capi_xfer_req_t *req_hndl,
                        nixl_capi_opt_args_t opt_args);

// Notification functions

/**
 * @brief Add received notifications to a notification map.
 *
 * @param  agent      [in]     Agent to poll
 * @param  notif_map  [in,out] Notification map the notifications are added to
 * @param  opt_args   [in]     Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_get_notifs(nixl_capi_agent_t agent,
                     nixl_capi_notif_map_t notif_map,
                     nixl_capi_opt_args_t opt_args);

/**
 * @brief Create an empty notification map.
 *
 * @param  notif_map  [out] Created notification map
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_notif_map(nixl_capi_notif_map_t *notif_map);

/**
 * @brief Destroy a notification map.
 *
 * @param  notif_map  [in] Notification map to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_notif_map(nixl_capi_notif_map_t notif_map);

/**
 * @brief Send a notification to a remote agent.
 *
 * @param  agent         [in] Local agent
 * @param  remote_agent  [in] Name of the remote agent
 * @param  data          [in] Message bytes, copied
 * @param  len           [in] Byte count
 * @param  opt_args      [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_gen_notif(nixl_capi_agent_t agent,
                    const char *remote_agent,
                    const void *data,
                    size_t len,
                    nixl_capi_opt_args_t opt_args);

// Transfer request functions

/**
 * @brief Create a transfer request from transfer descriptor lists.
 *
 * @param  agent         [in]  Local agent
 * @param  operation     [in]  Transfer operation (read or write)
 * @param  local_descs   [in]  Local transfer descriptor list
 * @param  remote_descs  [in]  Remote transfer descriptor list
 * @param  remote_agent  [in]  Name of the remote agent
 * @param  req_hndl      [out] Created transfer request
 * @param  opt_args      [in]  Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_xfer_req(nixl_capi_agent_t agent,
                          nixl_capi_xfer_op_t operation,
                          nixl_capi_xfer_dlist_t local_descs,
                          nixl_capi_xfer_dlist_t remote_descs,
                          const char *remote_agent,
                          nixl_capi_xfer_req_t *req_hndl,
                          nixl_capi_opt_args_t opt_args);

// Cost estimation method enum
typedef enum {
    NIXL_CAPI_COST_ANALYTICAL_BACKEND = 0, ///< Analytical estimate computed by the backend
} nixl_capi_cost_t;

/**
 * @brief Estimate the cost of a transfer request.
 *
 * @param  agent          [in]  Local agent
 * @param  req_hndl       [in]  Transfer request to estimate
 * @param  opt_args       [in]  Optional arguments, may be NULL
 * @param  duration_us    [out] Estimated duration in microseconds
 * @param  err_margin_us  [out] Error margin in microseconds
 * @param  method         [out] Estimation method used
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_estimate_xfer_cost(nixl_capi_agent_t agent,
                             nixl_capi_xfer_req_t req_hndl,
                             nixl_capi_opt_args_t opt_args,
                             int64_t *duration_us,
                             int64_t *err_margin_us,
                             nixl_capi_cost_t *method);

/**
 * @brief Post a transfer request for execution.
 *
 * @param  agent     [in] Local agent
 * @param  req_hndl  [in] Transfer request to post
 * @param  opt_args  [in] Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on completion, NIXL_CAPI_IN_PROG if still in
 *         progress, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_post_xfer_req(nixl_capi_agent_t agent,
                        nixl_capi_xfer_req_t req_hndl,
                        nixl_capi_opt_args_t opt_args);

/**
 * @brief Get the status of a posted transfer request.
 *
 * @param  agent     [in] Local agent
 * @param  req_hndl  [in] Transfer request to query
 * @return NIXL_CAPI_SUCCESS on completion, NIXL_CAPI_IN_PROG if still in
 *         progress, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_get_xfer_status(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl);

/**
 * @brief Query which backend was chosen for a transfer request.
 *
 * @param  agent     [in]  Local agent
 * @param  req_hndl  [in]  Transfer request to query
 * @param  backend   [out] Backend handle, destroyed with nixl_capi_destroy_backend()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_query_xfer_backend(nixl_capi_agent_t agent,
                             nixl_capi_xfer_req_t req_hndl,
                             nixl_capi_backend_t *backend);

/**
 * @brief Release the NIXL transfer resources held by @a req with the agent
 *        that created it.
 *
 * The wrapper remains and must still be passed to nixl_capi_destroy_xfer_req().
 *
 * @param  agent  [in] Agent that created the request
 * @param  req    [in] Transfer request to release
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_release_xfer_req(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req);

/**
 * @brief Free the transfer request wrapper only.
 *
 * Fails with NIXL_CAPI_ERROR_INVALID_STATE while @a req still holds NIXL
 * resources (release them with nixl_capi_release_xfer_req() first).
 *
 * @param  req  [in] Transfer request wrapper to free
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_xfer_req(nixl_capi_xfer_req_t req);

// Descriptor list functions

/**
 * @brief Create a transfer descriptor list.
 *
 * @param  mem_type  [in]  NIXL memory type of the descriptor list
 * @param  dlist     [out] Created list, destroyed with nixl_capi_destroy_xfer_dlist()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_xfer_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_xfer_dlist_t *dlist);

/**
 * @brief Destroy a transfer descriptor list.
 *
 * @param  dlist  [in] Descriptor list to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_xfer_dlist(nixl_capi_xfer_dlist_t dlist);

/**
 * @brief Get the memory type of a transfer descriptor list.
 *
 * @param  dlist     [in]  Descriptor list
 * @param  mem_type  [out] Memory type of the list
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_get_type(nixl_capi_xfer_dlist_t dlist, nixl_capi_mem_type_t *mem_type);

/**
 * @brief Add a descriptor to a transfer descriptor list.
 *
 * @param  dlist   [in] Descriptor list to add to
 * @param  addr    [in] Start of the buffer
 * @param  len     [in] Length of the buffer
 * @param  dev_id  [in] deviceID/BlockID/bufferID
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_add_desc(nixl_capi_xfer_dlist_t dlist,
                              uintptr_t addr,
                              size_t len,
                              uint64_t dev_id);

/**
 * @brief Get the number of descriptors in a transfer descriptor list.
 *
 * @param  dlist  [in]  Descriptor list
 * @param  count  [out] Number of descriptors
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_desc_count(nixl_capi_xfer_dlist_t dlist, size_t *count);

/**
 * @brief Get the number of descriptors in a transfer descriptor list.
 *
 * @param  dlist  [in]  Descriptor list
 * @param  len    [out] Number of descriptors in the list
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 *
 * @deprecated Use nixl_capi_xfer_dlist_desc_count() instead; this function
 *             returns a descriptor count, not a byte length.
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_len(nixl_capi_xfer_dlist_t dlist, size_t *len);

/**
 * @brief Check whether a transfer descriptor list is empty.
 *
 * @param  dlist     [in]  Descriptor list
 * @param  is_empty  [out] Whether the list is empty
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_is_empty(nixl_capi_xfer_dlist_t dlist, bool *is_empty);

/**
 * @brief Release the extra capacity of a transfer descriptor list.
 *
 * @param  dlist  [in] Descriptor list to trim
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_trim(nixl_capi_xfer_dlist_t dlist);

/**
 * @brief Remove the descriptor at an index of a transfer descriptor list.
 *
 * @param  dlist  [in] Descriptor list
 * @param  index  [in] Index of the descriptor to remove
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_rem_desc(nixl_capi_xfer_dlist_t dlist, int index);

/**
 * @brief Remove all descriptors from a transfer descriptor list.
 *
 * @param  dlist  [in] Descriptor list to clear
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_clear(nixl_capi_xfer_dlist_t dlist);

/**
 * @brief Resize a transfer descriptor list.
 *
 * @param  dlist     [in] Descriptor list to resize
 * @param  new_size  [in] New number of descriptors
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_resize(nixl_capi_xfer_dlist_t dlist, size_t new_size);

/**
 * @brief Print a transfer descriptor list to standard output.
 *
 * @param  dlist  [in] Descriptor list to print
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_xfer_dlist_print(nixl_capi_xfer_dlist_t dlist);

/**
 * @brief Create a registration descriptor list.
 *
 * @param  mem_type  [in]  NIXL memory type of the descriptor list
 * @param  dlist     [out] Created list, destroyed with nixl_capi_destroy_reg_dlist()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_reg_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_reg_dlist_t *dlist);

/**
 * @brief Destroy a registration descriptor list.
 *
 * @param  dlist  [in] Descriptor list to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_reg_dlist(nixl_capi_reg_dlist_t dlist);

/**
 * @brief Get the memory type of a registration descriptor list.
 *
 * @param  dlist     [in]  Descriptor list
 * @param  mem_type  [out] Memory type of the list
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_get_type(nixl_capi_reg_dlist_t dlist, nixl_capi_mem_type_t *mem_type);

/**
 * @brief Add a descriptor to a registration descriptor list.
 *
 * @param  dlist         [in] Descriptor list to add to
 * @param  addr          [in] Start of the buffer
 * @param  len           [in] Length of the buffer
 * @param  dev_id        [in] deviceID/BlockID/bufferID
 * @param  metadata      [in] Backend-specific metadata bytes, copied; may be
 *                            NULL when metadata_len is 0
 * @param  metadata_len  [in] Metadata byte count
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_add_desc(nixl_capi_reg_dlist_t dlist,
                             uintptr_t addr,
                             size_t len,
                             uint64_t dev_id,
                             const void *metadata,
                             size_t metadata_len);

/**
 * @brief Get the number of descriptors in a registration descriptor list.
 *
 * @param  dlist  [in]  Descriptor list
 * @param  count  [out] Number of descriptors
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_desc_count(nixl_capi_reg_dlist_t dlist, size_t *count);

/**
 * @brief Check whether a registration descriptor list is empty.
 *
 * @param  dlist     [in]  Descriptor list
 * @param  is_empty  [out] Whether the list is empty
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_is_empty(nixl_capi_reg_dlist_t dlist, bool *is_empty);

/**
 * @brief Release the extra capacity of a registration descriptor list.
 *
 * @param  dlist  [in] Descriptor list to trim
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_trim(nixl_capi_reg_dlist_t dlist);

/**
 * @brief Remove the descriptor at an index of a registration descriptor list.
 *
 * @param  dlist  [in] Descriptor list
 * @param  index  [in] Index of the descriptor to remove
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_rem_desc(nixl_capi_reg_dlist_t dlist, int index);

/**
 * @brief Remove all descriptors from a registration descriptor list.
 *
 * @param  dlist  [in] Descriptor list to clear
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_clear(nixl_capi_reg_dlist_t dlist);

/**
 * @brief Resize a registration descriptor list.
 *
 * @param  dlist     [in] Descriptor list to resize
 * @param  new_size  [in] New number of descriptors
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_resize(nixl_capi_reg_dlist_t dlist, size_t new_size);

/**
 * @brief Print a registration descriptor list to standard output.
 *
 * @param  dlist  [in] Descriptor list to print
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_reg_dlist_print(nixl_capi_reg_dlist_t dlist);

/**
 * @brief Get the number of agents in a notification map.
 *
 * @param  map   [in]  Notification map
 * @param  size  [out] Number of agents with notifications
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_notif_map_size(nixl_capi_notif_map_t map, size_t *size);

/**
 * @brief Get the agent name at an index of a notification map.
 *
 * @param  map         [in]  Notification map
 * @param  index       [in]  Index of the agent
 * @param  agent_name  [out] Agent name, valid while the map entry exists
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_notif_map_get_agent_at(nixl_capi_notif_map_t map, size_t index, const char **agent_name);

/**
 * @brief Get the number of notifications from an agent in a notification map.
 *
 * @param  map         [in]  Notification map
 * @param  agent_name  [in]  Name of the sending agent
 * @param  size        [out] Number of notifications from that agent
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_notif_map_get_notifs_size(nixl_capi_notif_map_t map,
                                    const char *agent_name,
                                    size_t *size);

/**
 * @brief Get a notification from an agent in a notification map.
 *
 * @param  map         [in]  Notification map
 * @param  agent_name  [in]  Name of the sending agent
 * @param  index       [in]  Index of the notification
 * @param  data        [out] Message bytes, valid while the map entry exists
 * @param  len         [out] Byte count
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_notif_map_get_notif(nixl_capi_notif_map_t map,
                              const char *agent_name,
                              size_t index,
                              const void **data,
                              size_t *len);

/**
 * @brief Remove all notifications from a notification map.
 *
 * @param  map  [in] Notification map to clear
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_notif_map_clear(nixl_capi_notif_map_t map);

// Query response list functions

/**
 * @brief Create an empty query response list.
 *
 * @param  list  [out] Created list, destroyed with nixl_capi_destroy_query_resp_list()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_query_resp_list(nixl_capi_query_resp_list_t *list);

/**
 * @brief Destroy a query response list.
 *
 * @param  list  [in] Query response list to destroy
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_query_resp_list(nixl_capi_query_resp_list_t list);

/**
 * @brief Get the number of responses in a query response list.
 *
 * @param  list  [in]  Query response list
 * @param  size  [out] Number of responses
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_query_resp_list_size(nixl_capi_query_resp_list_t list, size_t *size);

/**
 * @brief Check whether the response at an index has a value.
 *
 * @param  list       [in]  Query response list
 * @param  index      [in]  Index of the response
 * @param  has_value  [out] Whether the response has a value
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_query_resp_list_has_value(nixl_capi_query_resp_list_t list,
                                    size_t index,
                                    bool *has_value);

/**
 * @brief Get the parameters of the response at an index.
 *
 * @param  list    [in]  Query response list
 * @param  index   [in]  Index of the response
 * @param  params  [out] Parameter map, destroyed with nixl_capi_destroy_params()
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_query_resp_list_get_params(nixl_capi_query_resp_list_t list,
                                     size_t index,
                                     nixl_capi_params_t *params);

/**
 * @brief Query memory described by a registration descriptor list.
 *
 * @param  agent     [in]     Agent to query
 * @param  descs     [in]     Registration descriptor list describing the memory
 * @param  resp      [in,out] Query response list receiving one response per descriptor
 * @param  opt_args  [in]     Optional arguments, may be NULL
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_query_mem(nixl_capi_agent_t agent,
                    nixl_capi_reg_dlist_t descs,
                    nixl_capi_query_resp_list_t resp,
                    nixl_capi_opt_args_t opt_args);

/**
 * @brief Prepare a memory view handle for local buffers.
 *
 * The caller owns @a mvh and must release it with @ref nixl_capi_release_mem_view
 * before @a agent is destroyed. @a descs may be destroyed once this returns.
 *
 * @param  agent     [in]  Agent the buffers are registered with
 * @param  descs     [in]  Descriptor list for the local buffers
 * @param  mvh       [out] Memory view handle
 * @param  opt_args  [in]  Optional arguments, carrying the backend hint
 * @return nixl_capi_status_t Error code if call was not successful
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_prep_mem_view_local(nixl_capi_agent_t agent,
                              nixl_capi_xfer_dlist_t descs,
                              nixl_capi_mem_view_t *mvh,
                              nixl_capi_opt_args_t opt_args);

/**
 * @brief Prepare a memory view handle for remote buffers.
 *
 * Ownership is as for @ref nixl_capi_prep_mem_view_local. A view can span peers,
 * so each descriptor in @a descs names the agent that owns it.
 *
 * @param  agent     [in]  Initiator agent
 * @param  descs     [in]  Descriptor list for the remote buffers
 * @param  mvh       [out] Memory view handle
 * @param  opt_args  [in]  Optional arguments, carrying the backend hint
 * @return nixl_capi_status_t Error code if call was not successful
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_prep_mem_view_remote(nixl_capi_agent_t agent,
                               nixl_capi_remote_dlist_t descs,
                               nixl_capi_mem_view_t *mvh,
                               nixl_capi_opt_args_t opt_args);

/**
 * @brief Create a descriptor list for remote buffers.
 *
 * The caller owns @a dlist and must destroy it with
 * @ref nixl_capi_destroy_remote_dlist.
 *
 * @param  mem_type  [in]  NIXL memory type of the descriptor list
 * @param  dlist     [out] Created descriptor list
 * @return nixl_capi_status_t Error code if call was not successful
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_create_remote_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_remote_dlist_t *dlist);

/**
 * @brief Destroy a remote descriptor list.
 *
 * @param  dlist  [in] Descriptor list to destroy
 * @return nixl_capi_status_t Error code if call was not successful
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_destroy_remote_dlist(nixl_capi_remote_dlist_t dlist);

/**
 * @brief Add a descriptor to a remote descriptor list.
 *
 * @param  dlist         [in] Descriptor list to add to
 * @param  addr          [in] Start of the remote buffer
 * @param  len           [in] Length of the remote buffer
 * @param  dev_id        [in] deviceID/BlockID/bufferID (remote ID)
 * @param  remote_agent  [in] Name of the agent that owns the buffer. NULL means
 *                            nixl_null_agent, a placeholder that keeps
 *                            descriptor indices aligned when only some peers
 *                            are addressed
 * @return nixl_capi_status_t Error code if call was not successful
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_remote_dlist_add_desc(nixl_capi_remote_dlist_t dlist,
                                uintptr_t addr,
                                size_t len,
                                uint64_t dev_id,
                                const char *remote_agent);

/**
 * @brief Release a memory view handle.
 *
 * @param  agent  [in] Agent that prepared @a mvh
 * @param  mvh    [in] Memory view handle to release
 * @return nixl_capi_status_t Error code if call was not successful. A handle
 *         prepared by a different agent is not reported, as the underlying C++
 *         call returns void and logs a warning
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_release_mem_view(nixl_capi_agent_t agent, nixl_capi_mem_view_t mvh);

// Telemetry structure for transfer requests
typedef struct nixl_capi_xfer_telemetry_s *nixl_capi_xfer_telemetry_t;

/**
 * @brief Get transfer telemetry data.
 *
 * @param  agent      [in]  Agent that created the request
 * @param  req_hndl   [in]  Transfer request to query
 * @param  telemetry  [out] Caller-provided struct filled with the telemetry
 * @return NIXL_CAPI_SUCCESS on success, error code otherwise
 */
NIXL_CAPI_EXPORT nixl_capi_status_t
nixl_capi_get_xfer_telemetry(nixl_capi_agent_t agent,
                             nixl_capi_xfer_req_t req_hndl,
                             nixl_capi_xfer_telemetry_t telemetry);

/**
 * @brief Report whether this library is the stub implementation.
 *
 * @return true if this is the stub library, false for the real implementation
 */
NIXL_CAPI_EXPORT bool
nixl_capi_is_stub(void);

#ifdef __cplusplus
}
#endif

#endif /* NIXL_SRC_API_C_NIXL_CAPI_H */
