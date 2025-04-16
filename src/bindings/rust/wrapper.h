#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status codes for our C API
typedef enum {
  NIXL_CAPI_SUCCESS = 0,
  NIXL_CAPI_ERROR_INVALID_PARAM = -1,
  NIXL_CAPI_ERROR_BACKEND = -2,
  NIXL_CAPI_ERROR_INVALID_STATE = -3,
  NIXL_CAPI_IN_PROG = 1,
} nixl_capi_status_t;

// Memory types enum (matching nixl's memory types)
typedef enum {
  NIXL_CAPI_MEM_UNKNOWN = 0,
  NIXL_CAPI_MEM_DRAM = 1,
  NIXL_CAPI_MEM_GPU = 2,
  // Add other memory types as needed
} nixl_capi_mem_type_t;

struct nixl_capi_agent_s;
struct nixl_capi_params_s;
struct nixl_capi_mem_list_s;
struct nixl_capi_string_list_s;
struct nixl_capi_backend_s;
struct nixl_capi_opt_args_s;
struct nixl_capi_param_iter_s;
struct nixl_capi_xfer_dlist_s;
struct nixl_capi_reg_dlist_s;
struct nixl_capi_xfer_req_s;
struct nixl_capi_notif_map_s;

// Opaque handle types for C++ objects
typedef struct nixl_capi_agent_s* nixl_capi_agent_t;
typedef struct nixl_capi_params_s* nixl_capi_params_t;
typedef struct nixl_capi_mem_list_s* nixl_capi_mem_list_t;
typedef struct nixl_capi_string_list_s* nixl_capi_string_list_t;
typedef struct nixl_capi_backend_s* nixl_capi_backend_t;
typedef struct nixl_capi_opt_args_s* nixl_capi_opt_args_t;
typedef struct nixl_capi_param_iter_s* nixl_capi_param_iter_t;
typedef struct nixl_capi_xfer_dlist_s* nixl_capi_xfer_dlist_t;
typedef struct nixl_capi_reg_dlist_s* nixl_capi_reg_dlist_t;
typedef struct nixl_capi_xfer_req_s* nixl_capi_xfer_req_t;
typedef struct nixl_capi_notif_map_s* nixl_capi_notif_map_t;

// Core API functions
nixl_capi_status_t nixl_capi_create_agent(const char* name, nixl_capi_agent_t* agent);

nixl_capi_status_t nixl_capi_destroy_agent(nixl_capi_agent_t agent);

// Get local metadata as a byte array
nixl_capi_status_t nixl_capi_get_local_md(nixl_capi_agent_t agent, void** data, size_t* len);

// Load remote metadata from a byte array
nixl_capi_status_t nixl_capi_load_remote_md(nixl_capi_agent_t agent, const void* data, size_t len, char** agent_name);

// Invalidate remote agent metadata
nixl_capi_status_t nixl_capi_invalidate_remote_md(nixl_capi_agent_t agent, const char* remote_agent);

// Plugin and parameter functions
nixl_capi_status_t nixl_capi_get_available_plugins(nixl_capi_agent_t agent, nixl_capi_string_list_t* plugins);
nixl_capi_status_t nixl_capi_destroy_string_list(nixl_capi_string_list_t list);
nixl_capi_status_t nixl_capi_string_list_size(nixl_capi_string_list_t list, size_t* size);
nixl_capi_status_t nixl_capi_string_list_get(nixl_capi_string_list_t list, size_t index, const char** str);

nixl_capi_status_t nixl_capi_get_plugin_params(
    nixl_capi_agent_t agent, const char* plugin_name, nixl_capi_mem_list_t* mems, nixl_capi_params_t* params);

nixl_capi_status_t nixl_capi_destroy_mem_list(nixl_capi_mem_list_t list);
nixl_capi_status_t nixl_capi_destroy_params(nixl_capi_params_t params);

// Backend creation and management
nixl_capi_status_t nixl_capi_create_backend(
    nixl_capi_agent_t agent, const char* plugin_name, nixl_capi_params_t params, nixl_capi_backend_t* backend);
nixl_capi_status_t nixl_capi_destroy_backend(nixl_capi_backend_t backend);

// Get backend parameters after initialization
nixl_capi_status_t nixl_capi_get_backend_params(
    nixl_capi_agent_t agent, nixl_capi_backend_t backend, nixl_capi_mem_list_t* mems, nixl_capi_params_t* params);

// Optional arguments management
nixl_capi_status_t nixl_capi_create_opt_args(nixl_capi_opt_args_t* args);
nixl_capi_status_t nixl_capi_destroy_opt_args(nixl_capi_opt_args_t args);
nixl_capi_status_t nixl_capi_opt_args_add_backend(nixl_capi_opt_args_t args, nixl_capi_backend_t backend);

// OptArgs notification and merge control
nixl_capi_status_t nixl_capi_opt_args_set_notif_msg(nixl_capi_opt_args_t args, const void* data, size_t len);
nixl_capi_status_t nixl_capi_opt_args_get_notif_msg(nixl_capi_opt_args_t args, void** data, size_t* len);
nixl_capi_status_t nixl_capi_opt_args_set_has_notif(nixl_capi_opt_args_t args, bool has_notif);
nixl_capi_status_t nixl_capi_opt_args_get_has_notif(nixl_capi_opt_args_t args, bool* has_notif);
nixl_capi_status_t nixl_capi_opt_args_set_skip_desc_merge(nixl_capi_opt_args_t args, bool skip_merge);
nixl_capi_status_t nixl_capi_opt_args_get_skip_desc_merge(nixl_capi_opt_args_t args, bool* skip_merge);

// Parameter access functions
nixl_capi_status_t nixl_capi_params_is_empty(nixl_capi_params_t params, bool* is_empty);
nixl_capi_status_t nixl_capi_params_create_iterator(nixl_capi_params_t params, nixl_capi_param_iter_t* iter);
nixl_capi_status_t nixl_capi_params_iterator_next(
    nixl_capi_param_iter_t iter, const char** key, const char** value, bool* has_next);
nixl_capi_status_t nixl_capi_params_destroy_iterator(nixl_capi_param_iter_t iter);

// Memory list access functions
nixl_capi_status_t nixl_capi_mem_list_is_empty(nixl_capi_mem_list_t list, bool* is_empty);
nixl_capi_status_t nixl_capi_mem_list_size(nixl_capi_mem_list_t list, size_t* size);
nixl_capi_status_t nixl_capi_mem_list_get(nixl_capi_mem_list_t list, size_t index, nixl_capi_mem_type_t* mem_type);
nixl_capi_status_t nixl_capi_mem_type_to_string(nixl_capi_mem_type_t mem_type, const char** str);

// Memory registration functions
nixl_capi_status_t nixl_capi_register_mem(
    nixl_capi_agent_t agent, nixl_capi_reg_dlist_t dlist, nixl_capi_opt_args_t opt_args);

nixl_capi_status_t nixl_capi_deregister_mem(
    nixl_capi_agent_t agent, nixl_capi_reg_dlist_t dlist, nixl_capi_opt_args_t opt_args);

// Notification functions
nixl_capi_status_t nixl_capi_get_notifs(
    nixl_capi_agent_t agent, nixl_capi_notif_map_t notif_map, nixl_capi_opt_args_t opt_args);

nixl_capi_status_t nixl_capi_create_notif_map(nixl_capi_notif_map_t* notif_map);

nixl_capi_status_t nixl_capi_destroy_notif_map(nixl_capi_notif_map_t notif_map);

// Transfer request functions
typedef enum {
  NIXL_CAPI_XFER_OP_READ = 0,
  NIXL_CAPI_XFER_OP_WRITE = 1,
} nixl_capi_xfer_op_t;

nixl_capi_status_t nixl_capi_create_xfer_req(
    nixl_capi_agent_t agent, nixl_capi_xfer_op_t operation, nixl_capi_xfer_dlist_t local_descs,
    nixl_capi_xfer_dlist_t remote_descs, const char* remote_agent, nixl_capi_xfer_req_t* req_hndl,
    nixl_capi_opt_args_t opt_args);

nixl_capi_status_t nixl_capi_post_xfer_req(
    nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl, nixl_capi_opt_args_t opt_args);

nixl_capi_status_t nixl_capi_get_xfer_status(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl);

nixl_capi_status_t nixl_capi_release_xfer_req(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req);

nixl_capi_status_t nixl_capi_destroy_xfer_req(nixl_capi_xfer_req_t req);

// Descriptor list functions
nixl_capi_status_t nixl_capi_create_xfer_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_xfer_dlist_t* dlist);
nixl_capi_status_t nixl_capi_destroy_xfer_dlist(nixl_capi_xfer_dlist_t dlist);
nixl_capi_status_t nixl_capi_xfer_dlist_add_desc(
    nixl_capi_xfer_dlist_t dlist, uintptr_t addr, size_t len, uint32_t dev_id);
nixl_capi_status_t nixl_capi_xfer_dlist_len(nixl_capi_xfer_dlist_t dlist, size_t* len);
nixl_capi_status_t nixl_capi_xfer_dlist_has_overlaps(nixl_capi_xfer_dlist_t dlist, bool* has_overlaps);
nixl_capi_status_t nixl_capi_xfer_dlist_clear(nixl_capi_xfer_dlist_t dlist);
nixl_capi_status_t nixl_capi_xfer_dlist_resize(nixl_capi_xfer_dlist_t dlist, size_t new_size);

nixl_capi_status_t nixl_capi_create_reg_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_reg_dlist_t* dlist);
nixl_capi_status_t nixl_capi_destroy_reg_dlist(nixl_capi_reg_dlist_t dlist);
nixl_capi_status_t nixl_capi_reg_dlist_add_desc(
    nixl_capi_reg_dlist_t dlist, uintptr_t addr, size_t len, uint32_t dev_id);
nixl_capi_status_t nixl_capi_reg_dlist_len(nixl_capi_reg_dlist_t dlist, size_t* len);
nixl_capi_status_t nixl_capi_reg_dlist_has_overlaps(nixl_capi_reg_dlist_t dlist, bool* has_overlaps);
nixl_capi_status_t nixl_capi_reg_dlist_clear(nixl_capi_reg_dlist_t dlist);
nixl_capi_status_t nixl_capi_reg_dlist_resize(nixl_capi_reg_dlist_t dlist, size_t new_size);

nixl_capi_status_t nixl_capi_notif_map_size(nixl_capi_notif_map_t map, size_t* size);

nixl_capi_status_t nixl_capi_notif_map_get_agent_at(nixl_capi_notif_map_t map, size_t index, const char** agent_name);

nixl_capi_status_t nixl_capi_notif_map_get_notifs_size(nixl_capi_notif_map_t map, const char* agent_name, size_t* size);

nixl_capi_status_t nixl_capi_notif_map_get_notif(
    nixl_capi_notif_map_t map, const char* agent_name, size_t index, const void** data, size_t* len);

#ifdef __cplusplus
}
#endif
