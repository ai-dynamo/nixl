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

/*
 * Minimal nixl C API example: version check, agent creation, plugin
 * listing, DRAM memory registration, and local metadata retrieval.
 */

#include <nixl_capi.h>

#include <stdio.h>
#include <stdlib.h>

#define BUF_SIZE 4096

/* Report a failing call and exit; nixl_capi_last_error_message() carries the
 * message of the C++ exception (if any) behind the failure. */
static void
die(const char *what, nixl_capi_status_t status) {
    fprintf(stderr,
            "%s failed: %s (%s)\n",
            what,
            nixl_capi_status_string(status),
            nixl_capi_last_error_message());
    exit(1);
}

#define MUST(call)                           \
    do {                                     \
        nixl_capi_status_t status_ = (call); \
        if (status_ != NIXL_CAPI_SUCCESS) {  \
            die(#call, status_);             \
        }                                    \
    } while (0)

int
main(void) {
    /* Verify the loaded library speaks the API this program was built for. */
    int major, minor, patch;
    nixl_capi_get_version(&major, &minor, &patch);
    printf("nixl C API %d.%d.%d\n", major, minor, patch);
    if (major != NIXL_CAPI_API_VERSION_MAJOR) {
        fprintf(stderr, "version mismatch: built against major %d\n", NIXL_CAPI_API_VERSION_MAJOR);
        return 1;
    }

    nixl_capi_agent_t agent;
    MUST(nixl_capi_create_agent("example-agent", &agent));

    /* List the transfer backend plugins nixl found. */
    nixl_capi_string_list_t plugins;
    MUST(nixl_capi_get_available_plugins(agent, &plugins));
    size_t n;
    MUST(nixl_capi_string_list_size(plugins, &n));
    printf("%zu plugin(s):\n", n);
    for (size_t i = 0; i < n; i++) {
        const char *name;
        MUST(nixl_capi_string_list_get(plugins, i, &name));
        printf("  %s\n", name);
    }
    if (n == 0) {
        fprintf(stderr, "no plugins available; cannot register memory\n");
        return 1;
    }

    /* Create a backend for the first plugin, using its default parameters. */
    const char *plugin;
    MUST(nixl_capi_string_list_get(plugins, 0, &plugin));
    nixl_capi_mem_list_t mems;
    nixl_capi_params_t params;
    MUST(nixl_capi_get_plugin_params(agent, plugin, &mems, &params));
    nixl_capi_backend_t backend;
    MUST(nixl_capi_create_backend(agent, plugin, params, &backend));
    printf("created backend for plugin %s\n", plugin);
    MUST(nixl_capi_destroy_mem_list(mems));
    MUST(nixl_capi_destroy_params(params));
    MUST(nixl_capi_destroy_string_list(plugins));

    /* Register a DRAM buffer. Registration makes the memory usable as a
     * transfer source or target. */
    void *buf = malloc(BUF_SIZE);
    if (buf == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    nixl_capi_reg_dlist_t reg_list;
    MUST(nixl_capi_create_reg_dlist(NIXL_CAPI_MEM_DRAM, &reg_list));
    MUST(nixl_capi_reg_dlist_add_desc(reg_list, (uintptr_t)buf, BUF_SIZE, 0, NULL, 0));
    MUST(nixl_capi_register_mem(agent, reg_list, NULL));

    /* Fetch this agent's metadata blob; a peer loads it with
     * nixl_capi_load_remote_md() to enable transfers. */
    void *md;
    size_t md_len;
    MUST(nixl_capi_get_local_md(agent, &md, &md_len));
    printf("local metadata: %zu bytes\n", md_len);
    nixl_capi_mem_free(md);

    /* Tear down in reverse order of creation. */
    MUST(nixl_capi_deregister_mem(agent, reg_list, NULL));
    MUST(nixl_capi_destroy_reg_dlist(reg_list));
    free(buf);
    MUST(nixl_capi_destroy_backend(backend));
    MUST(nixl_capi_destroy_agent(agent));

    printf("done\n");
    return 0;
}
