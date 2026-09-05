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

/* Runtime smoke test for the nixl C API from a pure C consumer. */

#include <nixl_capi.h>

#include <stdio.h>
#include <string.h>

static int failures;

static void
check(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static void
check_status(nixl_capi_status_t status, const char *what) {
    if (status != NIXL_CAPI_SUCCESS) {
        fprintf(stderr,
                "FAIL: %s: %s (%s)\n",
                what,
                nixl_capi_status_string(status),
                nixl_capi_last_error_message());
        failures++;
    }
}

int
main(void) {
    /* Version handshake. */
    int major = -1, minor = -1, patch = -1;
    nixl_capi_get_version(&major, &minor, &patch);
    check(major == NIXL_CAPI_API_VERSION_MAJOR, "library major version matches header");
    check(minor >= 0 && patch >= 0, "minor/patch are non-negative");
    nixl_capi_get_version(NULL, NULL, NULL); /* NULL out-pointers are allowed */

    /* Status strings are static, non-empty names. */
    check(strcmp(nixl_capi_status_string(NIXL_CAPI_SUCCESS), "NIXL_CAPI_SUCCESS") == 0,
          "status_string(SUCCESS)");
    check(nixl_capi_status_string(NIXL_CAPI_ERROR_INVALID_PARAM) != NULL &&
              nixl_capi_status_string(NIXL_CAPI_ERROR_INVALID_PARAM)[0] != '\0',
          "status_string(INVALID_PARAM) is non-empty");
    check(nixl_capi_status_string((nixl_capi_status_t)-9999) != NULL,
          "status_string on unknown code is non-NULL");

    /* No error has happened yet on this thread. */
    check(nixl_capi_last_error_message() != NULL, "last_error_message is non-NULL");

    /* Agent lifecycle with the default configuration. */
    nixl_capi_agent_t agent = NULL;
    check_status(nixl_capi_create_agent("c-smoke-agent", &agent), "create_agent");
    check(agent != NULL, "create_agent yields a handle");

    if (agent != NULL) {
        /* Local metadata requires a backend that supports remote operations;
         * create a UCX backend when the plugin is available, else skip. */
        int have_ucx = 0;
        nixl_capi_string_list_t plugins = NULL;
        check_status(nixl_capi_get_available_plugins(agent, &plugins), "get_available_plugins");
        if (plugins != NULL) {
            size_t n = 0;
            check_status(nixl_capi_string_list_size(plugins, &n), "string_list_size");
            for (size_t i = 0; i < n; i++) {
                const char *name = NULL;
                check_status(nixl_capi_string_list_get(plugins, i, &name), "string_list_get");
                if (name != NULL && strcmp(name, "UCX") == 0) {
                    have_ucx = 1;
                }
            }
            check_status(nixl_capi_destroy_string_list(plugins), "destroy_string_list");
        }

        nixl_capi_backend_t backend = NULL;
        if (have_ucx) {
            nixl_capi_params_t params = NULL;
            check_status(nixl_capi_create_params(&params), "create_params");
            check_status(nixl_capi_create_backend(agent, "UCX", params, &backend),
                         "create_backend(UCX)");
            check_status(nixl_capi_destroy_params(params), "destroy_params");

            void *md = NULL;
            size_t md_len = 0;
            check_status(nixl_capi_get_local_md(agent, &md, &md_len), "get_local_md");
            check(md != NULL && md_len > 0, "local metadata is non-empty");
            nixl_capi_mem_free(md);

            check_status(nixl_capi_destroy_backend(backend), "destroy_backend");
        } else {
            fprintf(stderr, "note: UCX plugin not available, skipping metadata checks\n");
        }

        check_status(nixl_capi_destroy_agent(agent), "destroy_agent");
    }

    /* Destroy functions accept NULL. */
    check_status(nixl_capi_destroy_agent(NULL), "destroy_agent(NULL)");
    check_status(nixl_capi_destroy_string_list(NULL), "destroy_string_list(NULL)");
    check_status(nixl_capi_destroy_opt_args(NULL), "destroy_opt_args(NULL)");
    nixl_capi_mem_free(NULL);

    if (failures != 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("nixl C API smoke test passed\n");
    return 0;
}
