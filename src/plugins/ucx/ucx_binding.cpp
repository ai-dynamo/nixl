// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "ucx_binding.h"
#include "common/configuration.h"
#include "common/nixl_log.h"

#include <dlfcn.h>
#include <string>
#include <ucp/api/ucp.h>

namespace nixl::ucx {
bool
validateBinding() {
    constexpr const char *expected_var = "NIXL_UCX_EXPECTED_SONAME";
    Dl_info info{};
    const std::string symbol_path =
        (dladdr(reinterpret_cast<void *>(ucp_get_version_string), &info) && info.dli_fname) ?
        info.dli_fname :
        "<unknown>";
    NIXL_INFO << "NIXL UCX backend bound to UCX " << ucp_get_version_string() << " at "
              << symbol_path;
    const auto expected = nixl::config::getValueOptional<std::string>(expected_var);
    if (!expected || expected->empty() || symbol_path.find(*expected) != std::string::npos) {
        return true;
    }
    NIXL_ERROR << expected_var << "=" << *expected << " but NIXL UCX backend bound to "
               << symbol_path;
    return false;
}
} // namespace nixl::ucx
