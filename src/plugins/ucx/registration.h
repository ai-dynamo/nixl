// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_UCX_REGISTRATION_H
#define NIXL_SRC_PLUGINS_UCX_REGISTRATION_H

#include <type_traits>

namespace nixl::ucx {

template<typename Release> class RegistrationRollback {
public:
    explicit RegistrationRollback(Release &release) noexcept : release_(release) {
        static_assert(std::is_nothrow_invocable_v<Release>);
    }

    RegistrationRollback(const RegistrationRollback &) = delete;
    RegistrationRollback &
    operator=(const RegistrationRollback &) = delete;

    ~RegistrationRollback() {
        if (active_) {
            release_();
        }
    }

    void
    commit() noexcept {
        active_ = false;
    }

private:
    Release &release_;
    bool active_ = true;
};

// Acquire transfers ownership only on success; prepare may fail or throw.
template<typename Acquire, typename Prepare, typename Release>
[[nodiscard]] bool
registrationTransaction(Acquire &&acquire, Prepare &&prepare, Release &&release) {
    if (!acquire()) {
        return false;
    }
    RegistrationRollback rollback(release);
    if (!prepare()) {
        return false;
    }
    rollback.commit();
    return true;
}

} // namespace nixl::ucx

#endif
