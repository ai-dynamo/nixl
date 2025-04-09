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
#ifndef _NIXL_PARAMS_H
#define _NIXL_PARAMS_H

#include <cstdint>
#include <string>

#include "nixl_types.h"

/**
 * @class nixlAgentConfig
 * @brief Per Agent configuration information, such as if progress thread should be used.
 *        Other configs such as assigned IP/port or device access can be added.
 */
class nixlAgentConfig {
private:
    /** @var Enable progress thread */
    bool useProgThread;

public:
    /**
     * @var Progress thread frequency knob (in us)
     *      The progress thread is calling sched_yield to avoid blocking a core
     *      If pthrDelay time is less than sched_yield time - option has no effect
     *      Otherwise pthread will be calling sched_yield until the specified
     *      amount of time has past.
     */
    uint64_t pthrDelay;

    /**
     * @brief  Agent configuration constructor. Important configs such as
     *         useProgThread must be given and can't be changed.
     * @param use_prog_thread  flag to determine use of progress thread
     * @param pthr_delay_us    Optional delay for pthread in us
     */
    nixlAgentConfig(const bool use_prog_thread, const uint64_t pthr_delay_us = 0)
    {
        this->useProgThread = use_prog_thread;
        this->pthrDelay = pthr_delay_us;
    }

    /**
     * @brief Copy constructor for nixlAgentConfig object
     *
     * @param cfg  nixlAgentConfig object
     */
    nixlAgentConfig(const nixlAgentConfig& cfg) = default;

    /**
     * @brief Default destructor for nixlAgentConfig
     */
    ~nixlAgentConfig() = default;

    friend class nixlAgent;
};

#endif
