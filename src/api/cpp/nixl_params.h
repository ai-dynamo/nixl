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

#include <string>
#include <cstdint>
#include "nixl_types.h"

/**
 * @class nixlAgentConfig
 * @brief Per Agent configuration information, such as if progress thread should be used.
 *        Other configs such as assigned IP/port or device access can be added.
 */
class nixlAgentConfig {
    private:

        /** @var Enable progress thread */
        bool     useProgThread;
        /** @var Enable listener thread */
        bool     useListenThread;
        /** @var Port for listener thread to use */
        int      listenPort;
        /** @var synchronization mode for multi-threaded environment execution */
        nixl_thread_sync_t syncMode;
        /** @var number of workers in the engine instance*/
        unsigned int numWorkers;

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
         * @var Listener thread frequency knob (in us)
         *      Listener thread sleeps in a similar way to progress thread, desrcibed previously.
         *      These will be combined into a unified NIXL Thread API in a future version.
         */
        uint64_t lthrDelay;


        /**
         * @brief  Agent configuration constructor for enabling various features.
         * @param use_prog_thread    flag to determine use of progress thread
         * @param use_listen_thread  flag to determine use of listener thread
         * @param port               specify port for listener thread to listen on
         * @param sync_mode          Thread synchronization mode
         * @param pthr_delay_us      Optional delay for pthread in us
         * @param lthr_delay_us      Optional delay for listener thread in us
         */
        nixlAgentConfig (const bool use_prog_thread,
                         const bool use_listen_thread=false,
                         const int port=0,
                         nixl_thread_sync_t sync_mode=nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT,
                         unsigned int num_workers = 1,
                         const uint64_t pthr_delay_us=0,
                         const uint64_t lthr_delay_us = 100000) :
                         useProgThread(use_prog_thread),
                         useListenThread(use_listen_thread),
                         listenPort(port),
                         syncMode(sync_mode),
                         numWorkers(num_workers),
                         pthrDelay(pthr_delay_us),
                         lthrDelay(lthr_delay_us) { }

        /**
         * @brief Copy constructor for nixlAgentConfig object
         *
         * @param cfg  nixlAgentConfig object
         */
        nixlAgentConfig (const nixlAgentConfig &cfg) = default;

        /**
         * @brief Default destructor for nixlAgentConfig
         */
        ~nixlAgentConfig () = default;

    friend class nixlAgent;
    friend class nixlAgentData;
};

#endif
