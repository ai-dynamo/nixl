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
    public:
        static constexpr bool kDefaultUseProgThread = false;
        static constexpr bool kDefaultUseListenThread = false;
        static constexpr int kDefaultListenPort = 0;
        static constexpr nixl_thread_sync_t kDefaultSyncMode =
            nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT;
        static constexpr bool kDefaultCaptureTelemetry = false;
        static constexpr uint64_t kDefaultPthrDelayUs = 0;
        static constexpr uint64_t kDefaultLthrDelayUs = 100000;
        static constexpr std::chrono::microseconds kDefaultEtcdWatchTimeout =
            std::chrono::microseconds(5000000);

        /** @var Enable progress thread */
        bool     useProgThread = kDefaultUseProgThread;
        /** @var Enable listener thread */
        bool     useListenThread = kDefaultUseListenThread;
        /** @var Port for listener thread to use */
        int      listenPort = kDefaultListenPort;
        /** @var synchronization mode for multi-threaded environment execution */
        nixl_thread_sync_t syncMode = kDefaultSyncMode;
        /** @var Capture telemetry info regardless of environment variables*/
        bool captureTelemetry = kDefaultCaptureTelemetry;

        /**
         * @var Progress thread event waiting timeout.
         *      Defines a delay between periodic main loop iterations that progress every worker
         *      unconditionally of any pending event signals.
         */
        uint64_t pthrDelay = kDefaultPthrDelayUs;
        /**
         * @var Listener thread frequency knob (in us)
         *      Listener thread sleeps in a similar way to progress thread, desrcibed previously.
         *      These will be combined into a unified NIXL Thread API in a future version.
         */
        uint64_t lthrDelay = kDefaultLthrDelayUs;

        /**
         * @var ETCD watch timeout in microseconds
         *      Timeout for waiting for metadata changes when watching etcd keys.
         */
        std::chrono::microseconds etcdWatchTimeout = kDefaultEtcdWatchTimeout;

        /**
         * @brief  Default constructor.
         */
        nixlAgentConfig() = default;

        /**
         * @brief  Agent configuration constructor for enabling various features.
         * @param use_prog_thread    flag to determine use of progress thread
         * @param use_listen_thread  Optional flag to determine use of listener thread
         * @param port               Optional port for listener thread to listen on
         * @param sync_mode          Optional Thread synchronization mode
         * @param num_workers        Optional number of shared workers per backend
         * @param pthr_delay_us      Optional delay for pthread in us
         * @param lthr_delay_us      Optional delay for listener thread in us
         * @param capture_telemetry  Optional flag to enable telemetry capture
         * @param etcd_watch_timeout Optional timeout for etcd watch operations in microseconds
         */
        nixlAgentConfig(const bool use_prog_thread,
                        const bool use_listen_thread = kDefaultUseListenThread,
                        const int port = kDefaultListenPort,
                        nixl_thread_sync_t sync_mode = kDefaultSyncMode,
                        unsigned int num_workers = 1,
                        const uint64_t pthr_delay_us = kDefaultPthrDelayUs,
                        const uint64_t lthr_delay_us = kDefaultLthrDelayUs,
                        const bool capture_telemetry = kDefaultCaptureTelemetry,
                        const std::chrono::microseconds &etcd_watch_timeout =
                            kDefaultEtcdWatchTimeout)
            : useProgThread(use_prog_thread),
              useListenThread(use_listen_thread),
              listenPort(port),
              syncMode(sync_mode),
              captureTelemetry(capture_telemetry),
              pthrDelay(pthr_delay_us),
              lthrDelay(lthr_delay_us),
              etcdWatchTimeout(etcd_watch_timeout) {}

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

};

#endif
