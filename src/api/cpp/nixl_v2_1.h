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
/**
 * @file nixl.h (NVIDIA Inference X Library)
 * @brief These are NIXL Core APIs for applications
 */
#ifndef _NIXL_H
#define _NIXL_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <functional>

/* notes:
    * Connections are handled implicitly by nixl backend
    * Memory registrations are handled implicitly by nixl backend
*/

namespace nixl {

    /* 
        a nixl pod is a group of nixl agents that can transfer memory between each other.
        the vision is of centralized meta data controller.
        nixl agents communicate meta data only via the centralized controller.
        the meta data messages are defined in the pod::msg namespace.
        the Client class is intantiated in the Agent class - this is the Agent interface with the centralized controller.
    */
    namespace pod {
        namespace obj {
            enum class Type {
                AGENT,
                MEMSPACE
            };

            struct Agent {
                std::string ip;
                uint16_t    port;
            };

            struct MemSpace {
                struct Agent agent;
                uint32_t     key;
            };
        } // namespace obj

        namespace msg {
            enum class Opcode {
                ADD_AGENT,
                DEL_MEMSPACE
            };

            struct AddAgent {
                struct obj::Agent agent;
            };

            struct DelAgent {
                struct obj::Agent agent;
            };

            struct AddMemSpace {
                struct obj::MemSpace           memSpace;
                uint64_t                       len;
                std::vector<struct obj::Agent> accessList;
            };

            struct DelMemSpace {
                struct obj::MemSpace memSpace;
            };

            struct Msg {
                Opcode                 opcode;
                union {
                    struct AddAgent    addAgent;
                    struct DelAgent    delAgent;
                    struct AddMemSpace addMemSpace;
                    struct DelMemSpace delMemSpace;
                };
            };
        } // namespace msg

        class Client {
            public:
                using RcvMsgCb = std::function<void(const struct Msg &msg)>;

                Client(const struct AgentID &serverID,
                       const RcvMsgCb       &rcvMsgCb,
                       const bool           autoPollRcvMsg);
                ~Client();

                void sendMsg(const struct Msg &msg);

                void pollRcvMsg();
        };
    } // namespace pod

    namespace mem {
        enum class Type {
            LOCAL_DRAM,
            LOCAL_VRAM,
            LOCAL_BLK,
            LOCAL_OBJ,
            LOCAL_FILE,
            REMOTE_AGENT
        };

        class Space {
        public:
            struct Local {
                Type     type;
                int      dev_id;
                uint64_t len;
            };

            Space(const struct Local &local);
            Space(const struct pod::obj::MemSpace &podMemSpace);

            // generate a key for granting access to a local mem space for remote agents
            uint32_t genPodMemSpaceKeyForLocalMemSpace();

            ~Space();
        };
    } // namespace mem
 
    struct MemXferType {
        mem::Type dst;
        mem::Type src;
    };

    namespace container {
        using BackendType = std::string;

        struct BackendTypeParams {
            std::vector<MemXferType> supportedMemXferTypes;
        };

        void getAvailBackendTypes(std::vector<BackendType> &backendTypes);

        void getBackendTypeParams(const struct BackendTypeParams &params);
    }; // namespace container

    class Agent {
        enum class PostState {
            NOT_POSTED,
            POSTED,
            COMPLETED
        };

        enum class PostCompletionStatus {
            OK,
            ERROR
        };

        class MemXfer {
        public:
            struct MemXferBuf {
                class mem::Space space;
                size_t           ofst;
                size_t           len;
            };

            PostState            postState;
            PostCompletionStatus postCompletionStatus;

            // dst and src must have the same length
            MemXfer(const std::vector<struct MemXferBuf> &dst,
                    const std::vector<struct MemXferBuf> &src);
            
            MemXfer(const std::vector<struct MemXferBuf> &dst,
                    const std::vector<struct MemXferBuf> &src,
                    const std::vector<int>               &dstIndices,
                    const std::vector<int>               &srcIndices);

            ~MemXfer();
        };

        class Signal {
        public:
            PostState            postSndState;
            PostCompletionStatus postSndCompletionStatus;

            Signal(const std::string &signalData);

            ~Signal();
        };

        class pod::Client podClient;

        using RcvSignalCb = std::function<void(class Signal &signal)>;

        Agent(const struct pod::AgentID                 &id,
              const std::vector<MemXferType>            &memXferTypes,
              const std::vector<container::BackendType> backendTypes,
              const struct pod::AgentID                 &podClientServerID,
              const pod::Client::RcvMsgCb               &podClientRcvMsgCb,
              const bool                                podClientAutoPoll,
              const RcvSignalCb                         rcvSignalCb,
              const bool                                autoPoll)
              : podClient(podClientServerID, podClientRcvMsgCb, podClientAutoPoll) {}

        ~Agent();

        void postMemXfer(class MemXfer &memXfer);

        void getCompletedMemXfers(std::vector<class MemXfer> &memXfers);

        void postSndSignal(const class Signal &signal);

        void getCompletedSndSignals(std::vector<class Signal> &signals);

        void flushPosts();

        void poll();
    };
} // namespace nixl

#endif