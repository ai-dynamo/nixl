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
#ifndef __BACKEND_AUX_H_
#define __BACKEND_AUX_H_

#include <mutex>
#include <string>
#include "nixl_types.h"
#include "nixl_descriptors.h"
#include "common/nixl_time.h"

// Might be removed to be decided by backend, or changed to high
// level direction or so.
typedef std::vector<std::pair<std::string, std::string>> notif_list_t;


struct nixlBackendOptionalArgs {
    // During postXfer, user might ask for a notification if supported
    nixl_blob_t notifMsg;
    bool        hasNotif = false;
};

using nixl_opt_b_args_t = nixlBackendOptionalArgs;


// A base class to point to backend initialization data
// User doesn't know about fields such as local_agent but can access it
// after the backend is initialized by agent. If we needed to make it private
// from the user, we should make nixlBackendEngine/nixlAgent friend classes.
class nixlBackendInitParams {
    public:
        std::string       localAgent;

        nixl_backend_t    type;
        nixl_b_params_t*  customParams;

        bool              enableProgTh;
        nixlTime::us_t    pthrDelay;
        nixl_thread_sync_t syncMode;
};

// Pure virtual class to have a common pointer type
class nixlBackendReqH {
public:
    nixlBackendReqH() { }
    virtual ~nixlBackendReqH() { }
};

// Pure virtual class to have a common pointer type for different backendMD.
class nixlBackendMD {
    protected:
        bool isPrivateMD;

    public:
        nixlBackendMD(bool isPrivate){
            isPrivateMD = isPrivate;
        }

        virtual ~nixlBackendMD(){
        }
};

// Each backend can have different connection requirement
// This class would include the required information to make
// a connection to a remote node. Note that local information
// is passed during the constructor and through BackendInitParams
class nixlBackendConnMD {
  public:
    // And some other details
    std::string dstIpAddress;
    uint16_t    dstPort;
};

// A pointer required to a metadata object for backends next to each BasicDesc
class nixlMetaDesc : public nixlBasicDesc {
  public:
        // To be able to point to any object
        nixlBackendMD* metadataP;

        // Reuse parent constructor without the metadata pointer
        using nixlBasicDesc::nixlBasicDesc;

        nixlMetaDesc() : nixlBasicDesc() { metadataP = nullptr; }

        // No serializer or deserializer, using parent not to expose the metadata

        inline friend bool operator==(const nixlMetaDesc &lhs, const nixlMetaDesc &rhs) {
            return (((nixlBasicDesc)lhs == (nixlBasicDesc)rhs) &&
                          (lhs.metadataP == rhs.metadataP));
        }

        inline void print(const std::string &suffix) const {
            nixlBasicDesc::print(", Backend ptr val: " +
                                 std::to_string((uintptr_t)metadataP) + suffix);
        }
};

typedef nixlDescList<nixlMetaDesc> nixl_meta_dlist_t;

#endif
