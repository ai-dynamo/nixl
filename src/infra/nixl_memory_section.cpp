/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <map>
#include <algorithm>
#include <iostream>
#include "nixl.h"
#include "nixl_descriptors.h"
#include "mem_section.h"
#include "backend/backend_engine.h"
#include "nixl_types.h"
#include "serdes/serdes.h"

/*** Class nixlMemSection implementation ***/

nixlSecDescList &
nixlMemSection::emplace(const nixl_mem_t nixl_mem, nixlBackendEngine *backend) {
    const section_key_t sec_key(nixl_mem, backend);
    const auto [it, inserted] = sectionMap.try_emplace(sec_key, sec_key.first);
    if (inserted) {
        memToBackend[sec_key.first].emplace(sec_key.second);
    }
    return it->second;
}

backend_set_t *
nixlMemSection::queryBackends(const nixl_mem_t mem) noexcept {
    if ((mem < DRAM_SEG) || (mem > FILE_SEG)) {
        return nullptr;
    }
    return &memToBackend[mem];
}

const backend_set_t *
nixlMemSection::queryBackends(const nixl_mem_t mem) const noexcept {
    if ((mem < DRAM_SEG) || (mem > FILE_SEG)) {
        return nullptr;
    }
    return &memToBackend[mem];
}

nixl_status_t nixlMemSection::populate (const nixl_xfer_dlist_t &query,
                                        nixlBackendEngine* backend,
                                        nixl_meta_dlist_t &resp) const {

    if ((query.getType() != resp.getType()) || (query.isEmpty())) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const section_key_t sec_key(query.getType(), backend);
    const auto it = sectionMap.find(sec_key);
    if (it == sectionMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    const nixlSecDescList &base = it->second;
    resp.resize(query.descCount());

    int size = base.descCount();
    int s_index = 0;

    // Use logN search for the first element, instead of linear search
    s_index = base.getCoveringIndex(query[0]);
    if (s_index < 0) {
        resp.clear();
        return NIXL_ERR_UNKNOWN;
    }
    static_cast<nixlBasicDesc &>(resp[0]) = query[0];
    resp[0].metadataP = base[s_index].metadataP;

    // Walk forward for non-decreasing elements; logN search on temporal disorder
    for (int i = 1; i < query.descCount(); ++i) {
        if (__builtin_expect(query[i] < query[i - 1], 0)) {
            // Disorder in the list, resolve this element using logN search
            s_index = base.getCoveringIndex(query[i]);
            if (__builtin_expect(s_index < 0, 0)) {
                resp.clear();
                return NIXL_ERR_UNKNOWN;
            }
        } else {
            while (s_index < size && !base[s_index].covers(query[i]))
                ++s_index;
            if (__builtin_expect(s_index == size, 0)) {
                resp.clear();
                return NIXL_ERR_UNKNOWN;
            }
        }

        static_cast<nixlBasicDesc &>(resp[i]) = query[i];
        resp[i].metadataP = base[s_index].metadataP;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMemSection::addElement(const nixlRemoteDesc &query,
                           nixlBackendEngine *backend,
                           nixl_remote_meta_dlist_t &resp) const {
    const section_key_t sec_key{VRAM_SEG, backend};
    const auto it = sectionMap.find(sec_key);
    if (it == sectionMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    const nixlSecDescList &base = it->second;

    const int s_index = base.getCoveringIndex(query);
    if (s_index < 0) {
        return NIXL_ERR_UNKNOWN;
    }

    resp.addDesc({query.addr, query.len, query.devId, base[s_index].metadataP});
    return NIXL_SUCCESS;
}

/*** Class nixlLocalSection implementation ***/

// Calls into backend engine to register the memories in the desc list
nixl_status_t
nixlLocalSection::addDescList(const nixl_reg_dlist_t &mem_elms,
                              nixlBackendEngine *backend,
                              nixlSecDescList &remote_self) {

    if (!backend) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // Find the MetaDesc list, or add it to the map
    const nixl_mem_t nixl_mem = mem_elms.getType();

    nixlSecDescList &target = emplace(nixl_mem, backend);

    // Add entries to the target list
    nixlSectionDesc local_sec, self_sec;
    nixlBasicDesc *lp = &local_sec;
    nixlBasicDesc *rp = &self_sec;
    nixl_status_t ret = NIXL_SUCCESS;

    int i;
    for (i = 0; i < mem_elms.descCount(); ++i) {
        // TODO: For now trusting the user, but there can be a more checks mode
        //       where we find overlaps and split the memories or warn the user
        ret = backend->registerMem(mem_elms[i], nixl_mem, local_sec.metadataP);
        if (ret != NIXL_SUCCESS)
            break;

        if (backend->supportsLocal()) {
            ret = backend->loadLocalMD(local_sec.metadataP, self_sec.metadataP);
            if (ret != NIXL_SUCCESS) {
                backend->deregisterMem(local_sec.metadataP);
                break;
            }
        }
        if (backend->supportsRemote()) {
            ret = backend->getPublicData(local_sec.metadataP, local_sec.metaBlob);
            if (ret != NIXL_SUCCESS) {
                // A backend might use the same object for both initiator/target
                // side of a transfer, so no need for unloadMD in that case.
                if (backend->supportsLocal() && self_sec.metadataP != local_sec.metadataP)
                    backend->unloadMD(self_sec.metadataP);
                backend->deregisterMem(local_sec.metadataP);
                break;
            }
        }

        *lp = mem_elms[i]; // Copy the basic desc part
        if (((nixl_mem == BLK_SEG) || (nixl_mem == OBJ_SEG) ||
             (nixl_mem == FILE_SEG)) && (lp->len==0))
            lp->len = SIZE_MAX; // File has no range limit

        target.addDesc(local_sec);

        if (backend->supportsLocal()) {
            *rp = *lp;
            remote_self.addDesc(self_sec);
        }
    }

    // Abort in case of error
    if (ret != NIXL_SUCCESS) {
        for (int j = 0; j < i; ++j) {
            int index = target.getIndex(mem_elms[j]);

            if (backend->supportsLocal()) {
                int self_index = remote_self.getIndex(mem_elms[j]);
                // Should never be negative, as we just added it in previous loop
                if (self_index >= 0 && remote_self[self_index].metadataP != target[index].metadataP)
                    backend->unloadMD(remote_self[self_index].metadataP);
            }
            backend->deregisterMem(target[index].metadataP);
            target.remDesc(index);
        }
        remote_self.clear();
    }
    return ret;
}

nixl_status_t nixlLocalSection::remDescList (const nixl_reg_dlist_t &mem_elms,
                                             nixlBackendEngine *backend) {
    if (!backend) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const nixl_mem_t nixl_mem = mem_elms.getType();
    const section_key_t sec_key(nixl_mem, backend);
    const auto it = sectionMap.find(sec_key);
    if (it == sectionMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlSecDescList &target = it->second;

    // First check if the mem_elms are present in the list,
    // don't deregister anything in case any is missing.
    for (auto & elm : mem_elms) {
        int index = target.getIndex(elm);
        if (index < 0)
            return NIXL_ERR_NOT_FOUND;
    }

    for (auto & elm : mem_elms) {
        int index = target.getIndex(elm);
        // Already checked, elm should always be found. Can add a check in debug mode.
        backend->deregisterMem(target[index].metadataP);
        target.remDesc(index);
    }

    if (target.isEmpty()) {
        sectionMap.erase(sec_key); // Invalidates target.
        // Note that sectionMap contains one entry per memory type and backend pair,
        // wherefore each backend can only have been inserted once into a memory type
        // specific memToBackend and we can now erase it when the sectionMap entry
        // for that memory type and backend is erased above.
        memToBackend[nixl_mem].erase(backend);
    }

    return NIXL_SUCCESS;
}

namespace {
nixl_status_t
serializeSections(nixlSerDes *serializer, const section_map_t &sectionMap) {
    size_t seg_count = std::count_if(sectionMap.begin(), sectionMap.end(), [](const auto &pair) {
        const section_key_t &sec_key = pair.first;
        return sec_key.second->supportsRemote();
    });

    auto ret = serializer->addBuf("nixlSecElms", &seg_count, sizeof(seg_count));
    if (ret) {
        return ret;
    }

    for (const auto &[sec_key, dlist] : sectionMap) {
        nixlBackendEngine *eng = sec_key.second;
        if (!eng->supportsRemote()) {
            continue;
        }

        ret = serializer->addStr("bknd", eng->getType());
        if (ret) {
            return ret;
        }

        ret = dlist.serialize(serializer);
        if (ret) {
            return ret;
        }
    }

    return NIXL_SUCCESS;
}
};

nixl_status_t nixlLocalSection::serialize(nixlSerDes* serializer) const {
    return serializeSections(serializer, sectionMap);
}

nixl_status_t nixlLocalSection::serializePartial(nixlSerDes* serializer,
                                                 const backend_set_t &backends,
                                                 const nixl_reg_dlist_t &mem_elms) const {
    const nixl_mem_t nixl_mem = mem_elms.getType();
    nixl_status_t ret = NIXL_SUCCESS;
    section_map_t mem_elms_to_serialize;

    // If there are no descriptors to serialize, just serialize empty list of sections
    if (mem_elms.isEmpty()) {
        return serializeSections(serializer, mem_elms_to_serialize);
    }

    // TODO: consider concatenating 2 serializers instead of using mem_elms_to_serialize
    for (const auto &backend : backends) {
        const section_key_t sec_key(nixl_mem, backend);
        const auto it = sectionMap.find(sec_key);
        if (it == sectionMap.end()) {
            continue;
        }

        const nixlSecDescList &base = it->second;
        nixlSecDescList resp(nixl_mem);
        for (const auto &desc : mem_elms) {
            int index = base.getIndex(desc);
            if (index < 0) {
                ret = NIXL_ERR_NOT_FOUND;
                break;
            }
            resp.addDesc(base[index]);
        }
        if (ret != NIXL_SUCCESS) {
            break;
        }
        mem_elms_to_serialize.try_emplace(sec_key, std::move(resp));
    }

    if (ret == NIXL_SUCCESS) {
        ret = serializeSections(serializer, mem_elms_to_serialize);
    }

    return ret;
}

nixlLocalSection::~nixlLocalSection() {
    for (auto &[sec_key, dlist] : sectionMap) {
        nixlBackendEngine* eng = sec_key.second;
        for (auto &elm : dlist) {
            eng->deregisterMem(elm.metadataP);
        }
    }
}

/*** Class nixlRemoteSection implementation ***/

nixlRemoteSection::nixlRemoteSection(std::string agent_name) noexcept
    : agentName(std::move(agent_name)) {}

nixl_status_t nixlRemoteSection::addDescList (
                                 const nixl_reg_dlist_t& mem_elms,
                                 nixlBackendEngine* backend) {
    if (!backend->supportsRemote()) {
        return NIXL_ERR_UNKNOWN;
    }

    // Fewer checks than LocalSection, as it's private and called by loadRemoteData
    // In RemoteSection, if we support updates, value for a key gets overwritten
    // Without it, it's corrupt data, we keep the last option without raising an error
    const nixl_mem_t nixl_mem = mem_elms.getType();

    nixlSecDescList &target = emplace(nixl_mem, backend);

    // Add entries to the target list.
    nixlSectionDesc out;
    nixlBasicDesc *p = &out;
    nixl_status_t ret;
    for (int i=0; i<mem_elms.descCount(); ++i) {
        // TODO: Can add overlap checks (erroneous)
        int idx = target.getIndex(mem_elms[i]);
        if (idx < 0) {
            ret = backend->loadRemoteMD(mem_elms[i], nixl_mem, agentName, out.metadataP);
            // In case of errors, no need to remove the previous entries
            // Agent will delete the full object.
            if (ret<0)
                return ret;
            *p = mem_elms[i]; // Copy the basic desc part
            out.metaBlob = mem_elms[i].metaInfo;
            target.addDesc(out);
        } else {
            const nixl_blob_t &prev_meta_info = target[idx].metaBlob;
            // TODO: Support metadata updates
            if (prev_meta_info != mem_elms[i].metaInfo)
                return NIXL_ERR_NOT_ALLOWED;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlRemoteSection::loadRemoteData (nixlSerDes* deserializer,
                                                 backend_map_t &backendToEngineMap) {
    nixl_status_t ret;
    size_t seg_count;
    nixl_backend_t nixl_backend;

    ret = deserializer->getBuf("nixlSecElms", &seg_count, sizeof(seg_count));
    if (ret) return ret;

    for (size_t i=0; i<seg_count; ++i) {
        // In case of errors, no need to remove the previous entries
        // Agent will delete the full object.
        nixl_backend = deserializer->getStr("bknd");
        if (nixl_backend.size()==0)
            return NIXL_ERR_INVALID_PARAM;
        nixl_reg_dlist_t s_desc(deserializer);
        if (s_desc.descCount()==0) // can be used for entry removal in future
            return NIXL_ERR_NOT_FOUND;
        if (backendToEngineMap.count(nixl_backend) != 0) {
            ret = addDescList(s_desc, backendToEngineMap[nixl_backend]);
            if (ret) return ret;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRemoteSection::loadLocalData(const nixlSecDescList &mem_elms, nixlBackendEngine *backend) {

    if (mem_elms.isEmpty()) { // Shouldn't happen
        return NIXL_ERR_UNKNOWN;
    }

    const nixl_mem_t nixl_mem = mem_elms.getType();

    nixlSecDescList &target = emplace(nixl_mem, backend);

    for (auto &elm : mem_elms) {
        target.addDesc(elm);
    }
    return NIXL_SUCCESS;
}

nixlRemoteSection::~nixlRemoteSection() {
    for (auto &[sec_key, dlist] : sectionMap) {
        nixlBackendEngine* eng = sec_key.second;
        for (auto &elm : dlist) {
            eng->unloadMD(elm.metadataP);
        }
    }
}
