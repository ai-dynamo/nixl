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
#include <map>
#include <iostream>
#include "nixl.h"
#include "nixl_descriptors.h"
#include "mem_section.h"
#include "backend/backend_engine.h"
#include "nixl_types.h"
#include "serdes/serdes.h"

/*** Class nixlMemSection implementation ***/

// It's pure virtual, but base also class needs a destructor due to its members.
nixlMemSection::~nixlMemSection () {}

backend_set_t* nixlMemSection::queryBackends (const nixl_mem_t &mem) {
    if (mem<DRAM_SEG || mem>FILE_SEG)
        return nullptr;
    else
        return &memToBackend[mem];
}

nixl_status_t nixlMemSection::populate (const nixl_xfer_dlist_t &query,
                                        nixlBackendEngine* backend,
                                        nixl_meta_dlist_t &resp) const {

    if (query.getType() != resp.getType())
        return NIXL_ERR_INVALID_PARAM;
    // 1-to-1 mapping cannot hold
    if (query.isSorted() != resp.isSorted())
        return NIXL_ERR_INVALID_PARAM;

    section_key_t sec_key = std::make_pair(query.getType(), backend);
    auto it = sectionMap.find(sec_key);
    if (it==sectionMap.end())
        return NIXL_ERR_NOT_FOUND;

    nixlBasicDesc *p;
    nixl_sec_dlist_t* base = it->second;
    resp.resize(query.descCount());

    if (!base->isSorted()) {
        int count = 0;
        for (int i=0; i<query.descCount(); ++i)
            for (const auto & elm : *base)
                if (elm.covers(query[i])){
                    p = &resp[i];
                    *p = query[i];
                    resp[i].metadataP = elm.metadataP;
                    count++;
                    break;
                }

        if (query.descCount()==count) {
            return NIXL_SUCCESS;
        } else {
            resp.clear();
            return NIXL_ERR_UNKNOWN;
        }
    } else {
        const nixlBasicDesc *q;
        bool found, q_sorted = query.isSorted();

        if (q_sorted) {
            int s_index, q_index, size;
            const nixlSectionDesc *s;

            size = base->descCount();
            s_index = 0;
            q_index = 0;

            while (q_index<query.descCount()){
                s = &(*base)[s_index];
                q = &query[q_index];
                if (s->covers(*q)) {
                    p = &resp[q_index];
                    *p = *q;
                    resp[q_index].metadataP = s->metadataP;
                    q_index++;
                } else {
                    s_index++;
                    // TODO: add early termination if already (*q < *s),
                    // but s was not properly covering q
                    if (s_index==size) {
                        resp.clear();
                        return NIXL_ERR_UNKNOWN;
                    }
                }
            }

            // To be added only in debug mode
            // resp.verifySorted();
            return NIXL_SUCCESS;

        } else {
            int last_found = 0;
            for (int i=0; i<query.descCount(); ++i) {
                found = false;
                q = &query[i];
                auto itr = std::lower_bound(base->begin() + last_found,
                                            base->end(), *q);

                // Same start address case
                if (itr != base->end()){
                    if (itr->covers(*q)) {
                        found = true;
                    }
                }

                // query starts starts later, try previous entry
                if ((!found) && (itr != base->begin())){
                    itr = std::prev(itr , 1);
                    if (itr->covers(*q)) {
                        found = true;
                    }
                }

                if (found) {
                    p = &resp[i];
                    *p = *q;
                    resp[i].metadataP = itr->metadataP;
                } else {
                    resp.clear();
                    return NIXL_ERR_UNKNOWN;
                }
            }
            return NIXL_SUCCESS;
        }
    }
}

/*** Class nixlLocalSection implementation ***/

// Calls into backend engine to register the memories in the desc list
nixl_status_t nixlLocalSection::addDescList (const nixl_reg_dlist_t &mem_elms,
                                             nixlBackendEngine* backend,
                                             nixl_sec_dlist_t &remote_self) {

    if (!backend)
        return NIXL_ERR_INVALID_PARAM;
    // Find the MetaDesc list, or add it to the map
    nixl_mem_t     nixl_mem     = mem_elms.getType();
    section_key_t  sec_key      = std::make_pair(nixl_mem, backend);

    auto it = sectionMap.find(sec_key);
    if (it==sectionMap.end()) { // New desc list
        sectionMap[sec_key] = new nixl_sec_dlist_t(nixl_mem, true);
        memToBackend[nixl_mem].insert(backend);
    }
    nixl_sec_dlist_t *target = sectionMap[sec_key];

    // Add entries to the target list
    nixlSectionDesc local_sec, self_sec;
    nixlBasicDesc *lp = &local_sec;
    nixlBasicDesc *rp = &self_sec;
    nixl_status_t ret;

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

        target->addDesc(local_sec);

        if (backend->supportsLocal()) {
            *rp = *lp;
            remote_self.addDesc(self_sec);
        }
    }

    // Abort in case of error
    if (ret != NIXL_SUCCESS) {
        for (int j = 0; j < i; ++j) {
            int index = target->getIndex(mem_elms[j]);

            if (backend->supportsLocal()) {
                int self_index = remote_self.getIndex(mem_elms[j]);
                // Should never be negative, as we just added it in previous loop
                if (self_index >= 0 && remote_self[self_index].metadataP != (*target)[index].metadataP)
                    backend->unloadMD(remote_self[self_index].metadataP);
            }
            backend->deregisterMem((*target)[index].metadataP);
            target->remDesc(index);
        }
        remote_self.clear();
    }
    return ret;
}

nixl_status_t nixlLocalSection::remDescList (const nixl_reg_dlist_t &mem_elms,
                                             nixlBackendEngine *backend) {
    if (!backend)
        return NIXL_ERR_INVALID_PARAM;
    nixl_mem_t     nixl_mem     = mem_elms.getType();
    section_key_t sec_key = std::make_pair(nixl_mem, backend);
    auto it = sectionMap.find(sec_key);
    if (it==sectionMap.end())
        return NIXL_ERR_NOT_FOUND;
    nixl_sec_dlist_t *target = it->second;

    // First check if the mem_elms are present in the list,
    // don't deregister anything in case any is missing.
    for (auto & elm : mem_elms) {
        int index = target->getIndex(elm);
        if (index < 0)
            return NIXL_ERR_NOT_FOUND;
    }

    for (auto & elm : mem_elms) {
        int index = target->getIndex(elm);
        // Already checked, elm should always be found. Can add a check in debug mode.
        backend->deregisterMem((*target)[index].metadataP);
        target->remDesc(index);
    }

    if (target->descCount()==0) {
        delete target;
        sectionMap.erase(sec_key);
        memToBackend[nixl_mem].erase(backend);
    }

    return NIXL_SUCCESS;
}

namespace {
nixl_status_t serializeSections(nixlSerDes* serializer,
                                const section_map_t &sections) {
    nixl_status_t ret;

    size_t seg_count = sections.size();
    ret = serializer->addBuf("nixlSecElms", &seg_count, sizeof(seg_count));
    if (ret) return ret;

    for (const auto &[sec_key, dlist] : sections) {
        nixlBackendEngine* eng = sec_key.second;
        if (!eng->supportsRemote())
            continue;

        ret = serializer->addStr("bknd", eng->getType());
        if (ret) return ret;
        ret = dlist->serialize(serializer);
        if (ret) return ret;
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
    nixl_mem_t nixl_mem = mem_elms.getType();
    nixl_status_t ret = NIXL_SUCCESS;
    section_map_t mem_elms_to_serialize;

    // If there are no descriptors to serialize, just serialize empty list of sections
    if (mem_elms.descCount() == 0)
        return serializeSections(serializer, mem_elms_to_serialize);

    // TODO: consider concatenating 2 serializers instead of using mem_elms_to_serialize
    for (const auto &backend : backends) {
        section_key_t sec_key = std::make_pair(nixl_mem, backend);
        auto it = sectionMap.find(sec_key);
        if (it == sectionMap.end())
            continue;

        // TODO: consider section_map_t to be a map of unique_ptr or instance of nixl_meta_dlist_t.
        //       This will avoid the need to delete the nixl_sec_dlist_t instances.
        const nixl_sec_dlist_t *base = it->second;
        nixl_sec_dlist_t *resp = new nixl_sec_dlist_t(nixl_mem, mem_elms.isSorted());
        for (const auto &desc : mem_elms) {
            int index = base->getIndex(desc);
            if (index < 0) {
                ret = NIXL_ERR_NOT_FOUND;
                break;
            }
            resp->addDesc((*base)[index]);
        }
        if (ret != NIXL_SUCCESS)
            break;
        mem_elms_to_serialize.emplace(sec_key, resp);
    }

    if (ret == NIXL_SUCCESS)
        ret = serializeSections(serializer, mem_elms_to_serialize);

    for (auto &[sec_key, m_desc] : mem_elms_to_serialize)
        delete m_desc;
    return ret;
}

nixlLocalSection::~nixlLocalSection() {
    for (auto &[sec_key, dlist] : sectionMap) {
        nixlBackendEngine* eng = sec_key.second;
        for (auto & elm : *dlist)
            eng->deregisterMem(elm.metadataP);
        delete dlist;
    }
    // nixlMemSection destructor will clean up the rest
}

/*** Class nixlRemoteSection implementation ***/

nixlRemoteSection::nixlRemoteSection (const std::string &agent_name) {
    this->agentName = agent_name;
}

nixl_status_t nixlRemoteSection::addDescList (
                                 const nixl_reg_dlist_t& mem_elms,
                                 nixlBackendEngine* backend) {
    if (!backend->supportsRemote())
        return NIXL_ERR_UNKNOWN;

    // Less checks than LocalSection, as it's private and called by loadRemoteData
    // In RemoteSection, if we support updates, value for a key gets overwritten
    // Without it, its corrupt data, we keep the last option without raising an error
    nixl_mem_t nixl_mem   = mem_elms.getType();
    section_key_t sec_key = std::make_pair(nixl_mem, backend);
    if (sectionMap.count(sec_key) == 0)
        sectionMap[sec_key] = new nixl_sec_dlist_t(nixl_mem, true);
    memToBackend[nixl_mem].insert(backend); // Fine to overwrite, it's a set
    nixl_sec_dlist_t *target = sectionMap[sec_key];


    // Add entries to the target list.
    nixlSectionDesc out;
    nixlBasicDesc *p = &out;
    nixl_status_t ret;
    for (int i=0; i<mem_elms.descCount(); ++i) {
        // TODO: Can add overlap checks (erroneous)
        int idx = target->getIndex(mem_elms[i]);
        if (idx < 0) {
            ret = backend->loadRemoteMD(mem_elms[i], nixl_mem, agentName, out.metadataP);
            // In case of errors, no need to remove the previous entries
            // Agent will delete the full object.
            if (ret<0)
                return ret;
            *p = mem_elms[i]; // Copy the basic desc part
            out.metaBlob = mem_elms[i].metaInfo;
            target->addDesc(out);
        } else {
            const nixl_blob_t &prev_meta_info = (*target)[idx].metaBlob;
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

nixl_status_t nixlRemoteSection::loadLocalData (
                                 const nixl_sec_dlist_t& mem_elms,
                                 nixlBackendEngine* backend) {

    if (mem_elms.descCount()==0) // Shouldn't happen
        return NIXL_ERR_UNKNOWN;

    nixl_mem_t     nixl_mem     = mem_elms.getType();
    section_key_t sec_key = std::make_pair(nixl_mem, backend);

    if (sectionMap.count(sec_key) == 0)
        sectionMap[sec_key] = new nixl_sec_dlist_t(nixl_mem, true);
    memToBackend[nixl_mem].insert(backend); // Fine to overwrite, it's a set
    nixl_sec_dlist_t *target = sectionMap[sec_key];

    for (auto & elm: mem_elms)
        target->addDesc(elm);

    return NIXL_SUCCESS;
}

nixlRemoteSection::~nixlRemoteSection() {
    for (auto &[sec_key, dlist] : sectionMap) {
        nixlBackendEngine* eng = sec_key.second;
        for (auto & elm : *dlist)
            eng->unloadMD(elm.metadataP);
        delete dlist;
    }
    // nixlMemSection destructor will clean up the rest
}
