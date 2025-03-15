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
#include "nixl.h"
#include "nixl_descriptors.h"
#include "mem_section.h"
#include "backend/backend_engine.h"
#include "serdes/serdes.h"

/*** Class nixlMemSection implementation ***/

// It's pure virtual, but base also class needs a destructor due to its members.
nixlMemSection::~nixlMemSection () {}

nixl_status_t nixlMemSection::populate (const nixl_xfer_dlist_t &query,
                                        const nixl_backend_t &nixl_backend,
                                        nixl_meta_dlist_t &resp) const {

    if (query.getType() != resp.getType())
        return NIXL_ERR_INVALID_PARAM;
    section_key_t sec_key = std::make_pair(query.getType(), nixl_backend);
    auto it = sectionMap.find(sec_key);
    if (it==sectionMap.end())
        return NIXL_ERR_NOT_FOUND;
    else
        return it->second->populate(query, resp);
}

/*** Class nixlLocalSection implementation ***/

nixl_reg_dlist_t nixlLocalSection::getStringDesc (
                             const nixlBackendEngine* backend,
                             const nixl_meta_dlist_t &d_list) const {
    nixl_status_t ret;
    nixlBlobDesc element;
    nixlBasicDesc *p = &element;
    nixl_reg_dlist_t output_desclist(d_list.getType(),
                                     d_list.isUnifiedAddr(),
                                     d_list.isSorted());

    // The string information of each registered block are updated by
    // required serialized metadata provided by the backend
    for (int i=0; i<d_list.descCount(); ++i) {
        *p = (nixlBasicDesc) d_list[i];
        ret = backend->getPublicData(d_list[i].metadataP, element.metaInfo);
        if(ret != NIXL_SUCCESS){
            //something has gone wrong
            output_desclist.clear();
            return output_desclist;
        }

        output_desclist.addDesc(element);
    }
    return output_desclist;
}

nixl_status_t nixlLocalSection::addBackendHandler (nixlBackendEngine* backend) {
    if (!backend)
        return NIXL_ERR_INVALID_PARAM;
    // Agent has already checked for not being the same type of backend
    backendToEngineMap[backend->getType()] = backend;
    return NIXL_SUCCESS;
}

// Calls into backend engine to register the memories in the desc list
nixl_status_t nixlLocalSection::addDescList (const nixl_reg_dlist_t &mem_elms,
                                             nixlBackendEngine* backend,
                                             nixl_meta_dlist_t &remote_self) {

    if (!backend)
        return NIXL_ERR_INVALID_PARAM;
    // Find the MetaDesc list, or add it to the map
    nixl_mem_t     nixl_mem     = mem_elms.getType();
    nixl_backend_t nixl_backend = backend->getType();
    section_key_t  sec_key      = std::make_pair(nixl_mem, nixl_backend);

    if ((nixl_mem == FILE_SEG) && mem_elms.isUnifiedAddr())
        return NIXL_ERR_INVALID_PARAM;

    auto it = sectionMap.find(sec_key);
    if (it==sectionMap.end()) { // New desc list
        sectionMap[sec_key] = new nixl_meta_dlist_t(
                                  nixl_mem, mem_elms.isUnifiedAddr(), true);
        memToBackendMap[nixl_mem].insert(nixl_backend);
    }
    nixl_meta_dlist_t *target = sectionMap[sec_key];

    // Add entries to the target list
    nixlMetaDesc local_meta, self_meta;
    nixlBasicDesc *lp = &local_meta;
    nixlBasicDesc *rp = &self_meta;
    nixl_status_t ret1, ret2=NIXL_SUCCESS;
    int index;

    for (int i=0; i<mem_elms.descCount(); ++i) {
        // TODO: For now trusting the user, but there can be a more checks mode
        //       where we find overlaps and split the memories or warn the user
        ret1 = backend->registerMem(mem_elms[i], nixl_mem, local_meta.metadataP);

        if ((ret1==NIXL_SUCCESS) && backend->supportsLocal()) {
            ret2 = backend->loadLocalMD(local_meta.metadataP, self_meta.metadataP);
        }

        if ((ret1!=NIXL_SUCCESS) || (ret2!=NIXL_SUCCESS)) {
            for (int j=0; j<i; ++j) {
                index = target->getIndex(mem_elms[j]);
                backend->deregisterMem
                    ((*(const nixl_meta_dlist_t*)target)[index].metadataP);
                target->remDesc(index);
            }
            remote_self.clear();
            if (ret1!=NIXL_SUCCESS)
                return ret1;
            else
                return ret2;
        }

        *lp = mem_elms[i]; // Copy the basic desc part
        if ((nixl_mem == FILE_SEG) && (lp->len==0))
            lp->len = SIZE_MAX; // File has no range limit

        target->addDesc(local_meta);

        if (backend->supportsLocal()) {
            *rp = *lp;
            remote_self.addDesc(self_meta);
        }
    }
    return NIXL_SUCCESS;
}

// Per each nixlBasicDesc, the full region that got registered should be deregistered
nixl_status_t nixlLocalSection::remDescList (const nixl_meta_dlist_t &mem_elms,
                                             nixlBackendEngine *backend) {
    if (!backend)
        return NIXL_ERR_INVALID_PARAM;
    nixl_mem_t     nixl_mem     = mem_elms.getType();
    nixl_backend_t nixl_backend = backend->getType();
    section_key_t sec_key = std::make_pair(nixl_mem, nixl_backend);
    auto it = sectionMap.find(sec_key);
    if (it==sectionMap.end())
        return NIXL_ERR_NOT_FOUND;
    nixl_meta_dlist_t *target = it->second;

    for (auto & elm : mem_elms) {
        int index = target->getIndex(elm);
        // Errorful situation, not sure helpful to deregister the rest,
        // registering back what was deregistered is not meaningful.
        // Can be secured by going through all the list then deregister
        if (index<0)
            return NIXL_ERR_UNKNOWN;

        backend->deregisterMem
            ((*(const nixl_meta_dlist_t*)target)[index].metadataP);
        target->remDesc(index);
    }

    if (target->descCount()==0){
        delete target;
        sectionMap.erase(sec_key);
        memToBackendMap[nixl_mem].erase(nixl_backend);
    }

    return NIXL_SUCCESS;
}

nixlBackendEngine* nixlLocalSection::findQuery(
                       const nixl_xfer_dlist_t &query,
                       const nixl_mem_t &remote_nixl_mem,
                       const backend_set_t &remote_backends,
                       nixl_meta_dlist_t &resp) const {

    nixlBackendEngine* backend = nullptr;

    nixl_mem_t q_mem = query.getType();
    if (q_mem>FILE_SEG)
        return backend;

    const backend_set_t* backend_set = &memToBackendMap.at(q_mem);
    if (backend_set->empty())
        return backend;

    // Decision making based on supported local backends for this
    // memory type, supported remote backends and remote memory type
    // or here we loop through and find first local match. The more
    // complete option (overkill) is to try all possible scenarios and
    // see which populates on both side are successful and then decide

    for (auto & elm : *backend_set) {
        // If populate fails, it clears the resp before return
        if (populate(query, elm, resp) == NIXL_SUCCESS)
            return backendToEngineMap.at(elm);
    }
    return backend;
}

nixl_status_t nixlLocalSection::serialize(nixlSerDes* serializer) const {
    nixl_status_t ret;
    size_t seg_count = sectionMap.size();
    nixl_backend_t nixl_backend;
    nixlBackendEngine* eng;

    ret = serializer->addBuf("nixlSecElms", &seg_count, sizeof(seg_count));
    if (ret) return ret;

    for (auto &seg : sectionMap) {
        nixl_backend = seg.first.second;
        eng = backendToEngineMap.at(nixl_backend);
        if (!eng->supportsRemote())
            continue;

        nixl_reg_dlist_t s_desc = getStringDesc(eng, *seg.second);
        ret = serializer->addStr("bknd", nixl_backend);
        if (ret) return ret;
        ret = s_desc.serialize(serializer);
        if (ret) return ret;
    }

    return NIXL_SUCCESS;
}

nixlLocalSection::~nixlLocalSection() {
    for (auto &seg : sectionMap)
        remDescList(*seg.second, backendToEngineMap[seg.first.second]);
}

/*** Class nixlRemoteSection implementation ***/

nixlRemoteSection::nixlRemoteSection (
                   const std::string &agent_name,
                   backend_map_t &engine_map) {
    this->agentName    = agent_name;
    backendToEngineMap = engine_map;
}

nixl_status_t nixlRemoteSection::addDescList (
                                 const nixl_reg_dlist_t& mem_elms,
                                 nixlBackendEngine* backend) {
    if (!backend->supportsRemote())
        return NIXL_ERR_UNKNOWN;

    // Less checks than LocalSection, as it's private and called by loadRemoteData
    // In RemoteSection, if we support updates, value for a key gets overwritten
    // Without it, its corrupt data, we keep the last option without raising an error
    nixl_mem_t     nixl_mem     = mem_elms.getType();
    nixl_backend_t nixl_backend = backend->getType();
    section_key_t sec_key = std::make_pair(nixl_mem, nixl_backend);
    if (sectionMap.count(sec_key) == 0)
        sectionMap[sec_key] = new nixl_meta_dlist_t(
                                  nixl_mem, mem_elms.isUnifiedAddr(), true);
    memToBackendMap[nixl_mem].insert(nixl_backend); // Fine to overwrite, it's a set
    nixl_meta_dlist_t *target = sectionMap[sec_key];


    // Add entries to the target list.
    nixlMetaDesc out;
    nixlBasicDesc *p = &out;
    nixl_status_t ret;
    for (int i=0; i<mem_elms.descCount(); ++i) {
        // TODO: remote might change the metadata, have to keep stringDesc to compare
        //       if we support partial updates. Also Can add overlap checks (erroneous)
        if (target->getIndex((const nixlBasicDesc) mem_elms[i]) < 0) {
            ret = backend->loadRemoteMD(mem_elms[i], nixl_mem, agentName, out.metadataP);
            // In case of errors, no need to remove the previous entries
            // Agent will delete the full object.
            if (ret<0)
                return ret;
            *p = mem_elms[i]; // Copy the basic desc part
            target->addDesc(out);
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlRemoteSection::loadRemoteData (nixlSerDes* deserializer) {
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
        ret = addDescList(s_desc, backendToEngineMap[nixl_backend]);
        if (ret) return ret;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlRemoteSection::loadLocalData (
                                 const nixl_meta_dlist_t& mem_elms,
                                 nixlBackendEngine* backend) {

    if (mem_elms.descCount()==0) // Shouldn't happen
        return NIXL_ERR_UNKNOWN;

    nixl_mem_t     nixl_mem     = mem_elms.getType();
    nixl_backend_t nixl_backend = backend->getType();
    section_key_t sec_key = std::make_pair(nixl_mem, nixl_backend);

    if (sectionMap.count(sec_key) == 0)
        sectionMap[sec_key] = new nixl_meta_dlist_t(
                                  nixl_mem, mem_elms.isUnifiedAddr(), true);
    memToBackendMap[nixl_mem].insert(nixl_backend); // Fine to overwrite, it's a set
    nixl_meta_dlist_t *target = sectionMap[sec_key];

    for (auto & elm: mem_elms)
        target->addDesc(elm);

    if(backendToEngineMap.count(nixl_backend)==0)
        backendToEngineMap[nixl_backend]=backend;

    return NIXL_SUCCESS;
}

nixlRemoteSection::~nixlRemoteSection() {
    nixl_backend_t nixl_backend;
    nixl_meta_dlist_t *m_desc;

    for (auto &seg : sectionMap) {
        nixl_backend = seg.first.second;
        m_desc = seg.second;
        for (auto & elm : *m_desc)
            backendToEngineMap[nixl_backend]->unloadMD(elm.metadataP);
        delete m_desc;
    }
    // nixlMemSection destructor will clean up the rest
}
