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
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <iostream>
#include "nixl.h"
#include "nixl_descriptors.h"
#include "mem_section.h"
#include "backend/backend_aux.h"
#include "serdes/serdes.h"
#include "common/nixl_log.h"

/*** Class nixlBasicDesc implementation ***/

// No Virtual function in nixlBasicDesc class or its children, as we want
// each object to just have the members during serialization.

nixlBasicDesc::nixlBasicDesc(const uintptr_t &addr,
                             const size_t &len,
                             const uint64_t &dev_id) {
    this->addr  = addr;
    this->len   = len;
    this->devId = dev_id;
}

nixlBasicDesc::nixlBasicDesc(const nixl_blob_t &blob) {
    if (blob.size()==sizeof(nixlBasicDesc)) {
        blob.copy(reinterpret_cast<char*>(this), sizeof(nixlBasicDesc));
    } else { // Error indicator, not possible by descList deserializer call
        addr  = 0;
        len   = 0;
        devId = 0;
    }
}

bool nixlBasicDesc::operator<(const nixlBasicDesc &desc) const {
    if (devId != desc.devId)
        return (devId < desc.devId);
    else if (addr != desc.addr)
        return (addr < desc.addr);
    else
        return (len < desc.len);
}

bool operator==(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs) {
    return ((lhs.addr  == rhs.addr ) &&
            (lhs.len   == rhs.len  ) &&
            (lhs.devId == rhs.devId));
}

bool operator!=(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs) {
    return !(lhs==rhs);
}

bool nixlBasicDesc::covers (const nixlBasicDesc &query) const {
    if (devId == query.devId) {
        if ((addr <=  query.addr) &&
            (addr + len >= query.addr + query.len))
            return true;
    }
    return false;
}

bool nixlBasicDesc::overlaps (const nixlBasicDesc &query) const {
    if (devId != query.devId)
        return false;
    if ((addr + len <= query.addr) || (query.addr + query.len <= addr))
        return false;
    return true;
}

nixl_blob_t nixlBasicDesc::serialize() const {
    return std::string(reinterpret_cast<const char*>(this),
                       sizeof(nixlBasicDesc));
}

void nixlBasicDesc::print(const std::string &suffix) const {
    NIXL_INFO << "Desc (" << addr << ", " << len
              << ") from devID " << devId << suffix;
}


/*** Class nixlBlobDesc implementation ***/

nixlBlobDesc::nixlBlobDesc(const uintptr_t &addr,
                           const size_t &len,
                           const uint64_t &dev_id,
                           const nixl_blob_t &meta_info) :
                           nixlBasicDesc(addr, len, dev_id) {
    this->metaInfo = meta_info;
}

nixlBlobDesc::nixlBlobDesc(const nixlBasicDesc &desc,
                           const nixl_blob_t &meta_info) :
                           nixlBasicDesc(desc) {
    this->metaInfo = meta_info;
}

nixlBlobDesc::nixlBlobDesc(const nixl_blob_t &blob) {
    if (blob.size() > sizeof(nixlBasicDesc)) {
        const size_t meta_size = blob.size() - sizeof(nixlBasicDesc);
        metaInfo.resize(meta_size);
        blob.copy(reinterpret_cast<char*>(this), sizeof(nixlBasicDesc));
        blob.copy(reinterpret_cast<char*>(&metaInfo[0]),
                 meta_size, sizeof(nixlBasicDesc));
    } else if (blob.size() == sizeof(nixlBasicDesc)) {
        blob.copy(reinterpret_cast<char*>(this), sizeof(nixlBasicDesc));
    } else { // Error
        addr  = 0;
        len   = 0;
        devId = 0;
        metaInfo.clear();
    }
}

bool operator==(const nixlBlobDesc &lhs, const nixlBlobDesc &rhs) {
    return (((nixlBasicDesc)lhs == (nixlBasicDesc)rhs) &&
                  (lhs.metaInfo == rhs.metaInfo));
}

nixl_blob_t nixlBlobDesc::serialize() const {
    return nixlBasicDesc::serialize() + metaInfo;
}

void nixlBlobDesc::print(const std::string &suffix) const {
    nixlBasicDesc::print(", Metadata: " + metaInfo + suffix);
}

/*** Class nixlDescList implementation ***/

// The template is used to select from nixlBasicDesc/nixlMetaDesc/nixlBlobDesc
// There are no virtual functions, so the object is all data, no pointers.

template<class T> nixlDescList<T>::nixlDescList(const nixl_mem_t &type, const int &init_size) {
    static_assert (std::is_base_of<nixlBasicDesc, T>::value);
    this->type = type;
    this->descs.resize(init_size);
}

template <class T>
nixlDescList<T>::nixlDescList(nixlSerDes* deserializer) {
    size_t n_desc;
    std::string str;

    descs.clear();

    str = deserializer->getStr("nixlDList"); // Object type
    if (str.size()==0)
        return;

    // nixlMetaDesc and nixlSectionDesc should be internal and not be serialized
    if (std::is_same<nixlMetaDesc, T>::value || std::is_same<nixlSectionDesc, T>::value)
        return;

    if (deserializer->getBuf("t", &type, sizeof(type)))
        return;

    if (deserializer->getBuf("n", &n_desc, sizeof(n_desc)))
        return;

    if (std::is_same<nixlBasicDesc, T>::value) {
        // Contiguous in memory, so no need for per elm deserialization
        if (str!="nixlBDList")
            return;
        str = deserializer->getStr("");
        if (str.size()!= n_desc * sizeof(nixlBasicDesc))
            return;
        // If size is proper, deserializer cannot fail
        descs.resize(n_desc);
        str.copy(reinterpret_cast<char*>(descs.data()), str.size());

    } else if (std::is_same<nixlBlobDesc, T>::value) {
        if (str!="nixlSDList")
            return;
        for (size_t i=0; i<n_desc; ++i) {
            str = deserializer->getStr("");
            // If size is proper, deserializer cannot fail
            // Allowing empty strings, might change later
            if (str.size() < sizeof(nixlBasicDesc)) {
                descs.clear();
                return;
            }
            T elm(str);
            descs.push_back(elm);
        }
    } else {
        return; // Unknown type, error
    }
}

// Getter
template <class T>
inline const T& nixlDescList<T>::operator[](unsigned int index) const {
    // To be added only in debug mode
    // if (index >= descs.size())
    //     throw std::out_of_range("Index is out of range");
    return descs[index];
}

// Setter
template <class T>
inline T& nixlDescList<T>::operator[](unsigned int index) {
    assert(index < descs.size());
    return descs[index];
}

template <class T>
void nixlDescList<T>::addDesc (const T &desc) {
    descs.push_back(desc);
}

template <class T>
void nixlDescList<T>::remDesc (const int &index){
    if (((size_t) index >= descs.size()) || (index < 0))
        throw std::out_of_range("Index is out of range");
    descs.erase(descs.begin() + index);
}

template<class T>
void
nixlDescList<T>::resize(const size_t &count) {
    descs.resize(count);
}

template <class T>
nixlDescList<nixlBasicDesc> nixlDescList<T>::trim() const {

    if constexpr (std::is_same<nixlBasicDesc, T>::value) {
        return *this;
    } else {
        nixlDescList<nixlBasicDesc> trimmed(type);
        nixlBasicDesc* p;

        for (auto & elm: descs) {
            p = (nixlBasicDesc*) (&elm);
            trimmed.addDesc(*p);
        }

        // No failure scenario
        return trimmed;
    }
}

template <class T>
int nixlDescList<T>::getIndex(const nixlBasicDesc &query) const {
    auto itr = std::find(descs.begin(), descs.end(), query);
    if (itr == descs.end()) return NIXL_ERR_NOT_FOUND; // not found
    return itr - descs.begin();
}

template <class T>
nixl_status_t nixlDescList<T>::serialize(nixlSerDes* serializer) const {

    nixl_status_t ret;
    size_t n_desc = descs.size();

    // nixlMetaDesc should be internal and not be serialized
    if (std::is_same<nixlMetaDesc, T>::value)
        return NIXL_ERR_INVALID_PARAM;

    // For now very few descriptor types, if needed can add a name method to each
    // descriptor. std::string_view(typeid(T).name()) is compiler dependent
    if (std::is_same<nixlBasicDesc, T>::value)
        ret = serializer->addStr("nixlDList", "nixlBDList");
    // We serialize SectionDesc the same as BlobDesc so it will be deserialized as BlobDesc on the other side
    else if (std::is_same<nixlBlobDesc, T>::value || std::is_same<nixlSectionDesc, T>::value)
        ret = serializer->addStr("nixlDList", "nixlSDList");
    else
        return NIXL_ERR_INVALID_PARAM;

    if (ret) return ret;

    ret = serializer->addBuf("t", &type, sizeof(type));
    if (ret) return ret;

    ret = serializer->addBuf("n", &(n_desc), sizeof(n_desc));
    if (ret) return ret;

    if (n_desc==0)
        return NIXL_SUCCESS; // Unusual, but supporting it

    // Optimization for nixlBasicDesc,
    // contiguous in memory, so no need for per elm serialization
    if (std::is_same<nixlBasicDesc, T>::value) {
        ret = serializer->addStr("", std::string(
                                 reinterpret_cast<const char*>(descs.data()),
                                 n_desc * sizeof(nixlBasicDesc)));
        if (ret) return ret;
    } else { // already checked it can be only nixlBlobDesc or nixlSectionDesc
        for (auto & elm : descs) {
            ret = serializer->addStr("", elm.serialize());
            if (ret) return ret;
        }
    }

    return NIXL_SUCCESS;
}

template<>
nixl_status_t
nixlDescList<nixlRemoteMetaDesc>::serialize(nixlSerDes *serializer) const {
    return NIXL_ERR_NOT_SUPPORTED;
}

template <class T>
void nixlDescList<T>::print() const {
    std::cout << "DescList of mem type " << type << std::endl;
    for (auto & elm : descs) {
        elm.print("");
    }
}

template <class T>
bool operator==(const nixlDescList<T> &lhs, const nixlDescList<T> &rhs) {
    if ((lhs.getType() != rhs.getType()) || (lhs.descCount() != rhs.descCount())) return false;

    for (size_t i=0; i<lhs.descs.size(); ++i)
        if (lhs.descs[i] != rhs.descs[i])
            return false;
    return true;
}

// Since we implement a template class declared in a header files, this is necessary
template class nixlDescList<nixlBasicDesc>;
template class nixlDescList<nixlMetaDesc>;
template class nixlDescList<nixlBlobDesc>;
template class nixlDescList<nixlSectionDesc>;
template class nixlDescList<nixlRemoteDesc>;
template class nixlDescList<nixlRemoteMetaDesc>;

template bool operator==<nixlBasicDesc> (const nixlDescList<nixlBasicDesc> &lhs,
                                         const nixlDescList<nixlBasicDesc> &rhs);
template bool operator==<nixlMetaDesc>  (const nixlDescList<nixlMetaDesc> &lhs,
                                         const nixlDescList<nixlMetaDesc> &rhs);
template bool operator==<nixlBlobDesc>(const nixlDescList<nixlBlobDesc> &lhs,
                                       const nixlDescList<nixlBlobDesc> &rhs);
template bool operator==<nixlSectionDesc>(const nixlDescList<nixlSectionDesc> &lhs,
                                          const nixlDescList<nixlSectionDesc> &rhs);
template bool
operator== <nixlRemoteDesc>(const nixlDescList<nixlRemoteDesc> &lhs,
                            const nixlDescList<nixlRemoteDesc> &rhs);
template bool
operator== <nixlRemoteMetaDesc>(const nixlDescList<nixlRemoteMetaDesc> &lhs,
                                const nixlDescList<nixlRemoteMetaDesc> &rhs);

// nixlSecDescList keeps the elements sorted
void
nixlSecDescList::addDesc(const nixlSectionDesc &desc) {
    auto &vec = this->descs;
    auto itr = std::upper_bound(vec.begin(), vec.end(), desc);
    if (itr == vec.end())
        vec.push_back(desc);
    else
        vec.insert(itr, desc);
}

bool
nixlSecDescList::verifySorted() const {
    const auto &vec = this->descs;
    int size = (int)vec.size();
    if (size <= 1) return (size == 1);
    for (int i = 0; i < size - 1; ++i) {
        if (vec[i + 1] < vec[i]) return false;
    }
    return true;
}

nixlSectionDesc &
nixlSecDescList::operator[](unsigned int index) {
    nixlSectionDesc &ref = this->descs[index];
    assert(verifySorted());
    return ref;
}

int
nixlSecDescList::getIndex(const nixlBasicDesc &query) const {
    auto itr = std::lower_bound(this->descs.begin(), this->descs.end(), query);
    if (itr == this->descs.end()) return NIXL_ERR_NOT_FOUND;
    if (static_cast<const nixlBasicDesc &>(*itr) == query)
        return static_cast<int>(itr - this->descs.begin());
    return NIXL_ERR_NOT_FOUND;
}

int
nixlSecDescList::getCoveringIndex(const nixlBasicDesc &query) const {
    auto itr = std::lower_bound(this->descs.begin(), this->descs.end(), query);
    if (itr != this->descs.end() && itr->covers(query))
        return static_cast<int>(itr - this->descs.begin());
    // If query and element don't have the same start address, try previous entry
    if (itr != this->descs.begin()) {
        auto prev_itr = std::prev(itr, 1);
        if (prev_itr->covers(query)) return static_cast<int>(prev_itr - this->descs.begin());
    }
    return -1;
}

void
nixlSecDescList::resize(const size_t &count) {
    if (count > this->descs.size())
        throw std::logic_error(
            "nixlSecDescList: to keep list sorted, resize growth is not allowed.");
    this->descs.resize(count);
}

nixlRemoteDesc::nixlRemoteDesc(const uintptr_t addr,
                               const size_t len,
                               const uint64_t dev_id,
                               const std::string &remote_agent)
    : nixlBasicDesc{addr, len, dev_id},
      remoteAgent{remote_agent} {}

nixlRemoteDesc::nixlRemoteDesc(const nixlBasicDesc &desc, const std::string &remote_agent)
    : nixlBasicDesc{desc},
      remoteAgent{remote_agent} {}

nixlRemoteDesc::nixlRemoteDesc(const nixl_blob_t &blob) {
    if (blob.size() < sizeof(nixlBasicDesc)) {
        addr = 0;
        len = 0;
        devId = 0;
        return;
    }

    std::memcpy(static_cast<nixlBasicDesc *>(this), blob.data(), sizeof(nixlBasicDesc));
    if (blob.size() > sizeof(nixlBasicDesc)) {
        remoteAgent = blob.substr(sizeof(nixlBasicDesc));
    }
}

nixl_blob_t
nixlRemoteDesc::serialize() const {
    return nixlBasicDesc::serialize() + remoteAgent;
}

bool
operator==(const nixlRemoteDesc &lhs, const nixlRemoteDesc &rhs) {
    return (static_cast<const nixlBasicDesc &>(lhs) == static_cast<const nixlBasicDesc &>(rhs)) &&
        (lhs.remoteAgent == rhs.remoteAgent);
}
