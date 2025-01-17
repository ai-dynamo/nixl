#include <algorithm>
#include <iostream>
#include <functional>
#include <stdexcept>
#include "nixl.h"
#include "nixl_descriptors.h"
#include "backend/backend_engine.h"
#include "utils/serdes/serdes.h"

/*** Class nixlBasicDesc implementation ***/

// No Virtual function in nixlBasicDesc class or its children, as we want
// each object to just have the members during serialization.

nixlBasicDesc::nixlBasicDesc(const uintptr_t &addr,
                             const size_t &len,
                             const uint32_t &dev_id) {
    this->addr  = addr;
    this->len   = len;
    this->devId = dev_id;
}

nixlBasicDesc::nixlBasicDesc(const std::string &str) {
    if (str.size()==sizeof(nixlBasicDesc)) {
        str.copy(reinterpret_cast<char*>(this), sizeof(nixlBasicDesc));
    } else { // Error indicator, not possible by descList deserializer call
        addr  = 0;
        len   = 0;
        devId = 0;
    }
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

std::string nixlBasicDesc::serialize() const {
    return std::string(reinterpret_cast<const char*>(this),
                       sizeof(nixlBasicDesc));
}

void nixlBasicDesc::print(const std::string &suffix) const {
    std::cout << "DEBUG: Desc (" << addr << ", " << len
              << ") from devID " << devId << suffix << "\n";
}


/*** Class nixlStringDesc implementation ***/

nixlStringDesc::nixlStringDesc(const uintptr_t &addr,
                               const size_t &len,
                               const uint32_t &dev_id,
                               const std::string &meta_info) {
    nixlBasicDesc(addr, len, dev_id);
    this->metaInfo = meta_info;
}

nixlStringDesc::nixlStringDesc(const nixlBasicDesc &desc,
                               const std::string &meta_info) {
    this->addr     = desc.addr;
    this->len      = desc.len;
    this->devId    = desc.devId;
    this->metaInfo = meta_info;
}

nixlStringDesc::nixlStringDesc(const std::string &str) {
    size_t meta_size = str.size() - sizeof(nixlBasicDesc);
    if (meta_size>0) {
        metaInfo.resize(meta_size);
        str.copy(reinterpret_cast<char*>(this), sizeof(nixlBasicDesc));
        str.copy(reinterpret_cast<char*>(&metaInfo[0]),
                 meta_size, sizeof(nixlBasicDesc));
    } else { // Error indicator, not possible by descList deserializer call
        addr  = 0;
        len   = 0;
        devId = 0;
        metaInfo.resize(0);
    }
}

bool operator==(const nixlStringDesc &lhs, const nixlStringDesc &rhs) {
    return (((nixlBasicDesc)lhs == (nixlBasicDesc)rhs) &&
                  (lhs.metaInfo == rhs.metaInfo));
}

std::string nixlStringDesc::serialize() const {
    return nixlBasicDesc::serialize() + metaInfo;
}

void nixlStringDesc::copyMeta (const nixlStringDesc &info){
    this->metaInfo = info.metaInfo;
}

void nixlStringDesc::print(const std::string &suffix) const {
    nixlBasicDesc::print(", Metadata: " + metaInfo + suffix);
}

/*** Class nixlDescList implementation ***/

// The template is used to select from nixlBasicDesc/nixlMetaDesc/nixlStringDesc
// There are no virtual functions, so the object is all data, no pointers.

template <class T>
nixlDescList<T>::nixlDescList (const nixl_mem_t &type, const bool &unified_addr,
                               const bool &sorted, const int &init_size) {
    static_assert(std::is_base_of<nixlBasicDesc, T>::value);
    this->type        = type;
    this->unifiedAddr = unified_addr;
    this->sorted      = sorted;
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

    // nixlMetaDesc should be internall and not be serialized
    if ((str == "nixlMDList") || (std::is_same<nixlMetaDesc, T>::value))
        return;

    if (deserializer->getBuf("t", &type, sizeof(type)))
        return;
    if (deserializer->getBuf("u", &unifiedAddr, sizeof(unifiedAddr)))
        return;
    if (deserializer->getBuf("s", &sorted, sizeof(sorted)))
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

    } else if(std::is_same<nixlStringDesc, T>::value) {
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
    if (index >= descs.size())
        throw std::out_of_range("Index is out of range");
    return descs[index];
}

// Setter
template <class T>
inline T& nixlDescList<T>::operator[](unsigned int index) {
    if (index >= descs.size())
        throw std::out_of_range("Index is out of range");
    sorted = false;
    return descs[index];
}

// Internal function used for sorting the Vector and logarithmic search
bool descAddrCompare (const nixlBasicDesc &a, const nixlBasicDesc &b,
                      bool unifiedAddr) {
    if (unifiedAddr) // Ignore devId
        return (a.addr < b.addr);
    if (a.devId < b.devId)
        return true;
    if (a.devId == b.devId)
        return (a.addr < b.addr);
    return false;
}

#define desc_comparator_f [&](const nixlBasicDesc &a, const nixlBasicDesc &b) {\
                              return descAddrCompare(a, b, unifiedAddr); }

template <class T>
void nixlDescList<T>::addDesc (const T &desc) {
    if (!sorted) {
        descs.push_back(desc);
    } else {
        // Since vector is kept soted, we can use upper_bound
        auto itr = std::upper_bound(descs.begin(), descs.end(),
                                    desc, desc_comparator_f);
        if (itr == descs.end())
            descs.push_back(desc);
        else
            descs.insert(itr, desc);
    }
}

template <class T>
bool nixlDescList<T>::overlaps (const T &desc, int &index) const {
    if (!sorted) {
        for (size_t i=0; i<descs.size(); ++i) {
            if (descs[i].overlaps(desc)) {
                index = i;
                return true;
            }
        }
        index = descs.size();
        return false;
    } else {
        // Since desc vector is kept sorted, we can use upper_bound
        auto itr = std::upper_bound(descs.begin(), descs.end(),
                                    desc, desc_comparator_f);
        if (itr == descs.end()) {
            index = descs.size();
            return false;
        } else {
            index = itr - descs.begin();
            // If between 2 descriptors, index can be used for insertion
            return ((*itr).overlaps(desc));
        }
    }
}

template <class T>
bool nixlDescList<T>::hasOverlaps () const {
    if ((descs.size()==0) || (descs.size()==1))
        return false;

    if (!sorted) {
        for (size_t i=0; i<descs.size()-1; ++i)
            for (size_t j=i+1; j<descs.size(); ++j)
                if (descs[i].overlaps(descs[j]))
                    return true;
    } else {
        for (size_t i=0; i<descs.size()-1; ++i)
            if (descs[i].overlaps(descs[i+1]))
                return true;
    }

    return false;
}

template <class T>
nixl_status_t nixlDescList<T>::remDesc (const int &index){
    if (((size_t) index >= descs.size()) || (index < 0))
        return NIXL_ERR_INVALID_PARAM;
    descs.erase(descs.begin() + index);
    return NIXL_SUCCESS;
}

template <class T>
void nixlDescList<T>::resize (const size_t &count) {
    if (count > descs.size())
        sorted = false;
    descs.resize(count);
}

template <class T>
bool nixlDescList<T>::verifySorted() {
    int size = (int) descs.size();
    if (size==0) {
        return false;
    } else if (size == 1) {
        sorted = true;
        return true;
    }

    for (int i=0; i<size-1; ++i) {
        if (descAddrCompare(descs[i+1], descs[i], unifiedAddr)) {
            sorted = false;
            return false;
        }
    }
    sorted = true;
    return true;
}

template <class T>
nixl_status_t nixlDescList<T>::populate (const nixlDescList<nixlBasicDesc> &query,
                                         nixlDescList<T> &resp) const {
    // Populate only makes sense when there is extra metadata
    if(std::is_same<nixlBasicDesc, T>::value)
        return NIXL_ERR_INVALID_PARAM;

    if ((type != query.getType()) || (type != resp.type))
        return NIXL_ERR_INVALID_PARAM;

    if ((unifiedAddr != query.isUnifiedAddr()) ||
        (unifiedAddr != resp.unifiedAddr))
        return NIXL_ERR_INVALID_PARAM;

    // 1-to-1 mapping cannot hold
    if (query.isSorted() != resp.sorted)
        return NIXL_ERR_INVALID_PARAM;

    T new_elm;
    nixlBasicDesc *p = &new_elm;
    int count = 0, last_found = 0;
    int s_index, q_index, size;
    bool found, q_sorted = query.isSorted();
    const nixlBasicDesc *q, *s;

    resp.resize(query.descCount());

    if (!sorted) {
        for (int i=0; i<query.descCount(); ++i)
            for (auto & elm : descs)
                if (elm.covers(query[i])){
                    *p = query[i];
                    new_elm.copyMeta(elm);
                    resp.descs[i]=new_elm;
                    count++;
                    break;
                }

        if (query.descCount()==count) {
            return NIXL_SUCCESS;
        } else {
            resp.clear();
            return NIXL_ERR_BAD;
        }
    } else {
        if (q_sorted) {
            size = (int) descs.size();
            s_index = 0;
            q_index = 0;

            while (q_index<query.descCount()){
                s = &descs[s_index];
                q = &query[q_index];
                if ((*s).covers(*q)) {
                    *p = *q;
                    new_elm.copyMeta(descs[s_index]); // needs const nixlBasicDesc&
                    resp.descs[q_index] = new_elm;
                    q_index++;
                } else {
                    s_index++;
                    // if ((descAddrCompare(*q, descs[s_index], unifiedAddr)) ||
                    if (s_index==size) {
                        resp.clear();
                        return NIXL_ERR_BAD;
                    }
                }
            }

            resp.sorted = true; // Should be redundant
            return NIXL_SUCCESS;

        } else {
            for (int i=0; i<query.descCount(); ++i) {
                found = false;
                q = &query[i];
                auto itr = std::lower_bound(descs.begin() + last_found,
                                            descs.end(), *q, desc_comparator_f);

                // Same start address case
                if (itr != descs.end()){
                    if ((*itr).covers(*q)) {
                        found = true;
                    }
                }

                // query starts starts later, try previous entry
                if ((!found) && (itr != descs.begin())){
                    itr = std::prev(itr , 1);
                    if ((*itr).covers(*q)) {
                        found = true;
                    }
                }

                if (found) {
                    *p = *q;
                    new_elm.copyMeta(*itr);
                    resp.descs[i] = new_elm;
                } else {
                    resp.clear();
                    return NIXL_ERR_BAD;
                }
            }
            resp.sorted = query.isSorted(); // Update as resize resets it
            return NIXL_SUCCESS;
        }
    }
}

template <class T>
nixlDescList<nixlBasicDesc> nixlDescList<T>::trim() const {

    // Potential optimization for (std::is_same<nixlBasicDesc, T>::value)
    nixlDescList<nixlBasicDesc> trimmed(type, unifiedAddr, sorted);
    nixlBasicDesc* p;

    for (auto & elm: descs) {
        p = (nixlBasicDesc*) (&elm);
        trimmed.addDesc(*p);
    }

    // No failure scenario
    return trimmed;
}

template <class T>
int nixlDescList<T>::getIndex(const nixlBasicDesc &query) const {
    if (!sorted) {
        auto itr = std::find(descs.begin(), descs.end(), query);
        if (itr == descs.end())
            return -1; // not found
        return itr - descs.begin();
    } else {
        auto itr = std::lower_bound(descs.begin(), descs.end(),
                                    query, desc_comparator_f);
        if (itr == descs.end())
            return -1; // not found
        // As desired, becomes nixlBasicDesc on both sides
        if (*itr == query)
            return itr - descs.begin();
    }
    return -1;
}

template <class T>
nixl_status_t nixlDescList<T>::serialize(nixlSerDes* serializer) const {

    nixl_status_t ret;
    size_t n_desc = descs.size();

    // nixlMetaDesc should be internall and not be serialized
    if(std::is_same<nixlMetaDesc, T>::value)
        return NIXL_ERR_INVALID_PARAM;

    if (std::is_same<nixlBasicDesc, T>::value)
        ret = serializer->addStr("nixlDList", "nixlBDList");
    else if (std::is_same<nixlStringDesc, T>::value)
        ret = serializer->addStr("nixlDList", "nixlSDList");
    else
        return NIXL_ERR_INVALID_PARAM;

    if (ret) return ret;

    ret = serializer->addBuf("t", &type, sizeof(type));
    if (ret) return ret;

    ret = serializer->addBuf("u", &unifiedAddr, sizeof(unifiedAddr));
    if (ret) return ret;

    ret = serializer->addBuf("s", &sorted, sizeof(sorted));
    if (ret) return ret;

    ret = serializer->addBuf("n", &(n_desc), sizeof(n_desc));
    if (ret) return ret;

    if (n_desc==0)
        return NIXL_SUCCESS; // Unusual, but supporting it

    if (std::is_same<nixlBasicDesc, T>::value) {
        // Contiguous in memory, so no need for per elm serialization
        ret = serializer->addStr("", std::string(
                                 reinterpret_cast<const char*>(descs.data()),
                                 n_desc * sizeof(nixlBasicDesc)));
        if (ret) return ret;
    } else { // already checked it can be only nixlStringDesc
        for(auto & elm : descs) {
            ret = serializer->addStr("", elm.serialize());
            if(ret) return ret;
        }
    }

    return NIXL_SUCCESS;
}

template <class T>
void nixlDescList<T>::print() const {
    std::cout << "DEBUG: DescList of mem type " << type
              << (unifiedAddr ? " with" : " without") << " unified addressing and "
              << (sorted ? "sorted" : "unsorted") << "\n";
    for (auto & elm : descs) {
        std::cout << "    ";
        elm.print("");
    }
}

template <class T>
bool operator==(const nixlDescList<T> &lhs, const nixlDescList<T> &rhs) {
    if ((lhs.getType()       != rhs.getType())       ||
        (lhs.descCount()     != rhs.descCount())     ||
        (lhs.isUnifiedAddr() != rhs.isUnifiedAddr()) ||
        (lhs.isSorted()      != rhs.isSorted()))
        return false;

    for (size_t i=0; i<lhs.descs.size(); ++i)
        if (lhs.descs[i] != rhs.descs[i])
            return false;
    return true;
}

// Since we implement a template class declared in a header files, this is necessary
template class nixlDescList<nixlBasicDesc>;
template class nixlDescList<nixlMetaDesc>;
template class nixlDescList<nixlStringDesc>;

template bool operator==<nixlBasicDesc> (const nixlDescList<nixlBasicDesc> &lhs,
                                         const nixlDescList<nixlBasicDesc> &rhs);
template bool operator==<nixlMetaDesc>  (const nixlDescList<nixlMetaDesc> &lhs,
                                         const nixlDescList<nixlMetaDesc> &rhs);
template bool operator==<nixlStringDesc>(const nixlDescList<nixlStringDesc> &lhs,
                                         const nixlDescList<nixlStringDesc> &rhs);
