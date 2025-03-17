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
#ifndef _NIXL_DESCRIPTORS_H
#define _NIXL_DESCRIPTORS_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include "nixl_types.h"

// A basic descriptor class, contiguous in memory, with some supporting methods
class nixlBasicDesc {
    public:
        uintptr_t addr;  // Start of buffer
        size_t    len;   // Buffer length
        uint32_t  devId; // Device ID

        nixlBasicDesc() {}; // No initialization to zero
        nixlBasicDesc(const uintptr_t &addr,
                      const size_t &len,
                      const uint32_t &dev_id);
        nixlBasicDesc(const nixl_blob_t &blob); // deserializer
        nixlBasicDesc(const nixlBasicDesc &desc) = default;
        nixlBasicDesc& operator=(const nixlBasicDesc &desc) = default;
        ~nixlBasicDesc() = default;

        friend bool operator==(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
        friend bool operator!=(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
        bool covers (const nixlBasicDesc &query) const;
        bool overlaps (const nixlBasicDesc &query) const;

        void copyMeta (const nixlBasicDesc &desc) {}; // No meta info in BasicDesc
        nixl_blob_t serialize() const;
        void print(const std::string &suffix) const; // For debugging
};


// String next to each BasicDesc, used for extra info for memory registrartion
class nixlBlobDesc : public nixlBasicDesc {
    public:
        nixl_blob_t metaInfo;

        // Reuse parent constructor without the extra info
        using nixlBasicDesc::nixlBasicDesc;

        nixlBlobDesc(const uintptr_t &addr, const size_t &len,
                     const uint32_t &dev_id, const nixl_blob_t &meta_info);
        nixlBlobDesc(const nixlBasicDesc &desc, const nixl_blob_t &meta_info);
        nixlBlobDesc(const nixl_blob_t &blob); // Deserializer

        friend bool operator==(const nixlBlobDesc &lhs,
                               const nixlBlobDesc &rhs);

        nixl_blob_t serialize() const;
        void copyMeta (const nixlBlobDesc &info);
        void print(const std::string &suffix) const;
};


// A class for a list of descriptors, where transfer requests are made from.
// It has some additional methods to help with creation and population.
template<class T>
class nixlDescList {
    private:
        nixl_mem_t     type;
        bool           unifiedAddr;
        bool           sorted;
        std::vector<T> descs;

    public:
        nixlDescList(const nixl_mem_t &type, const bool &unifiedAddr=true,
                     const bool &sorted=false, const int &init_size=0);
        nixlDescList(nixlSerDes* deserializer);
        nixlDescList(const nixlDescList<T> &d_list) = default;
        nixlDescList& operator=(const nixlDescList<T> &d_list) = default;
        ~nixlDescList () = default;

        inline nixl_mem_t getType() const { return type; }
        inline bool isUnifiedAddr() const { return unifiedAddr; }
        inline int descCount() const { return descs.size(); }
        inline bool isEmpty() const { return (descs.size()==0); }
        inline bool isSorted() const { return sorted; }
        bool hasOverlaps() const;

        const T& operator[](unsigned int index) const;
        T& operator[](unsigned int index);
        inline typename std::vector<T>::const_iterator begin() const
            { return descs.begin(); }
        inline typename std::vector<T>::const_iterator end() const
            { return descs.end(); }
        inline typename std::vector<T>::iterator begin()
            { return descs.begin(); }
        inline typename std::vector<T>::iterator end()
            { return descs.end(); }

        template <class Y> friend bool operator==(const nixlDescList<Y> &lhs,
                                                  const nixlDescList<Y> &rhs);

        void resize (const size_t &count);
        bool verifySorted();
        inline void clear() { descs.clear(); }
        void addDesc(const T &desc); // If sorted, keeps it sorted
        nixl_status_t remDesc(const int &index);
        nixl_status_t populate(const nixlDescList<nixlBasicDesc> &query,
                               nixlDescList<T> &resp) const;
        nixlDescList<nixlBasicDesc> trim() const;

        bool overlaps (const T &desc, int &index) const;
        int getIndex(const nixlBasicDesc &query) const;
        nixl_status_t serialize(nixlSerDes* serializer) const;
        void print() const;
};

typedef nixlDescList<nixlBasicDesc> nixl_xfer_dlist_t;
typedef nixlDescList<nixlBlobDesc>  nixl_reg_dlist_t;

#endif
