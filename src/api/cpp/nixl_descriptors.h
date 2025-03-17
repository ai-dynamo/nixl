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

/**
 * @class nixlBasicDesc
 * @brief A basic descriptor class, contiguous in memory, with some supporting methods
 */
class nixlBasicDesc {
    public:
        /** @var Start of Buffer */
        uintptr_t addr;
        /** @var Buffer Length */
        size_t    len;
        /** @var device/file/blockID */
        uint32_t  devId;

        /**
         * @brief Default constructor for nixlBasicDesc
         *	  Does not initialize members to zero
         */
        nixlBasicDesc() {};
        /**
         * @brief Parametrized constructor for nixlBasicDesc
         *
         * @param addr  Start of buffer/block/offset in file
         * @param len   Length of buffer
         * @param devID deviceID/fileID/BlockID
         */
        nixlBasicDesc(const uintptr_t &addr,
                      const size_t &len,
                      const uint32_t &dev_id);
        /**
         * @brief Parametrized constructor for nixlBasicDesc
         *        with serialized blob of another nixlBasicDesc
         *
         * @param str   Serialized Descriptor
         */
        nixlBasicDesc(const nixl_blob_t &str); // deserializer
        /**
         * @brief Copy constructor for nixlBasicDesc
         *
         * @param desc   nixlBasicDesc object
         */
        nixlBasicDesc(const nixlBasicDesc &desc) = default;
        /**
         * @brief Operator overloading constructor
         *        with nixlBasicDesc object
         *
         * @param desc   nixlBasicDesc object
         */
        nixlBasicDesc& operator=(const nixlBasicDesc &desc) = default;
        /**
         * @brief BasicDesc Destructor
         */
        ~nixlBasicDesc() = default;
        /**
         * @brief Operator overloading (==) to compare BasicDesc objects
         *
         * @param lhs   nixlBasicDesc object
         * @param rhs   nixlBasicDesc object
         *
         */
        friend bool operator==(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
        /**
         * @brief Operator overloading (!=) to compare BasicDesc objects
         *
         * @param lhs   nixlBasicDesc object
         * @param rhs   nixlBasicDesc object
         *
         */
        friend bool operator!=(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
        /**
         * @brief Check for complete coverage of BasicDesc object
         *
         * @param query   nixlBasicDesc object
         */
        bool covers (const nixlBasicDesc &query) const;
        /**
         * @brief Check for overlap between BasicDesc objects
         *
         * @param query   nixlBasicDesc Object
         */
        bool overlaps (const nixlBasicDesc &query) const;

        /**
         * @brief Copy Metadata from one descriptor to another.
         *        No meta info in BasicDesc, so not implemented
         */
        void copyMeta (const nixlBasicDesc &desc) {};
        /**
         * @brief Serialize descriptor into BLOB
         */
        nixl_blob_t serialize() const;
        /**
         * @brief Print descriptor for debugging
         *
         * @param Suffix to append to descriptor for debugging
         */
	    void print(const std::string &suffix) const;
};

/**
 * @class nixlBlobDesc
 * @brief A basic descriptor class, with additional metadata in a BLOB
 */
class nixlBlobDesc : public nixlBasicDesc {
    public:
        /** @var BLOB for metadata information */
        nixl_blob_t metaInfo;

        /** @var Reuse parent constructor without the metadata */
        nixl_blob_t metaInfo;
        using nixlBasicDesc::nixlBasicDesc;

        /**
         * @brief Parametrized constructor for nixlBlobDesc
         *
         * @param addr      Start of buffer/block/offset in file
         * @param len       Length of buffer
         * @param devID     deviceID/fileID/BlockID
         * @param meta_info Metadata Information String
         */
         nixlBlobDesc(const uintptr_t &addr, const size_t &len,
                      const uint32_t &dev_id, const nixl_blob_t &meta_info);
        /**
         * @brief Constructor for nixlBlobDesc from nixlBasicDesc
         *
         * @param desc      nixlBasicDesc object
         * @param meta_info Metadata information BLOB
         */
        nixlBlobDesc(const nixlBasicDesc &desc, const nixl_blob_t &meta_info);
        /**
         * @brief Constructor for nixlBlobDesc with serialized BLOB
         *
         * @param str   Serialized BLOB from other nixlBlobDesc
         */
        nixlBlobDesc(const nixl_blob_t &str);
        /**
         * @brief Operator overloading (==) to compare BlobDesc objects
         *
         * @param lhs   nixlBlobDesc object
         * @param rhs   nixlBlobDesc object
         */
        friend bool operator==(const nixlBlobDesc &lhs,
                               const nixlBlobDesc &rhs);
        /**
         * @brief Serialize nixlBlobDesc to a BLOB
         */
        nixl_blob_t serialize() const;
        /**
         * @brief Copy nixlBlobDesc metadata from one object to another
         */
        void copyMeta (const nixlBlobDesc &info);
        /**
         * @brief Print nixlBlobDesc for debugging purpose
         *
		 * @param suffix  String to append to the nixlBlobDesc for debugging
		 */
        void print(const std::string &suffix) const;
};

/**
 * @class nixlDescList
 * @brief A class for describing a list of various nixlDesc types
 */
template<class T>
class nixlDescList {
    private:
        /** @var NIXL memory type */
        nixl_mem_t     type;
        /** @var Unified addressing flag
		  *
		  * Should be true for DRAM/VRAM with global addressing over PCIe
		  * Should be false for file or other storage objects
		  */
        bool           unifiedAddr;
        /** @var Flag for if list should be sorted */
        bool           sorted;
        /** @var Vector for storing nixlDescs */
        std::vector<T> descs;

    public:
        /**
         * @brief Parametrized Constructor for nixlDescList
         *
         * @param type         NIXL memory type of descriptor list
         * @param unifiedAddr  Flag to set unified addressing (default = true)
         * @param sorted       Flag to set sorted option (default = false)
         * @param init_size    initial size for descriptor list (default = 0)
         */
        nixlDescList(const nixl_mem_t &type, const bool &unifiedAddr=true,
                     const bool &sorted=false, const int &init_size=0);
        /**
         * @brief Constructor for nixlDescList from nixlSerDes object
         *        nixlSerDes serializes/deserializes our classes into strings
         *
         * @param nixlSerDes object to construct nixlDescList
         */
        nixlDescList(nixlSerDes* deserializer);
        /**
         * @brief Constructor for creating nixlDescList from another list
         *
         * @param d_list other nixlDescList object
         */
        nixlDescList(const nixlDescList<T> &d_list) = default;
        /**
         * @brief Operator(=) overload for nixlDescList to assign DescList
         *
         * @param d_list nixlDescList object
         */
        nixlDescList& operator=(const nixlDescList<T> &d_list) = default;
        /**
         * @brief Descriptor List Destructor
         */
        ~nixlDescList () = default;
        /**
         * @brief      Get NIXL memory type for this DescList
         */
        inline nixl_mem_t getType() const { return type; }
        /**
         * @brief       Get unifiedAddr flag
         */
        inline bool isUnifiedAddr() const { return unifiedAddr; }
        /**
         * @brief        Get count of descriptors in list
         */
        inline int descCount() const { return descs.size(); }
        /**
         * @brief Check if DescList is empty or not
         */
        inline bool isEmpty() const { return (descs.size()==0); }
        /**
         * @brief Check if DescList is sorted or not
         *
         * nixlDescList is sorted in two different ways
         * First, for unifiedAddr cases, the list is sorted just on address
         * Second, for not unifiedAddr cases, the devID may contain file
         * information, and so we sort first on devID, then addr
         */
        inline bool isSorted() const { return sorted; }
        /**
         * @brief Check if nixlDescs in list overlap with each other
         */
        bool hasOverlaps() const;
        /**
         * @brief Operator overloading getting/setting descriptor at [index]
         */
        const T& operator[](unsigned int index) const;
        T& operator[](unsigned int index);
        /**
         * @brief Vector iterators for const and non-const elements
         */
        inline typename std::vector<T>::const_iterator begin() const
            { return descs.begin(); }
        inline typename std::vector<T>::const_iterator end() const
            { return descs.end(); }
        inline typename std::vector<T>::iterator begin()
            { return descs.begin(); }
        inline typename std::vector<T>::iterator end()
            { return descs.end(); }
        /**
         * @brief Operator overloading (==) to compare nixlDescList objects
         *
         * @param lhs   nixlDescList object
         * @param rhs   nixlDescList object
         *
         */
        template <class Y> friend bool operator==(const nixlDescList<Y> &lhs,
                                                  const nixlDescList<Y> &rhs);
        /**
         * @brief  Resize nixlDescList object
         *
         * @param count resize DescList object
         */
        void resize (const size_t &count);
        /**
         * @brief Recomputes if a nixlDescList is still sorted or not
         */
        bool verifySorted();
        /**
         * @brief Empty the descriptor lists
         */
        inline void clear() { descs.clear(); }
        /**
         * @brief     Add Descriptors to descriptor list
         * 	          If sorted, keeps it sorted
         */
        void addDesc(const T &desc);
        /**
         * @brief Remove descriptor from list at index
         *
         * @return status  Status value for if removal was successful
         */
        nixl_status_t remDesc(const int &index);
        /**
         * @brief Populate adds metadata to response based on query
         *        descriptor list. If one descriptor fully belongs to
         *        a descriptor in the target list copies the metadata
         *        key to it.
         *
         * @param  query    DescList object used as query
         * @param  resp     populated response for the nixlBasicDesc
         *
         * @return status   Status value in NIXL returned
         */
        nixl_status_t populate(const nixlDescList<nixlBasicDesc> &query,
                               nixlDescList<T> &resp) const;
        /**
         * @brief Converts a nixlDescList with metadata to BasicDesc list
         */
        nixlDescList<nixlBasicDesc> trim() const;
        /**
         * @brief  Check if desc overlaps descriptor list at index
         *
         * @param  index index of Descriptor in the list
         */
        bool overlaps (const T &desc, int &index) const;
        /**
         * @brief  Get the index of a list element based on query
         *
         * @param  query nixlBasicDesc object to use as list query
         * @return int   index of the queried BasicDesc
         */
        int getIndex(const nixlBasicDesc &query) const;
        /**
         * @brief Serialize a descriptor list with nixlSerDes class
         */
        nixl_status_t serialize(nixlSerDes* serializer) const;
        /**
         * @brief Print the descriptor list for debugging
         */
        void print() const;
};
/**
 * @brief A typedef for a nixlDescList<nixlBasicDesc>
 *        used for creating transfer descriptor lists
 */
typedef nixlDescList<nixlBasicDesc> nixl_xfer_dlist_t;
/**
 * @brief A typedef for a nixlDescList<nixlBlobDesc>
 *        used for creating registratoin descriptor lists
 */
typedef nixlDescList<nixlBlobDesc>  nixl_reg_dlist_t;

#endif
