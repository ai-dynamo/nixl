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
         *	  No initialization to zero
         */
        nixlBasicDesc() {};
        /**
         * @brief Parametrized Constructor for nixlBasicDesc
         *
         * @param addr  Start of buffer/block/offset in file
         * @param len   Length of buffer
         * @param devID deviceID/filID/BlockID
         */
        nixlBasicDesc(const uintptr_t &addr,
                      const size_t &len,
                      const uint32_t &dev_id);
        /**
         * @brief Parametrized Constructor for nixlBasicDesc
         *        with serialized string
         *
         * @param str   Serialized Descriptor
         */
        nixlBasicDesc(const nixl_blob_t &str); // deserializer
        /**
         * @brief Copy Constructor for nixlBasicDesc
         *        with Descriptor object
         *
         * @param desc   Descriptor Object
         */
        nixlBasicDesc(const nixlBasicDesc &desc) = default;
        /**
         * @brief Operator overloading constructor
         *        with Descriptor object
         *
         * @param desc   Descriptor Object
         */
        nixlBasicDesc& operator=(const nixlBasicDesc &desc) = default;
        /**
         * @brief Basic Desc Destructor
         */
        ~nixlBasicDesc() = default;
        /**
         * @brief Operator overloading (==) to compare BasicDesc objects
         *
         * @param lhs   BasicDesc Object
         * @param rhs   BasicDesc Object to compare
         *
         */
        friend bool operator==(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
        /**
         * @brief Operator overloading (!=) to compare BasicDesc objects
         *
         * @param lhs   Descriptor Object
         * @param rhs   Descriptor Object
         *
         */
        friend bool operator!=(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
        /**
         * @brief Check for complete coverage of BasicDesc object
         *
         * @param query   Descriptor Object
         */
        bool covers (const nixlBasicDesc &query) const;
        /**
         * @brief Check for overlap of BasicDesc object
         *
         * @param query   Descriptor Object
         */
        bool overlaps (const nixlBasicDesc &query) const;

        /**
         * @brief Copy Metadata from one
         *        object to another.
         *        No meta info in BasicDesc
         */
        void copyMeta (const nixlBasicDesc &desc) {};
        /**
         * @brief Serialize descriptor
         */
        nixl_blob_t serialize() const;
        /**
         * @brief Print descriptor for Debugging
         *
         * @param Serialized descriptor object
         */
	    void print(const std::string &suffix) const;
};


// String next to each BasicDesc, used for extra info for memory registrartion
class nixlBlobDesc : public nixlBasicDesc {
    public:
        /** @var String for metadata information */
        nixl_blob_t metaInfo;

        /** @var Reuse parent constructor without the extra info */
        using nixlBasicDesc::nixlBasicDesc;

        /**
         * @brief Parametrized Constructor for nixlBlobDesc
         *
         * @param addr      Start of buffer/block/offset in file
         * @param len       Length of buffer
         * @param devID     deviceID/filID/BlockID
         * @param meta_info Metadata Information String
         */
         nixlBlobDesc(const uintptr_t &addr, const size_t &len,
                      const uint32_t &dev_id, const nixl_blob_t &meta_info);
        /**
         * @brief Parametrized Constructor for nixlBlobDesc from nixlBasicDesc
         *
         * @param desc      Object for nixlBasicDesc
         * @param meta_info Metadata Information String
         */
        nixlBlobDesc(const nixlBasicDesc &desc, const nixl_blob_t &meta_info);
        /**
         * @brief Parametrized Constructor for nixlBasicDesc
         *        with serialized string
         *
         * @param str   Serialized Descriptor
         */
        nixlBlobDesc(const std::string &str); // Deserializer
        /**
         * @brief Operator overloading (==) to compare BlobDesc objects
         *
         * @param lhs   BlobDesc Object
         * @param rhs   BlobDesc Object to compare
         */
        friend bool operator==(const nixlBlobDesc &lhs,
                               const nixlBlobDesc &rhs);
        /**
         * @brief Serialize BlobDesc to a string
         */
        nixl_blob_t serialize() const;
        /**
         * @brief Copy nixlBlobDesc Metadata from one
         *        object to another
         */
        void copyMeta (const nixlBlobDesc &info);
        /**
         * @brief Print Descriptor based on a
         *        serialized string
         */
        void print(const std::string &suffix) const;
};

/**
 * @class nixlDescList
 * @brief A class for a list of descriptors, where transfer requests are made from.
 *        It has some additional methods to help with creation and population.
 */
template<class T>
class nixlDescList {
    private:
        /** @var NIXL memory type */
        nixl_mem_t     type;
        /** @var unified Addressing flag */
        bool           unifiedAddr;
        /** @var Descriptor List sorted flag */
        bool           sorted;
        /** @var Descriptor list vector */
        std::vector<T> descs;

    public:
        /**
         * @brief Parametrized Constructor for nixlDescList
         *
         * @param type         Memory type of descriptor list
         * @param unifiedAddr  Flag to set unified addressing (default = true)
         * @param sorted       Flag to set sorted option (default = false)
         * @param init_size    initial size for descriptor list (default = 0)
         */
        nixlDescList(const nixl_mem_t &type, const bool &unifiedAddr=true,
                     const bool &sorted=false, const int &init_size=0);
        /**
         * @brief Parametrized Constructor for nixlDescList from Serializer
         *
         *
         * @param deserializer Serializer object to construct DescList
         */
        nixlDescList(nixlSerDes* deserializer);
        /**
         * @brief Parametrized Constructor for nixlDescList from DescList
         *        object
         *
         * @param d_list DescList object
         */
        nixlDescList(const nixlDescList<T> &d_list) = default;
        /**
         * @brief Operator(=) overloaded for nixlDescList to assign
         *         DescList
         *
         * @param d_list DescList object
         */
        nixlDescList& operator=(const nixlDescList<T> &d_list) = default;
        /**
         * @brief Basic Desc Destructor
         */
        ~nixlDescList () = default;
        /**
         * @brief      Get NIXL memory type for this DescList
         * @return mem nixl_mem_t type returned
         */
        inline nixl_mem_t getType() const { return type; }
        /**
         * @brief       Get DescList unifiedAddr property
         * @return true if the address is unified/false otherwise
         */
        inline bool isUnifiedAddr() const { return unifiedAddr; }
        /**
         * @brief        Get DescList descriptor count
         *
         * @return count Count of descriptors in DescList
         */
        inline int descCount() const { return descs.size(); }
        /**
         * @brief Check if DescList is Empty()
         *
         * @return true if sorted/false otherwise
         */
        inline bool isEmpty() const { return (descs.size()==0); }
        /**
         * @brief Check if DescList is Sorted()
         * @return true if empty/false otherwise
         */
        inline bool isSorted() const { return sorted; }
        /**
         * @brief Check if DescList has overlaps
         * @return true if overlaps/false otherwise
         */
        bool hasOverlaps() const;
        /**
         * @brief Operator Overloading getting DescList at []
         */
        const T& operator[](unsigned int index) const;
        T& operator[](unsigned int index);
        /**
         * @brief DescList convenience Iterators for const and non-const
         *        objects
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
         * @brief Operator overloading (==) to compare Descriptor list objects
         *
         * @param lhs   Descriptor List Object
         * @param rhs   Descriptor List Object
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
         * @brief Check if DescList is sorted
         * @return true if sorted/false otherwise
         */
        bool verifySorted();
        /**
         * @brief Empty the descriptor lists
         */
        inline void clear() { descs.clear(); }
        /**
         * @brief     Add Descriptors to descriptor list
         * 	      If sorted, keeps it sorted
         */
        void addDesc(const T &desc);
        /**
         * @brief Remove descriptors from list at index
         *
         * @return status   Status value in NIXL returned
         */
        nixl_status_t remDesc(const int &index);
        /**
         * @brief Populate adds metadata to response based on queried
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
	 * @brief Converts a desc with metadata to BasicDesc
         *
         * @return nixlDescList<nixlBasicDesc>   DescList of type BasicDesc
         */
        nixlDescList<nixlBasicDesc> trim() const;
        /**
         * @brief  Check if desc overlaps descriptor list at index
         *
         * @param  index index of Descriptor in the list
         *
         * @return bool  Flag to say if it overlaps or not
         */
        bool overlaps (const T &desc, int &index) const;
        /**
         * @brief  Get the index of a BasicDesc object
         *
         * @param  query nixlBasicDesc object to get the index
         * @return int   index of the queried BasicDesc
         */
        int getIndex(const nixlBasicDesc &query) const;
        /**
         * @brief Serialize a descriptor list to a string
         *
         * @param serializer Object to a serializer for DescList
         */
        nixl_status_t serialize(nixlSerDes* serializer) const;
        /**
         * @brief Print the Descriptor List
         */
        void print() const;
};
/**
 * @brief A typedef for a nixlDescList<nixlBasicDesc>
 *        used for creating xfer decs list
 */
typedef nixlDescList<nixlBasicDesc> nixl_xfer_dlist_t;
/**
 * @brief A typedef for a nixlDescList<nixlBlobDesc>
 *        used for creating registratin desc liost
 */
typedef nixlDescList<nixlBlobDesc>  nixl_reg_dlist_t;

#endif
