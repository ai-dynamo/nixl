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
 * @brief A basic descriptor class, single contiguous memory/storage
 *        element, alongside supporting methods
 */
class nixlBasicDesc {
public:
    /** @var Start of Buffer */
    uintptr_t addr;
    /** @var Buffer Length */
    size_t len;
    /** @var deviceID/blockID/fileID */
    uint64_t devId;

    /**
     * @brief Default constructor for nixlBasicDesc
     *      Does not initialize members to zero
     */
    nixlBasicDesc() {};
    /**
     * @brief Parametrized constructor for nixlBasicDesc
     *
     * @param addr  Start of buffer/block/offset-in-file
     * @param len   Length of buffer
     * @param devID deviceID/BlockID/fileID
     */
    nixlBasicDesc(const uintptr_t &addr, const size_t &len, const uint64_t &dev_id);
    /**
     * @brief Deserializer constructor for nixlBasicDesc with
     *        serialized blob of another nixlBasicDesc
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
     * @brief Operator (=) overloading constructor
     *        with nixlBasicDesc object
     *
     * @param desc   nixlBasicDesc object
     */
    nixlBasicDesc &
    operator=(const nixlBasicDesc &desc) = default;
    /**
     * @brief nixlBasicDesc destructor
     */
    ~nixlBasicDesc() = default;
    /**
     * @brief Operator overloading (<) to compare BasicDesc objects
     *        Comparison criteria is devID, then addr, then len
     */
    bool
    operator<(const nixlBasicDesc &desc) const;
    /**
     * @brief Operator overloading (==) to compare BasicDesc objects
     *
     * @param lhs   nixlBasicDesc object
     * @param rhs   nixlBasicDesc object
     *
     */
    friend bool
    operator==(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
    /**
     * @brief Operator overloading (!=) to compare BasicDesc objects
     *
     * @param lhs   nixlBasicDesc object
     * @param rhs   nixlBasicDesc object
     *
     */
    friend bool
    operator!=(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
    /**
     * @brief Check if current object address range covers the input object's
     *
     * @param query   nixlBasicDesc object
     */
    bool
    covers(const nixlBasicDesc &query) const;
    /**
     * @brief Check for overlap between BasicDesc objects
     *
     * @param query   nixlBasicDesc Object
     */
    bool
    overlaps(const nixlBasicDesc &query) const;
    /**
     * @brief Serialize descriptor into a blob
     */
    nixl_blob_t
    serialize() const;
    /**
     * @brief Print descriptor for debugging
     *
     * @param suffix gets prepended to the descriptor print
     */
    void
    print(const std::string &suffix) const;
};

/**
 * @class nixlBlobDesc
 * @brief A descriptor class, with additional metadata in form of a blob
 *        bundled with a nixlBasicDesc.
 */
class nixlBlobDesc : public nixlBasicDesc {
public:
    /** @var blob for metadata information */
    nixl_blob_t metaInfo;

    /** @var Reuse parent constructor without the metadata */
    using nixlBasicDesc::nixlBasicDesc;

    /**
     * @brief Parametrized constructor for nixlBlobDesc
     *
     * @param addr      Start of buffer/block/offset-in-file
     * @param len       Length of buffer
     * @param devID     deviceID/BlockID/fileID
     * @param meta_info Metadata blob
     */
    nixlBlobDesc(const uintptr_t &addr,
                 const size_t &len,
                 const uint64_t &dev_id,
                 const nixl_blob_t &meta_info);
    /**
     * @brief Constructor for nixlBlobDesc from nixlBasicDesc and metadata blob
     *
     * @param desc      nixlBasicDesc object
     * @param meta_info Metadata blob
     */
    nixlBlobDesc(const nixlBasicDesc &desc, const nixl_blob_t &meta_info);
    /**
     * @brief Deserializer constructor for nixlBlobDesc with serialized blob
     *
     * @param str   Serialized blob from another nixlBlobDesc
     */
    nixlBlobDesc(const nixl_blob_t &str);
    /**
     * @brief Operator overloading (==) to compare nixlBlobDesc objects
     *
     * @param lhs   nixlBlobDesc object
     * @param rhs   nixlBlobDesc object
     */
    friend bool
    operator==(const nixlBlobDesc &lhs, const nixlBlobDesc &rhs);
    /**
     * @brief Serialize nixlBlobDesc to a blob
     */
    nixl_blob_t
    serialize() const;
    /**
     * @brief Print nixlBlobDesc for debugging purpose
     *
     * @param suffix gets prepended to the descriptor print
     */
    void
    print(const std::string &suffix) const;
};

/**
 * @class nixlDescList
 * @brief A class for describing a list of descriptors, as a template based on
 *        the nixlDesc type that is used.
 */
template<class T>
class nixlDescList {
private:
    /** @var NIXL memory type */
    nixl_mem_t type;
    /** @var Flag for if list should be kept sorted
     *       Comparison is done based on nixlBasicDesc (<) operator which
     *       has comparison order of devID, then addr, then len.
     */
    bool sorted;
    /** @var Vector for storing nixlDescs */
    std::vector<T> descs;

public:
    /**
     * @brief Parametrized Constructor for nixlDescList
     *
     * @param type         NIXL memory type of descriptor list
     * @param sorted       Flag to set sorted option (default = false)
     * @param init_size    initial size for descriptor list (default = 0)
     */
    nixlDescList(const nixl_mem_t &type, const bool &sorted = false, const int &init_size = 0);
    /**
     * @brief Deserializer constructor for nixlDescList from nixlSerDes object
     *        which serializes/deserializes our classes into/from blobs
     *
     * @param deserialize nixlSerDes object to construct nixlDescList
     */
    nixlDescList(nixlSerDes *deserializer);
    /**
     * @brief Copy constructor for creating nixlDescList from another object
     *        of the same type.
     *
     * @param d_list other nixlDescList object of the same type
     */
    nixlDescList(const nixlDescList<T> &d_list) = default;
    /**
     * @brief Operator = overloading constructor for nixlDescList
     *
     * @param d_list nixlDescList object
     */
    nixlDescList &
    operator=(const nixlDescList<T> &d_list) = default;
    /**
     * @brief nixlDescList Destructor
     */
    ~nixlDescList() = default;

    /**
     * @brief      Get NIXL memory type for this DescList
     */
    inline nixl_mem_t
    getType() const {
        return type;
    }

    /**
     * @brief get sorted flag
     */
    inline bool
    isSorted() const {
        return sorted;
    }

    /**
     * @brief       Get count of descriptors
     */
    inline int
    descCount() const {
        return descs.size();
    }

    /**
     * @brief Check if nixlDescList is empty or not
     */
    inline bool
    isEmpty() const {
        return (descs.size() == 0);
    }

    /**
     * @brief Operator [] overloading, get/set descriptor at [index].
     *        Can throw std::out_of_range exception.
     */
    const T &
    operator[](unsigned int index) const;
    T &
    operator[](unsigned int index);

    /**
     * @brief Vector like iterators for const and non-const elements
     */
    inline typename std::vector<T>::const_iterator
    begin() const {
        return descs.begin();
    }

    inline typename std::vector<T>::const_iterator
    end() const {
        return descs.end();
    }

    inline typename std::vector<T>::iterator
    begin() {
        return descs.begin();
    }

    inline typename std::vector<T>::iterator
    end() {
        return descs.end();
    }
    /**
     * @brief Operator overloading (==) to compare nixlDescList objects
     *
     * @param lhs   nixlDescList object
     * @param rhs   nixlDescList object
     *
     */
    template<class Y>
    friend bool
    operator==(const nixlDescList<Y> &lhs, const nixlDescList<Y> &rhs);
    /**
     * @brief Resize nixlDescList object. If new size is more than the
     *        original size, the sorted status will be negated if set.
     *
     * @param count Number of elements after resizing DescList object
     */
    void
    resize(const size_t &count);
    /**
     * @brief Verify if a nixlDescList is sorted, for instance after using
     *        resize and adding new elements. If true, the sorted flag is set.
     */
    bool
    verifySorted();

    /**
     * @brief Empty the descriptors list
     */
    inline void
    clear() {
        descs.clear();
    }

    /**
     * @brief     Add Descriptors to descriptor list
     *               If nixlDescList object is sorted, this method keeps it sorted
     */
    void
    addDesc(const T &desc);
    /**
     * @brief Remove descriptor from list at index
     *        Can throw std::out_of_range exception.
     */
    void
    remDesc(const int &index);
    /**
     * @brief Convert a nixlDescList with metadata by trimming it to a
     *        nixlDescList of nixlBasicDesc elements
     */
    nixlDescList<nixlBasicDesc>
    trim() const;
    /**
     * @brief  Get the index of a descriptor that matches the `query`
     *
     * @param  query nixlBasicDesc object to find among the object's descriptors
     * @return int   index of the queried nixlBasicDesc if found, or negative error value
     */
    int
    getIndex(const nixlBasicDesc &query) const;
    /**
     * @brief Serialize a descriptor list with nixlSerDes class
     * @param serializer nixlSerDes object to serialize nixlDescList
     * @return nixl_status_t Error code if serialize was not successful
     */
    nixl_status_t
    serialize(nixlSerDes *serializer) const;
    /**
     * @brief Print the descriptor list for debugging
     */
    void
    print() const;
};
/**
 * @brief A typedef for a nixlDescList<nixlBasicDesc>
 *        used for creating transfer descriptor lists
 */
using nixl_xfer_dlist_t = nixlDescList<nixlBasicDesc>;
/**
 * @brief A typedef for a nixlDescList<nixlBlobDesc>
 *        used for creating registratoin descriptor lists
 */
using nixl_reg_dlist_t = nixlDescList<nixlBlobDesc>;

#endif
