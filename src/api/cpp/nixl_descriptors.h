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
#ifndef _NIXL_DESCRIPTORS_H
#define _NIXL_DESCRIPTORS_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cassert>
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
    operator<(const nixlBasicDesc &desc) const noexcept {
        if (devId != desc.devId) return (devId < desc.devId);
        if (addr != desc.addr) return (addr < desc.addr);
        return (len < desc.len);
    }

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
    covers(const nixlBasicDesc &query) const noexcept {
        return (devId == query.devId) && (addr <= query.addr) &&
            ((addr + len) >= (query.addr + query.len));
    }

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
 * @struct nixlRemoteDesc
 * @brief A descriptor structure for remote buffers, with remote agent name bundled with a
 * nixlBasicDesc.
 */
struct nixlRemoteDesc : public nixlBasicDesc {
    /** Remote agent name */
    std::string remoteAgent;

    /** Reuse parent constructor without the remote agent name */
    using nixlBasicDesc::nixlBasicDesc;

    /**
     * @brief Parametrized constructor for nixlRemoteDesc
     *
     * @param addr          Start of buffer/block/offset-in-file
     * @param len           Length of buffer
     * @param dev_id        deviceID/BlockID/bufferID (remote ID)
     * @param remote_agent  Remote agent name
     */
    nixlRemoteDesc(const uintptr_t addr,
                   const size_t len,
                   const uint64_t dev_id,
                   const std::string &remote_agent);

    /**
     * @brief Constructor for nixlRemoteDesc from nixlBasicDesc and remote agent name
     *
     * @param desc          nixlBasicDesc object
     * @param remote_agent  Remote agent name
     */
    nixlRemoteDesc(const nixlBasicDesc &desc, const std::string &remote_agent);

    /**
     * @brief Deserializer constructor for nixlRemoteDesc with serialized blob
     *
     * @param str   Serialized blob from another nixlRemoteDesc
     */
    explicit nixlRemoteDesc(const nixl_blob_t &str);

    /**
     * @brief Serialize nixlRemoteDesc to a blob
     */
    nixl_blob_t
    serialize() const;
};

/**
 * @brief Operator overloading (==) to compare nixlRemoteDesc objects
 *
 * @param lhs   nixlRemoteDesc object
 * @param rhs   nixlRemoteDesc object
 */
bool
operator==(const nixlRemoteDesc &lhs, const nixlRemoteDesc &rhs);

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
    /** @var Vector for storing nixlDescs */
    std::vector<T> descs;
    /** @var Whether the descriptor list is a shallow copy */
    bool isShallowCopy_;
    /** @var Common view of the descriptor list, both for shallow copy and own data */
    const T *view_;
    /** @var Size of the descriptor list */
    size_t size_;

    /**
     * @brief Synchronize the view of the descriptor list with the data in the vector.
     *        Can be called only in owner mode (not shallow copy), after vector is modified
     */
    inline void
    syncView() {
        view_ = descs.data();
        size_ = descs.size();
    }

    /**
     * @brief Check that the descriptor list is not a shallow copy.
     *        Throws std::logic_error if it is.
     */
    inline void
    checkModifiable() const {
        if (isShallowCopy_) {
            throw std::logic_error("Descriptor list is a non-modifiable shallow copy");
        }
    }

    /**
     * @brief Swap two nixlDescList objects.
     */
    static void
    swap(nixlDescList<T> &first, nixlDescList<T> &second) noexcept;

    /**
     * @brief Private constructor to create a shallow copy of a nixlDescList.
     *        Used internally by the makeShallowCopy to return only a const reference
     *        to the shallow copy.
     */
    nixlDescList(const nixl_mem_t &type, const T *view, size_t size);

public:
    using iterator_t = typename std::vector<T>::iterator;
    using const_iterator_t = const T *;

    /**
     * @brief Parametrized Constructor for nixlDescList
     *
     * @param type         NIXL memory type of descriptor list
     * @param init_size    initial size for descriptor list (default = 0)
     */
    nixlDescList(const nixl_mem_t &type, size_t init_size = 0);

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
    nixlDescList(const nixlDescList<T> &d_list);

    /**
     * @brief Move constructor for creating nixlDescList from another object
     *        of the same type.
     *
     * @param d_list other nixlDescList object of the same type
     */
    nixlDescList(nixlDescList<T> &&d_list) noexcept;

    /**
     * @brief Operator = overloading constructor for nixlDescList
     *
     * @param d_list nixlDescList object
     */
    nixlDescList &
    operator=(nixlDescList<T> d_list) noexcept;

    /**
     * @brief Create a shallow copy of a nixlDescList.
     *
     * @param type         NIXL memory type of descriptor list
     * @param view         View of the descriptor list
     * @param size         Size of the descriptor list
     * @return Shallow copy of the descriptor list
     */
    static const nixlDescList<T>
    makeShallowCopy(const nixl_mem_t &type, const T *view, size_t size) {
        return nixlDescList<T>(type, view, size);
    }

    /**
     * @brief nixlDescList Destructor
     */
    virtual ~nixlDescList() = default;

    /**
     * @brief      Get NIXL memory type for this DescList
     */
    inline nixl_mem_t
    getType() const {
        return type;
    }

    /**
     * @brief       Get count of descriptors
     */
    inline int
    descCount() const {
        return size_;
    }

    inline size_t
    size() const {
        return size_;
    }

    /**
     * @brief Check if nixlDescList is empty or not
     */
    inline bool
    isEmpty() const {
        return (size_ == 0);
    }

    inline const T &
    operator[](size_t index) const {
        return view_[index];
    }

    inline T &
    operator[](size_t index) {
        assert(!isShallowCopy_);
        return descs[index];
    }

    /**
     * @brief Vector like iterators for const and non-const elements
     */
    inline const_iterator_t
    begin() const {
        return view_;
    }

    inline const_iterator_t
    end() const {
        return view_ + size_;
    }

    inline iterator_t
    begin() {
        assert(!isShallowCopy_);
        return descs.begin();
    }

    inline iterator_t
    end() {
        assert(!isShallowCopy_);
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
     * @brief Resize nixlDescList object.
     *
     * @param count Number of elements after resizing DescList object
     */
    virtual void
    resize(size_t count);

    /**
     * @brief Empty the descriptors list
     */
    inline void
    clear() {
        checkModifiable();
        descs.clear();
        syncView();
    }

    /**
     * @brief     Add Descriptors to descriptor list
     */
    virtual void
    addDesc(T desc);

    void
    addDesc(T desc, iterator_t it);

    /**
     * @brief Remove descriptor from list at index
     *        Can throw std::out_of_range exception.
     */
    void
    remDesc(size_t index);

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
    virtual int
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

    /**
     * @brief Dump the descriptor list into a string for debugging
     */
    std::string
    to_string(bool compact = false) const;
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
/**
 * @brief An alias for a nixlDescList<nixlRemoteDesc>
 *        used for preparing a memory view handle for remote buffers
 */
using nixl_remote_dlist_t = nixlDescList<nixlRemoteDesc>;

#endif
