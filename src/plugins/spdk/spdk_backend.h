/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_SPDK_SPDK_BACKEND_H
#define NIXL_SRC_PLUGINS_SPDK_SPDK_BACKEND_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "backend/backend_engine.h"

#include <spdk/bdev.h>

class nixlSpdkProgressEngine;
class nixlSpdkBackendReqH;

// Common base for this backend's metadata. The kind tag discriminates the
// subtypes without RTTI: one checked dynamic_cast at the NIXL boundary proves a
// pointer is ours, and everything after that switches on kind().
//
// Copying is deleted because nixlSpdkBdevMD owns SPDK handles that a duplicate
// would close twice; these are also polymorphic, so a copy would slice.
class nixlSpdkMD : public nixlBackendMD {
public:
    enum class Kind : uint8_t {
        Dram,
        Bdev,
        Obj,
    };

    explicit nixlSpdkMD(Kind kind) noexcept : nixlBackendMD(true), kind_(kind) {}

    nixlSpdkMD(const nixlSpdkMD &) = delete;
    nixlSpdkMD &
    operator=(const nixlSpdkMD &) = delete;

    [[nodiscard]] Kind
    kind() const noexcept {
        return kind_;
    }

private:
    Kind kind_;
};

struct nixlSpdkDramMD : nixlSpdkMD {
    nixlSpdkDramMD(uintptr_t addr, std::size_t len) noexcept
        : nixlSpdkMD(Kind::Dram),
          addr(addr),
          len(len) {}

    uintptr_t addr;
    std::size_t len;
};

struct nixlSpdkBdevMD : nixlSpdkMD {
    explicit nixlSpdkBdevMD(std::string bdev_name)
        : nixlSpdkMD(Kind::Bdev),
          bdevName(std::move(bdev_name)) {}

    std::string bdevName;
    spdk_bdev_desc *desc = nullptr;
    spdk_bdev *bdev = nullptr;
    spdk_io_channel *channel = nullptr;
    uint32_t blockSize = 0;
    uint32_t writeUnitSize = 0;
    uint64_t numBlocks = 0;
};

// NVMe KV keys are 1..16 bytes.
inline constexpr std::size_t kNixlSpdkMaxKeyLen = 16;

// One object (key) on the KV device. Keys are per-object, as in NIXL's other
// OBJ backends: captured from the descriptor metaInfo at registration and
// recovered via metadataP at transfer. The KV device itself is the backend's
// bdev (see 'bdev_name'), owned by the progress engine.
class nixlSpdkObjMD : public nixlSpdkMD {
public:
    // Returns nullptr for an out-of-range key rather than truncating it.
    [[nodiscard]] static std::unique_ptr<nixlSpdkObjMD>
    create(std::string_view object_key) {
        if (object_key.empty() || object_key.size() > kNixlSpdkMaxKeyLen) {
            return nullptr;
        }
        return std::unique_ptr<nixlSpdkObjMD>(new nixlSpdkObjMD(object_key));
    }

    [[nodiscard]] std::span<const std::byte>
    key() const noexcept {
        return std::span(key_).first(keyLen_);
    }

private:
    explicit nixlSpdkObjMD(std::string_view object_key) noexcept
        : nixlSpdkMD(Kind::Obj),
          keyLen_(static_cast<uint8_t>(object_key.size())) {
        std::ranges::copy(std::as_bytes(std::span(object_key)), key_.begin());
    }

    std::array<std::byte, kNixlSpdkMaxKeyLen> key_{};
    uint8_t keyLen_;
};

// A transfer targets either a block range on a bdev or a key on the KV device.
// The offset lives in the BLK alternative because NVMe KV has no intra-object
// offset.
struct nixlSpdkBlkTarget {
    nixlSpdkBdevMD *md;
    uint64_t offset;
};

struct nixlSpdkObjTarget {
    nixlSpdkObjMD *md;
};

using nixlSpdkIoTarget = std::variant<nixlSpdkBlkTarget, nixlSpdkObjTarget>;

// One bdev I/O within a request.
struct nixlSpdkIoContext {
    nixlSpdkBackendReqH *reqH = nullptr;
    nixlSpdkProgressEngine *engine = nullptr;
    nixlSpdkDramMD *dram = nullptr;
    nixlSpdkIoTarget target;
    void *buf = nullptr; // value buffer (from the local DRAM descriptor)
    uint64_t nbytes = 0; // value size
    bool ioWaitQueued = false;
    spdk_bdev_io_wait_entry waitEntry = {};
};

class nixlSpdkBackendReqH : public nixlBackendReqH {
public:
    nixlSpdkBackendReqH(nixl_xfer_op_t operation, std::vector<nixlSpdkIoContext> ios);
    ~nixlSpdkBackendReqH() override = default;

    // Each IoContext holds its own address as the SPDK completion cb_arg, so a
    // copy would leave the duplicate's callbacks pointing into the original's
    // vector.
    nixlSpdkBackendReqH(const nixlSpdkBackendReqH &) = delete;
    nixlSpdkBackendReqH &
    operator=(const nixlSpdkBackendReqH &) = delete;

    [[nodiscard]] nixl_status_t
    status() const noexcept;

private:
    // The engine drives this handle's whole lifetime: submission, completion
    // accounting, and the delete protocol below.
    friend class nixlSpdkProgressEngine;

    // The handle can be freed from either the completion path (all I/Os done)
    // or the release path (cancelRequest). Both record their arrival in
    // lifeState_ via a single atomic fetch_or; the thread that observes the
    // other's bit already set is the second arriver and is the sole deleter.
    // This makes the two paths race-free without a post-store read of *this.
    // Typed to match the atomic below: fetch_or() requires an integral type.
    static constexpr uint32_t kSubmitted = 1u << 0; // entered the SPDK execution context
    static constexpr uint32_t kDone = 1u << 1; // all I/Os retired (success or error)
    static constexpr uint32_t kReleased = 1u << 2; // NIXL released/cancelled the request

    // Clear per-post state so a handle can be reposted. outstanding_ is primed
    // with one extra "submission guard" reference (released by submitRequest)
    // so a synchronous completion cannot retire the request mid-submit.
    void
    reset() noexcept;

    nixl_xfer_op_t operation_;
    std::vector<nixlSpdkIoContext> ios_;
    std::atomic<uint32_t> outstanding_{0};
    std::atomic<uint32_t> lifeState_{0};
    std::atomic<bool> cancelled_{false};
    std::atomic<nixl_status_t> overallStatus_{NIXL_IN_PROG};
};

class nixlSpdkEngine : public nixlBackendEngine {
public:
    explicit nixlSpdkEngine(const nixlBackendInitParams *init_params);
    ~nixlSpdkEngine() override;

    [[nodiscard]] bool
    supportsRemote() const noexcept override {
        return false;
    }

    [[nodiscard]] bool
    supportsLocal() const noexcept override {
        return true;
    }

    [[nodiscard]] bool
    supportsNotif() const noexcept override {
        return false;
    }

    [[nodiscard]] nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG, BLK_SEG, OBJ_SEG};
    }

    [[nodiscard]] nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    [[nodiscard]] nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    [[nodiscard]] nixl_status_t
    connect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    loadLocalMD(nixlBackendMD *in, nixlBackendMD *&out) override {
        out = in;
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    [[nodiscard]] nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    [[nodiscard]] nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    [[nodiscard]] nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    // queryMem is deliberately not overridden. The registration descriptor list
    // carries no resolved metadata, so there is no bdev or KV handle to consult;
    // the base class reports NIXL_ERR_NOT_SUPPORTED, which is more useful to a
    // caller than echoing back the length it just passed in.

private:
    std::unique_ptr<nixlSpdkProgressEngine> progress_;
};

#endif // NIXL_SRC_PLUGINS_SPDK_SPDK_BACKEND_H
