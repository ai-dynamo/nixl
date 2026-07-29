/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "spdk_backend.h"
#include "spdk_progress_engine.h"

#include <memory>

#include "common/nixl_log.h"

namespace {

[[nodiscard]] nixl_status_t
validateLists(nixl_xfer_op_t operation,
              const nixl_meta_dlist_t &local,
              const nixl_meta_dlist_t &remote) {
    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        NIXL_ERROR << "SPDK: unsupported operation " << operation;
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << "SPDK: local descriptor list must be DRAM_SEG";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote.getType() != BLK_SEG && remote.getType() != OBJ_SEG) {
        NIXL_ERROR << "SPDK: remote descriptor list must be BLK_SEG or OBJ_SEG";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.descCount() == 0 || local.descCount() != remote.descCount()) {
        NIXL_ERROR << "SPDK: descriptor count mismatch local=" << local.descCount()
                   << " remote=" << remote.descCount();
        return NIXL_ERR_INVALID_PARAM;
    }
    return NIXL_SUCCESS;
}

// The dynamic_cast is the NIXL boundary check that the metadata is ours; the
// kind tag drives the downcast from there.
[[nodiscard]] nixl_status_t
resolveDescriptorPair(const nixlMetaDesc &local,
                      const nixlMetaDesc &remote,
                      nixl_mem_t remote_type,
                      nixlSpdkDramMD *&dram_out,
                      nixlSpdkIoTarget &target_out) {
    auto *local_md = dynamic_cast<nixlSpdkMD *>(local.metadataP);
    if (!local_md || local_md->kind() != nixlSpdkMD::Kind::Dram) {
        NIXL_ERROR << "SPDK: local (DRAM) descriptor metadata is missing or invalid";
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *dram = static_cast<nixlSpdkDramMD *>(local_md);
    if (local.addr < dram->addr || (local.addr + local.len) > (dram->addr + dram->len)) {
        NIXL_ERROR << "SPDK: local descriptor is outside registered DRAM range";
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *remote_md = dynamic_cast<nixlSpdkMD *>(remote.metadataP);
    if (!remote_md) {
        NIXL_ERROR << "SPDK: remote descriptor metadata is missing or invalid";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.len != remote.len) {
        NIXL_ERROR << "SPDK: local and remote descriptor lengths must match";
        return NIXL_ERR_INVALID_PARAM;
    }

    // validateLists() checks the list type and this checks the metadata kind;
    // neither implies the other. Dispatching on the kind alone would let a
    // BLK_SEG list holding OBJ metadata issue an NVMe KV command.
    const nixlSpdkMD::Kind expected =
        (remote_type == BLK_SEG) ? nixlSpdkMD::Kind::Bdev : nixlSpdkMD::Kind::Obj;
    if (remote_md->kind() != expected) {
        NIXL_ERROR << "SPDK: remote descriptor metadata does not match the list type";
        return NIXL_ERR_INVALID_PARAM;
    }

    switch (remote_md->kind()) {
    case nixlSpdkMD::Kind::Bdev: {
        auto *bdev = static_cast<nixlSpdkBdevMD *>(remote_md);
        if (!bdev->desc || !bdev->channel || !bdev->bdev) {
            NIXL_ERROR << "SPDK: BLK descriptor metadata is missing or invalid";
            return NIXL_ERR_INVALID_PARAM;
        }
        if ((remote.addr % bdev->blockSize) != 0 || (remote.len % bdev->blockSize) != 0) {
            NIXL_ERROR << "SPDK: BLK descriptor offset and length must be block aligned";
            return NIXL_ERR_INVALID_PARAM;
        }
        if ((remote.addr + remote.len) > (bdev->numBlocks * bdev->blockSize)) {
            NIXL_ERROR << "SPDK: BLK descriptor exceeds bdev capacity";
            return NIXL_ERR_INVALID_PARAM;
        }
        if ((remote.len / bdev->blockSize) % bdev->writeUnitSize != 0) {
            NIXL_ERROR << "SPDK: BLK descriptor length is not write-unit aligned";
            return NIXL_ERR_INVALID_PARAM;
        }
        dram_out = dram;
        target_out = nixlSpdkBlkTarget{bdev, remote.addr};
        return NIXL_SUCCESS;
    }
    case nixlSpdkMD::Kind::Obj: {
        // NVMe KV has no intra-object offset, so the remote addr must be 0.
        auto *obj = static_cast<nixlSpdkObjMD *>(remote_md);
        // Defensive only: nixlSpdkObjMD::create() rejects an empty key and the
        // constructor is private, so this cannot fire today. Kept so a future
        // construction path cannot quietly emit a zero-length KV key.
        if (obj->key().empty()) {
            NIXL_ERROR << "SPDK: OBJ descriptor has an empty key";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (remote.addr != 0) {
            NIXL_ERROR << "SPDK: OBJ descriptor offset must be 0 (KV whole-value)";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (remote.len == 0) {
            NIXL_ERROR << "SPDK: OBJ descriptor value length must be non-zero";
            return NIXL_ERR_INVALID_PARAM;
        }
        dram_out = dram;
        target_out = nixlSpdkObjTarget{obj};
        return NIXL_SUCCESS;
    }
    case nixlSpdkMD::Kind::Dram:
        break;
    }

    NIXL_ERROR << "SPDK: remote descriptor list must be BLK_SEG or OBJ_SEG";
    return NIXL_ERR_INVALID_PARAM;
}

} // namespace

nixlSpdkBackendReqH::nixlSpdkBackendReqH(nixl_xfer_op_t operation,
                                         std::vector<nixlSpdkIoContext> ios)
    : operation_(operation),
      ios_(std::move(ios)) {
    // Each entry carries its own address as the SPDK completion cb_arg, so this
    // has to happen after the vector reaches its final storage.
    for (auto &io : ios_) {
        io.reqH = this;
        io.waitEntry.cb_arg = &io;
    }
}

nixl_status_t
nixlSpdkBackendReqH::status() const noexcept {
    if ((lifeState_.load(std::memory_order_acquire) & kDone) == 0) {
        return NIXL_IN_PROG;
    }
    return overallStatus_.load(std::memory_order_acquire);
}

void
nixlSpdkBackendReqH::reset() noexcept {
    outstanding_.store(static_cast<uint32_t>(ios_.size()) + 1, std::memory_order_relaxed);
    overallStatus_.store(NIXL_IN_PROG, std::memory_order_relaxed);
    cancelled_.store(false, std::memory_order_relaxed);
    for (auto &io : ios_) {
        io.ioWaitQueued = false;
    }
    lifeState_.store(0, std::memory_order_release);
}

nixlSpdkEngine::nixlSpdkEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      progress_(std::make_unique<nixlSpdkProgressEngine>(init_params)) {
    initErr = progress_->hasInitError();
}

nixlSpdkEngine::~nixlSpdkEngine() = default;

nixl_status_t
nixlSpdkEngine::registerMem(const nixlBlobDesc &mem,
                            const nixl_mem_t &mem_type,
                            nixlBackendMD *&out) {
    out = nullptr;
    if (!progress_ || progress_->hasInitError()) {
        return NIXL_ERR_BACKEND;
    }

    if (mem_type == DRAM_SEG) {
        auto md = std::make_unique<nixlSpdkDramMD>(mem.addr, mem.len);
        const nixl_status_t status = progress_->registerDram(*md);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        out = md.release();
        return NIXL_SUCCESS;
    }

    if (mem_type == BLK_SEG) {
        if (mem.metaInfo.empty()) {
            NIXL_ERROR << "SPDK: BLK_SEG metaInfo must contain an SPDK bdev name";
            return NIXL_ERR_INVALID_PARAM;
        }
        auto md = std::make_unique<nixlSpdkBdevMD>(mem.metaInfo);
        const nixl_status_t status = progress_->openBdev(*md);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        out = md.release();
        return NIXL_SUCCESS;
    }

    if (mem_type == OBJ_SEG) {
        // The key must be given explicitly, as BLK_SEG requires its bdev name.
        // Deriving one from devId would silently collide: two descriptors that
        // share a devId and omit metaInfo would map to the same object, and the
        // second write would overwrite the first.
        if (mem.metaInfo.empty()) {
            NIXL_ERROR << "SPDK: OBJ_SEG requires the object key in the descriptor metaInfo";
            return NIXL_ERR_INVALID_PARAM;
        }
        auto md = nixlSpdkObjMD::create(mem.metaInfo);
        if (!md) {
            NIXL_ERROR << "SPDK: OBJ_SEG key must be 1.." << kNixlSpdkMaxKeyLen
                       << " bytes (NVMe KV limit), got " << mem.metaInfo.size();
            return NIXL_ERR_INVALID_PARAM;
        }
        const nixl_status_t status = progress_->ensureKvBdev();
        if (status != NIXL_SUCCESS) {
            return status;
        }
        out = md.release();
        return NIXL_SUCCESS;
    }

    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlSpdkEngine::deregisterMem(nixlBackendMD *meta) {
    // Exhaustive switch: a new Kind fails to compile rather than silently
    // falling through to an error.
    auto *md = dynamic_cast<nixlSpdkMD *>(meta);
    if (!md) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::unique_ptr<nixlSpdkMD> owned(md);

    switch (md->kind()) {
    case nixlSpdkMD::Kind::Dram:
        return progress_->deregisterDram(static_cast<nixlSpdkDramMD &>(*md));
    case nixlSpdkMD::Kind::Bdev:
        progress_->closeBdev(static_cast<nixlSpdkBdevMD &>(*md));
        return NIXL_SUCCESS;
    case nixlSpdkMD::Kind::Obj:
        // The shared KV device is owned by the progress engine and torn down at
        // finalization; per-object metadata carries only the key.
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_INVALID_PARAM;
}

nixl_status_t
nixlSpdkEngine::prepXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *) const {
    handle = nullptr;
    const nixl_status_t status = validateLists(operation, local, remote);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    std::vector<nixlSpdkIoContext> ios;
    ios.reserve(local.descCount());
    for (int i = 0; i < local.descCount(); ++i) {
        nixlSpdkDramMD *dram = nullptr;
        nixlSpdkIoTarget target;
        const nixl_status_t resolved =
            resolveDescriptorPair(local[i], remote[i], remote.getType(), dram, target);
        if (resolved != NIXL_SUCCESS) {
            return resolved;
        }
        ios.push_back(nixlSpdkIoContext{
            .dram = dram,
            .target = target,
            .buf = reinterpret_cast<void *>(local[i].addr),
            .nbytes = local[i].len,
        });
    }

    handle = std::make_unique<nixlSpdkBackendReqH>(operation, std::move(ios)).release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlSpdkEngine::postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        nixl_status_t status = prepXfer(operation, local, remote, remote_agent, handle, opt_args);
        if (status != NIXL_SUCCESS) {
            return status;
        }
    }

    auto *req_h = dynamic_cast<nixlSpdkBackendReqH *>(handle);
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return progress_->postXfer(req_h);
}

nixl_status_t
nixlSpdkEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *req_h = dynamic_cast<nixlSpdkBackendReqH *>(handle);
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return progress_->checkXfer(req_h);
}

nixl_status_t
nixlSpdkEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *req_h = dynamic_cast<nixlSpdkBackendReqH *>(handle);
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }
    progress_->cancelRequest(req_h);
    return NIXL_SUCCESS;
}
