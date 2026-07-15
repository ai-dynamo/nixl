/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_RAIL_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_RAIL_H

#include <vector>
#include <deque>
#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <ostream>
#include <stack>

#include "nixl.h"
#include "backend/backend_aux.h"
#include "libfabric/libfabric_common.h"

// Forward declarations
class nixlLibfabricConnection;

/**
 * @brief Request structure for libfabric operations
 *
 */
struct nixlLibfabricReq {
    fi_context ctx; ///< Libfabric context for operation tracking
    size_t rail_id; ///< Rail ID that owns this request
    size_t pool_index; ///< Index in the pool for deque compatibility
    uint32_t xfer_id; ///< Pre-assigned globally unique transfer ID
    void *buffer; ///< Pre-assigned buffer for CONTROL operations, nullptr for DATA
    struct fid_mr *mr; ///< Pre-assigned memory registration for CONTROL, nullptr for DATA
    size_t buffer_size; ///< Pre-assigned buffer size for CONTROL (2KB), 0 for DATA

    enum OpType { WRITE, READ, SEND, RECV } operation_type; ///< Operation type (pre-assigned)

    bool in_use; ///< Pool management flag
    size_t chunk_offset; ///< Chunk offset for DATA requests
    size_t chunk_size; ///< Chunk size for DATA requests
    std::function<void()> completion_callback; ///< Completion callback function
    void *local_addr; ///< Local memory address for transfers
    uint64_t remote_addr; ///< Remote memory address for transfers
    struct fid_mr *local_mr; ///< Local memory registration for transfers
    uint64_t remote_key; ///< Remote access key for transfers

    /** Default constructor initializing all fields */
    nixlLibfabricReq()
        : rail_id(0),
          pool_index(0),
          xfer_id(0),
          buffer(nullptr),
          mr(nullptr),
          buffer_size(0),
          operation_type(SEND),
          in_use(false),
          chunk_offset(0),
          chunk_size(0),
          local_addr(nullptr),
          remote_addr(0),
          local_mr(nullptr),
          remote_key(0) {
        memset(&ctx, 0, sizeof(fi_context));
    }
};

/** Thread-safe request pool with O(1) allocation/release */
class RequestPool {
public:
    /** Initialize request pool with specified size */
    RequestPool(size_t pool_size, size_t rail_id);

    /** Virtual destructor for proper cleanup */
    virtual ~RequestPool() = default;

    /** Release request back to the pool */
    virtual void
    release(nixlLibfabricReq *req) const;

    /** Find request by libfabric context pointer */
    nixlLibfabricReq *
    findByContext(void *context) const;

    /** Get count of currently active requests */
    size_t
    getActiveRequestCount() const;

    /** Get pool utilization as percentage (0-100) */
    size_t
    getPoolUtilization() const;

    /** Expand pool by doubling its size - virtual method for subclass implementation */
    virtual nixl_status_t
    expandPool() = 0;

protected:
    /** Common allocation logic shared by both pool types */
    nixlLibfabricReq *
    allocateReq(uint32_t req_id);

public:
    // Non-copyable and non-movable since we use unique_ptr for management
    RequestPool(const RequestPool &) = delete;
    RequestPool &
    operator=(const RequestPool &) = delete;
    RequestPool(RequestPool &&) = delete;
    RequestPool &
    operator=(RequestPool &&) = delete;

protected:
    /** Initialize base pool structure with specified size */
    void
    initializeBasePool(size_t pool_size);

    mutable std::deque<nixlLibfabricReq> requests_; ///< Expandable request pool
    mutable std::stack<size_t> free_indices_; ///< Stack of available request indices
    size_t rail_id_; ///< Rail ID for this pool
    size_t initial_pool_size_; ///< Original pool size for expansion calculations
    mutable std::mutex pool_mutex_; ///< Thread safety protection
};

/** Buffer chunk structure for control request pool */
struct BufferChunk {
    void *buffer; ///< Buffer memory
    size_t size; ///< Buffer size
    struct fid_mr *mr; ///< Memory registration for this chunk
};

/** Control request pool with pre-allocated buffers for SEND/RECV operations */
class ControlRequestPool : public RequestPool {
public:
    /** Initialize control request pool */
    ControlRequestPool(size_t pool_size, size_t rail_id);

    /** Destructor with explicit cleanup */
    ~ControlRequestPool();

    // Non-copyable and non-movable since we use unique_ptr for management
    ControlRequestPool(const ControlRequestPool &) = delete;
    ControlRequestPool &
    operator=(const ControlRequestPool &) = delete;
    ControlRequestPool(ControlRequestPool &&) = delete;
    ControlRequestPool &
    operator=(ControlRequestPool &&) = delete;

    /** Initialize pool with buffers */
    nixl_status_t
    initialize(struct fid_domain *domain);

    /** Allocate control request with size validation */
    nixlLibfabricReq *
    allocate(size_t needed_size, uint32_t req_id);

    /** Expand pool by adding new buffer chunk - implements pure virtual */
    nixl_status_t
    expandPool() override;

    /** Explicit cleanup method for proper resource ordering */
    void
    cleanup();

private:
    /** Create new buffer chunk and register with libfabric */
    nixl_status_t
    createBufferChunk(size_t chunk_size, BufferChunk &chunk);

    std::vector<BufferChunk> buffer_chunks_; ///< Multiple buffer chunks for expansion
    struct fid_domain *domain_; ///< Domain for MR registration (stored during init)
    size_t chunk_size_; ///< Size of each buffer chunk
};

/** Lightweight data request pool for WRITE/READ operations */
class DataRequestPool : public RequestPool {
public:
    /** Initialize data request pool */
    DataRequestPool(size_t pool_size, size_t rail_id);

    /** Default destructor (no special cleanup needed) */
    ~DataRequestPool() = default;

    // Non-copyable and non-movable since we use unique_ptr for management
    DataRequestPool(const DataRequestPool &) = delete;
    DataRequestPool &
    operator=(const DataRequestPool &) = delete;
    DataRequestPool(DataRequestPool &&) = delete;
    DataRequestPool &
    operator=(DataRequestPool &&) = delete;

    /** Initialize pool */
    nixl_status_t
    initialize();

    /** Allocate data request for specified operation type */
    nixlLibfabricReq *
    allocate(nixlLibfabricReq::OpType op_type, uint32_t req_id);

    /** Expand pool by doubling request count - implements pure virtual */
    nixl_status_t
    expandPool() override;
};


/** Connection state tracking for multi-rail connections */
enum class ConnectionState {
    DISCONNECTED, ///< No connection attempt made, initial state
    CONNECTED ///< Ready for data transfers.
};

// Stream operator for ConnectionState to enable logging
inline std::ostream &
operator<<(std::ostream &os, const ConnectionState &state) {
    switch (state) {
    case ConnectionState::DISCONNECTED:
        return os << "DISCONNECTED";
    case ConnectionState::CONNECTED:
        return os << "CONNECTED";
    default:
        return os << "UNKNOWN";
    }
}

/**
 * @brief Lock-free, and mostly wait-free, Multi-Producer Single-Consumer ring buffer.
 * Multiple producer threads (postXfer) push concurrently.
 * Single consumer (progress thread) pops and posts.
 * Uses fetch-add on head for multi-producer safety and freedom to execute in parallel.
 */
template<typename T> class MpscRing {
public:
    MpscRing() : head_(0), tail_(0), capacity_(0), ring_(nullptr), committed_(nullptr) {}

    MpscRing(const MpscRing &) = delete;
    MpscRing(MpscRing &&) = delete;

    MpscRing &
    operator=(const MpscRing &) = delete;

    ~MpscRing() {
        destroy();
    }

    bool
    initialize(size_t capacity) {
        if (capacity == 0) {
            NIXL_ERROR << "Invalid ring buffer zero size";
            return false;
        }
        if (ring_ != nullptr) {
            NIXL_WARN << "Ring buffer already initialized, request to initialize ignored";
            return true;
        }
        if (committed_ != nullptr) {
            // this should never happen
            NIXL_ERROR << "Ring buffer in invalid state";
            return false;
        }
        ring_ = new (std::nothrow) T[capacity];
        if (ring_ == nullptr) {
            NIXL_ERROR << "Failed to allocate ring buffer of size " << capacity;
            return false;
        }
        committed_ = new (std::nothrow) std::atomic<bool>[capacity];
        if (committed_ == nullptr) {
            NIXL_ERROR << "Failed to allocated ring buffer committed array of size " << capacity;
            delete[] ring_;
            ring_ = nullptr;
            return false;
        }
        capacity_ = capacity;
        return true;
    }

    void
    destroy() {
        if (ring_ != nullptr) {
            delete[] ring_;
            ring_ = nullptr;
        }
        if (committed_ != nullptr) {
            delete[] committed_;
            committed_ = nullptr;
        }
        capacity_ = 0;
        head_ = 0;
        tail_ = 0;
    }

    void
    push(const T &item) {
        // NOTE: although this is a blocking call, as long as there is enough room in the ring
        // buffer, this is WAIT-FREE (as opposed to any other CAS-based solution)

        // grab a unique slot among all concurrent writers
        size_t absHead = head_.fetch_add(1, std::memory_order_relaxed);
        size_t absTail = tail_.load(std::memory_order_acquire);

        // wait until reader catches up
        // NOTE: capacity should be matched with expected insert load and reader's capability to
        // handle requests (should be large enough to handle the largest expected burst)
        while (absHead - absTail >= capacity_) {
            std::this_thread::yield();
            absTail = tail_.load(std::memory_order_acquire);
            // NOTE: we may add here some counters for reporting a full ring buffer
        }

        // head slot is now reserved for us
        size_t head = absHead % capacity_;
        ring_[head] = item;
        // Signal that this slot is written (use a separate committed array)
        committed_[head].store(true, std::memory_order_release);
    }

    bool
    pop(T &item) {
        size_t absTail = tail_.load(std::memory_order_relaxed);
        size_t absHead = head_.load(std::memory_order_acquire);
        if (absTail == absHead) {
            return false;
        }
        size_t tail = absTail % capacity_;
        // Wait for the slot to be committed (producer might still be writing)
        // NOTE: we can actually peek entries ahead if head > tail + 1, but that would require
        // skipping them later, so we need a third state for committed array, for now this
        // optimization is not implemented, unless we see reader having difficulties
        if (!committed_[tail].load(std::memory_order_acquire)) {
            // item is not ready, so back-off
            return false;
        }
        item = ring_[tail];
        committed_[tail].store(false, std::memory_order_relaxed);
        tail_.store((absTail + 1), std::memory_order_release);
        return true;
    }

    bool
    empty() const {
        // NOTE: we don't care about order here, as long as both head and tail are loaded by the
        // time we compare them
        size_t absTail = tail_.load(std::memory_order_relaxed);
        size_t absHead = head_.load(std::memory_order_relaxed);

        // NOTE: if reader is busy reading last committed entry and has not advanced yet the tail
        // pointer, we still consider this as not-empty
        return (absTail == absHead);
    }

private:
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    size_t capacity_;
    T *ring_;
    std::atomic<bool> *committed_;
};

/**
 * @brief Queued post request for PT-owns-endpoint model
 * Main thread enqueues; progress thread dequeues and posts.
 */
struct nixlLibfabricPostRequest {
    enum post_type { WRITE, READ, SEND } type;

    void *local_addr;
    size_t length;
    void *local_desc;
    uint64_t immediate_data; // WRITE only
    fi_addr_t dest_addr;
    uint64_t remote_addr;
    uint64_t remote_key;
    nixlLibfabricReq *req;
    uint64_t fi_flags;
    int device_id; // CUDA device ordinal for cudaSetDevice in PT
    bool is_cuda_vram;
};

/** Individual libfabric rail managing fabric, domain, endpoint, CQ, and AV */
class nixlLibfabricRail {
public:
    uint16_t rail_id; ///< Unique rail identifier
    std::string device_name; ///< EFA device name for this rail
    std::string provider_name; ///< Provider name (e.g., "efa", "efa-direct")
    char ep_name[LF_EP_NAME_MAX_LEN]; ///< Endpoint name for connection setup
    struct fid_ep *endpoint; ///< Libfabric endpoint handle

    /** Initialize libfabric rail with all resources */
    nixlLibfabricRail(const std::string &device, const std::string &provider, uint16_t id);

    /** Destroy rail and cleanup all libfabric resources */
    ~nixlLibfabricRail();

    // Non-copyable and non-movable since we use unique_ptr for management
    nixlLibfabricRail(const nixlLibfabricRail &) = delete;
    nixlLibfabricRail &
    operator=(const nixlLibfabricRail &) = delete;
    nixlLibfabricRail(nixlLibfabricRail &&) = delete;
    nixlLibfabricRail &
    operator=(nixlLibfabricRail &&) = delete;

    /** Explicit cleanup method for proper resource ordering */
    void
    cleanup();

    /** Validate that rail is properly initialized */
    bool
    isProperlyInitialized() const;

    // Memory registration methods
    /** Register memory buffer with libfabric */
    nixl_status_t
    registerMemory(void *buffer,
                   size_t length,
                   nixl_mem_t mem_type,
                   int device_id,
                   enum fi_hmem_iface iface,
                   struct fid_mr **mr_out,
                   uint64_t *key_out) const;

    /** Deregister memory from libfabric */
    nixl_status_t
    deregisterMemory(struct fid_mr *mr) const;

    // Address vector management methods
    /** Insert remote endpoint address into address vector */
    nixl_status_t
    insertAddress(const void *addr, fi_addr_t *fi_addr_out) const;

    /** Remove address from address vector */
    nixl_status_t
    removeAddress(fi_addr_t fi_addr) const;

    // Memory descriptor helper methods
    /** Get libfabric memory descriptor for MR */
    void *
    getMemoryDescriptor(struct fid_mr *mr) const;

    /** Get remote access key for MR */
    uint64_t
    getMemoryKey(struct fid_mr *mr) const;

    // Libfabric operation wrappers
    /** Post receive operation */
    nixl_status_t
    postRecv(nixlLibfabricReq *req) const;

    /** Post send operation with immediate data */
    nixl_status_t
    postSend(uint64_t immediate_data, fi_addr_t dest_addr, nixlLibfabricReq *req);

    /** Post RDMA write operation with immediate data */
    nixl_status_t
    postWrite(const void *local_buffer,
              size_t length,
              void *local_desc,
              uint64_t immediate_data,
              fi_addr_t dest_addr,
              uint64_t remote_addr,
              uint64_t remote_key,
              nixlLibfabricReq *req,
              uint64_t fi_flags = 0);

    /** Post RDMA read operation */
    nixl_status_t
    postRead(void *local_buffer,
             size_t length,
             void *local_desc,
             fi_addr_t dest_addr,
             uint64_t remote_addr,
             uint64_t remote_key,
             nixlLibfabricReq *req);

    /** Process completion queue with batching support */
    nixl_status_t
    progressCompletionQueue();

    /** Enqueue a post request for the progress thread to execute */
    void
    enqueuePost(const nixlLibfabricPostRequest &req);

    /** Drain pending posts (called by progress thread) */
    nixl_status_t
    drainPostQueue();

    // Callback registration methods
    /** Set callback for notification message processing.
     *  Signature: (serialized_notif, sender_agent_idx_in_our_table).
     *  sender_agent_idx is the agent_idx field decoded from the fi_senddata's
     *  imm_data — the sender's local index in our agent_names_ table, which
     *  the sender learned via NIXL_LIBFABRIC_MSG_HANDSHAKE. */
    void
    setNotificationCallback(std::function<void(const std::string &, uint16_t)> callback);

    /** Set callback for handshake-message processing.
     *  Called with the serialized handshake payload on receipt of a
     *  NIXL_LIBFABRIC_MSG_HANDSHAKE control message. */
    void
    setHandshakeCallback(std::function<void(const std::string &)> callback);

    /**
     * @brief Enable or disable a progress thread that handles CQ draining.
     *
     * @param enabled  true to enable progress-thread CQ draining,
     *                 false to disable it.
     */
    void
    setProgressThreadEnabled(bool enabled);

    /** Check if progress thread is enabled for this rail */
    bool
    isProgressThreadEnabled() const {
        return progress_thread_enabled_;
    }

    /**
     * @brief Initializes the post transfer queue (lokc-free ring buffer) for deferring requests
     * when the progress thread is enabled.
     *
     * @param post_queue_size The size of the post transfer queue (lock-free ring buffer) size.
     *
     * @return True if initialization succeeded, otherwise false.
     */
    bool
    initPostQueue(size_t post_queue_size);

    /** Set callback for XFER_ID tracking.
     *  Signature: (imm_data, sender_agent_idx_in_our_table). sender_agent_idx
     *  is decoded from the imm_data the sender shipped with fi_writedata,
     *  via the handshake-negotiated agent_idx encoding. */
    void
    setXferIdCallback(std::function<void(uint64_t, uint16_t)> callback);

    // Optimized resource management methods
    /** Allocate control request with size validation */
    [[nodiscard]] nixlLibfabricReq *
    allocateControlRequest(size_t needed_size, uint32_t req_id) const;

    /** Allocate data request for specified operation */
    [[nodiscard]] nixlLibfabricReq *
    allocateDataRequest(nixlLibfabricReq::OpType op_type, uint32_t req_id) const;

    /** Release request back to appropriate pool */
    void
    releaseRequest(nixlLibfabricReq *req) const;

    /** Find request from libfabric context pointer */
    nixlLibfabricReq *
    findRequestFromContext(void *context) const;

    fi_info *
    getRailInfo() const;

private:
    // Core libfabric resources
    struct fi_info *info; // from rail_infos[rail_id]
    struct fid_fabric *fabric; // from rail_fabrics[rail_id]
    struct fid_domain *domain; // from rail_domains[rail_id]
    struct fid_cq *cq; // from rail_cqs[rail_id]
    struct fid_av *av; // from rail_avs[rail_id]

    // EP mutex to protect endpoint and CQ operations
    mutable std::mutex ep_mutex_;

    // Lock-free MPSC ring: main thread pushes, PT pops and posts
    using post_ring_t = MpscRing<nixlLibfabricPostRequest>;
    post_ring_t post_ring_;

    // the post/poll interleave ratio constants
    static constexpr size_t POST_MAX_BATCH_SIZE = 256;
    static constexpr size_t POST_POLL_INTERLEAVE_RATIO = 32;

    // Whether a progress thread is handling CQ draining
    bool progress_thread_enabled_ = false;

    // Callback functions. Receive completions carry the sender's agent_idx
    // (its index in OUR table, as told to it via the handshake) decoded from
    // the imm_data the sender shipped.
    std::function<void(const std::string &, uint16_t)> notificationCallback;
    std::function<void(uint64_t, uint16_t)> xferIdCallback;
    std::function<void(const std::string &)> handshakeCallback;

    // Separate request pools for optimal performance
    ControlRequestPool control_request_pool_;
    DataRequestPool data_request_pool_;

    // Provider capability flags
    bool provider_supports_hmem_;


    nixl_status_t
    processCompletionQueueEntry(struct fi_cq_data_entry *comp) const;
    nixl_status_t
    processLocalSendCompletion(struct fi_cq_data_entry *comp) const;
    nixl_status_t
    processLocalTransferCompletion(struct fi_cq_data_entry *comp, const char *operation_type) const;
    nixl_status_t
    processRecvCompletion(struct fi_cq_data_entry *comp) const;
    nixl_status_t
    processRemoteWriteCompletion(struct fi_cq_data_entry *comp) const;
};


#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_RAIL_H
