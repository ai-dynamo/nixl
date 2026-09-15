/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "daos_client.h"

#include <daos_event.h>
#include <daos_fs.h>
#include <daos_obj_class.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "common/backend.h"
#include "common/nixl_log.h"

namespace {

std::string
normalizePath(std::string_view input) {
    while (!input.empty() && input.front() == '/') {
        input.remove_prefix(1);
    }
    while (!input.empty() && input.back() == '/') {
        input.remove_suffix(1);
    }
    if (input.empty()) {
        throw std::invalid_argument("DAOS object path must not be empty");
    }

    size_t begin = 0;
    while (begin <= input.size()) {
        const size_t end = input.find('/', begin);
        const auto component = input.substr(begin, end - begin);
        if (component.empty() || component == "." || component == "..") {
            throw std::invalid_argument("DAOS object path contains an invalid component");
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return "/" + std::string(input);
}

std::vector<std::optional<unsigned>>
parseCpuAffinity(std::string_view input, size_t num_lanes) {
    if (input.empty()) {
        return std::vector<std::optional<unsigned>>(num_lanes);
    }

    std::vector<std::optional<unsigned>> cpus;
    size_t begin = 0;
    while (begin <= input.size()) {
        const size_t end = input.find(',', begin);
        std::string_view token = input.substr(begin, end - begin);
        while (!token.empty() && token.front() == ' ') {
            token.remove_prefix(1);
        }
        while (!token.empty() && token.back() == ' ') {
            token.remove_suffix(1);
        }

        unsigned cpu = 0;
        const auto [ptr, error] = std::from_chars(token.data(), token.data() + token.size(), cpu);
        if (token.empty() || error != std::errc() || ptr != token.data() + token.size()) {
            throw std::invalid_argument(
                "progress_cpu_affinity must be a comma-separated list of CPU IDs");
        }
        cpus.emplace_back(cpu);
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    if (cpus.size() != num_lanes) {
        throw std::invalid_argument(
            "progress_cpu_affinity must contain one CPU ID per event queue");
    }
    return cpus;
}

class libDfsObject final : public iDfsObject {
public:
    explicit libDfsObject(dfs_obj_t *object) : object_(object) {}

    ~libDfsObject() override {
        if (object_) {
            const int rc = dfs_release(object_);
            if (rc != 0) {
                NIXL_ERROR << "dfs_release failed: " << rc;
            }
        }
    }

    dfs_obj_t *
    get() const {
        return object_;
    }

private:
    dfs_obj_t *object_;
};

class scopedDfsObject {
public:
    ~scopedDfsObject() {
        if (object_) {
            dfs_release(object_);
        }
    }

    dfs_obj_t **
    out() {
        return &object_;
    }

    dfs_obj_t *
    get() const {
        return object_;
    }

    dfs_obj_t *
    release() {
        return std::exchange(object_, nullptr);
    }

private:
    dfs_obj_t *object_ = nullptr;
};

class libDfsClient final : public iDfsClient {
private:
    struct pendingIo {
        daos_event_t event{};
        d_iov_t iov{};
        d_sg_list_t sgl{};
        daos_size_t readSize = 0;
        size_t expectedSize = 0;
        uint64_t offset = 0;
        bool read = false;
        std::shared_ptr<iDfsObject> object;
        completion_t completion;
    };

    class eventQueueLane {
    public:
        eventQueueLane(size_t lane_id,
                       dfs_t *dfs,
                       size_t max_inflight,
                       size_t submission_batch_size,
                       size_t completion_batch_size,
                       int64_t poll_timeout_us,
                       std::optional<unsigned> cpu)
            : laneId_(lane_id),
              dfs_(dfs),
              maxInflight_(max_inflight),
              submissionBatchSize_(submission_batch_size),
              pollTimeoutUs_(poll_timeout_us),
              cpu_(cpu),
              completed_(completion_batch_size) {
            const int rc = daos_eq_create(&eq_);
            if (rc != 0) {
                throw std::runtime_error("daos_eq_create failed: " + std::to_string(rc));
            }
            eqCreated_ = true;
            try {
                thread_ = std::thread(&eventQueueLane::run, this);
            }
            catch (...) {
                daos_eq_destroy(eq_, 0);
                eqCreated_ = false;
                throw;
            }
        }

        eventQueueLane(const eventQueueLane &) = delete;
        eventQueueLane &
        operator=(const eventQueueLane &) = delete;

        ~eventQueueLane() {
            stopping_.store(true, std::memory_order_release);
            queueCv_.notify_all();
            if (thread_.joinable()) {
                thread_.join();
            }
            if (eqCreated_) {
                const int rc = daos_eq_destroy(eq_, 0);
                if (rc != 0) {
                    NIXL_ERROR << "daos_eq_destroy failed for lane " << laneId_ << ": " << rc;
                }
            }
        }

        int
        enqueue(std::unique_ptr<pendingIo> &io) {
            if (stopping_.load(std::memory_order_acquire)) {
                return ESHUTDOWN;
            }
            size_t outstanding = outstanding_.load(std::memory_order_relaxed);
            do {
                if (outstanding >= maxInflight_) {
                    return EAGAIN;
                }
            } while (!outstanding_.compare_exchange_weak(outstanding,
                                                         outstanding + 1,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_relaxed));

            try {
                std::lock_guard<std::mutex> lock(queueMutex_);
                submissions_.push_back(std::move(io));
            }
            catch (...) {
                outstanding_.fetch_sub(1, std::memory_order_release);
                throw;
            }
            queueCv_.notify_one();
            return 0;
        }

    private:
        void
        bindToCpu() {
            if (!cpu_) {
                return;
            }
#if defined(__linux__)
            if (*cpu_ >= CPU_SETSIZE) {
                NIXL_ERROR << "DAOS progress CPU " << *cpu_ << " is outside CPU_SETSIZE";
                return;
            }
            cpu_set_t cpu_set;
            CPU_ZERO(&cpu_set);
            CPU_SET(*cpu_, &cpu_set);
            const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
            if (rc != 0) {
                NIXL_ERROR << "failed to bind DAOS progress lane " << laneId_ << " to CPU " << *cpu_
                           << ": " << rc;
            } else {
                NIXL_INFO << "bound DAOS progress lane " << laneId_ << " to CPU " << *cpu_;
            }
#else
            NIXL_ERROR << "progress_cpu_affinity is only supported on Linux; lane " << laneId_
                       << " remains unbound";
#endif
        }

        std::unique_ptr<pendingIo>
        popSubmission() {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (submissions_.empty()) {
                return nullptr;
            }
            auto io = std::move(submissions_.front());
            submissions_.pop_front();
            return io;
        }

        bool
        queueEmpty() {
            std::lock_guard<std::mutex> lock(queueMutex_);
            return submissions_.empty();
        }

        void
        finish(std::unique_ptr<pendingIo> io, int rc, size_t bytes_read) {
            outstanding_.fetch_sub(1, std::memory_order_release);
            io->completion(rc, bytes_read);
        }

        void
        submitOne(std::unique_ptr<pendingIo> io) {
            int rc = daos_event_init(&io->event, eq_, nullptr);
            if (rc != 0) {
                finish(std::move(io), rc, 0);
                return;
            }

            pendingIo *raw = io.get();
            try {
                pending_.emplace(&raw->event, raw);
            }
            catch (...) {
                const int fini_rc = daos_event_fini(&raw->event);
                if (fini_rc != 0) {
                    NIXL_ERROR << "daos_event_fini failed after allocation failure: " << fini_rc;
                }
                finish(std::move(io), ENOMEM, 0);
                return;
            }
            io.release();

            const auto object = std::static_pointer_cast<libDfsObject>(raw->object);
            if (raw->read) {
                rc = dfs_read(
                    dfs_, object->get(), &raw->sgl, raw->offset, &raw->readSize, &raw->event);
            } else {
                rc = dfs_write(dfs_, object->get(), &raw->sgl, raw->offset, &raw->event);
            }
            if (rc != 0) {
                std::unique_ptr<pendingIo> failed(pending_.at(&raw->event));
                pending_.erase(&raw->event);
                const int fini_rc = daos_event_fini(&failed->event);
                if (fini_rc != 0) {
                    NIXL_ERROR << "daos_event_fini after submission failure failed: " << fini_rc;
                }
                finish(std::move(failed), rc, 0);
            }
        }

        void
        complete(daos_event_t *event) {
            auto it = pending_.find(event);
            if (it == pending_.end()) {
                NIXL_ERROR << "DAOS EQ lane " << laneId_ << " returned an unknown event";
                return;
            }
            std::unique_ptr<pendingIo> io(it->second);
            pending_.erase(it);

            const int event_rc = io->event.ev_error;
            const size_t bytes_read = io->read ? io->readSize : io->expectedSize;
            const int fini_rc = daos_event_fini(&io->event);
            if (fini_rc != 0) {
                NIXL_ERROR << "daos_event_fini failed: " << fini_rc;
            }
            finish(std::move(io), event_rc != 0 ? event_rc : fini_rc, bytes_read);
        }

        void
        run() {
            bindToCpu();
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(queueMutex_);
                    queueCv_.wait(lock, [this] {
                        return stopping_.load(std::memory_order_acquire) || !submissions_.empty() ||
                            !pending_.empty();
                    });
                    if (stopping_.load(std::memory_order_acquire) && submissions_.empty() &&
                        pending_.empty()) {
                        break;
                    }
                }

                for (size_t i = 0; i < submissionBatchSize_; ++i) {
                    auto io = popSubmission();
                    if (!io) {
                        break;
                    }
                    submitOne(std::move(io));
                }

                if (!pending_.empty()) {
                    const int64_t timeout = queueEmpty() ? pollTimeoutUs_ : DAOS_EQ_NOWAIT;
                    const int count = daos_eq_poll(eq_,
                                                   1,
                                                   timeout,
                                                   static_cast<unsigned>(completed_.size()),
                                                   completed_.data());
                    if (count < 0) {
                        NIXL_ERROR << "daos_eq_poll failed for lane " << laneId_ << ": " << count;
                        std::this_thread::yield();
                    } else {
                        for (int i = 0; i < count; ++i) {
                            complete(completed_[i]);
                        }
                    }
                }
            }
        }

        size_t laneId_;
        dfs_t *dfs_;
        size_t maxInflight_;
        size_t submissionBatchSize_;
        int64_t pollTimeoutUs_;
        std::optional<unsigned> cpu_;
        daos_handle_t eq_ = DAOS_HDL_INVAL;
        bool eqCreated_ = false;
        std::atomic<bool> stopping_{false};
        std::atomic<size_t> outstanding_{0};
        std::thread thread_;
        std::mutex queueMutex_;
        std::condition_variable queueCv_;
        std::deque<std::unique_ptr<pendingIo>> submissions_;
        std::unordered_map<daos_event_t *, pendingIo *> pending_;
        std::vector<daos_event_t *> completed_;
    };

public:
    explicit libDfsClient(const nixl_b_params_t *custom_params)
        : pool_(nixl::getBackendParamDefaulted(custom_params, "pool", std::string())),
          container_(nixl::getBackendParamDefaulted(custom_params, "container", std::string())),
          system_(nixl::getBackendParamDefaulted(custom_params, "system", std::string())),
          readOnly_(nixl::getBackendParamDefaulted(custom_params, "read_only", false)),
          createContainer_(
              nixl::getBackendParamDefaulted(custom_params, "create_container", false)),
          chunkSize_(nixl::getBackendParamDefaulted(custom_params, "chunk_size", uint64_t{0})),
          oclassId_(nixl::getBackendParamDefaulted(custom_params, "oclass_id", uint32_t{0})),
          objectClass_(
              nixl::getBackendParamDefaulted(custom_params, "object_class", std::string())),
          objectClassHint_(
              nixl::getBackendParamDefaulted(custom_params, "object_class_hint", std::string())),
          numEventQueues_(
              nixl::getBackendParamDefaulted(custom_params, "num_event_queues", size_t{1})),
          maxInflightPerQueue_(nixl::getBackendParamDefaulted(custom_params,
                                                              "max_inflight_per_queue",
                                                              size_t{1024})),
          submissionBatchSize_(
              nixl::getBackendParamDefaulted(custom_params, "submission_batch_size", size_t{32})),
          completionBatchSize_(
              nixl::getBackendParamDefaulted(custom_params, "completion_batch_size", size_t{128})),
          pollTimeoutUs_(nixl::getBackendParamDefaulted(custom_params,
                                                        "progress_poll_timeout_us",
                                                        int64_t{1000})) {
        if (pool_.empty()) {
            throw std::invalid_argument("DAOS backend requires parameter 'pool'");
        }
        if (container_.empty()) {
            throw std::invalid_argument("DAOS backend requires parameter 'container'");
        }
        if (readOnly_ && createContainer_) {
            throw std::invalid_argument("create_container cannot be used with read_only");
        }
        const unsigned layout_selectors = !objectClass_.empty() + !objectClassHint_.empty() +
            static_cast<unsigned>(oclassId_ != 0);
        if (layout_selectors > 1) {
            throw std::invalid_argument(
                "set only one of object_class, object_class_hint, or oclass_id");
        }
        if (!objectClass_.empty()) {
            const int class_id = daos_oclass_name2id(objectClass_.c_str());
            if (class_id == OC_UNKNOWN) {
                throw std::invalid_argument("unknown DAOS object class '" + objectClass_ + "'");
            }
            oclassId_ = static_cast<uint32_t>(class_id);
        }
        if (numEventQueues_ == 0 || maxInflightPerQueue_ == 0 || submissionBatchSize_ == 0 ||
            completionBatchSize_ == 0 || pollTimeoutUs_ <= 0) {
            throw std::invalid_argument(
                "DAOS queue and batch parameters must be greater than zero");
        }
        const auto affinity = parseCpuAffinity(
            nixl::getBackendParamDefaulted(custom_params, "progress_cpu_affinity", std::string()),
            numEventQueues_);

        int rc = dfs_init();
        if (rc != 0) {
            throw std::runtime_error("dfs_init failed: " + std::to_string(rc));
        }
        initialized_ = true;

        int flags = readOnly_ ? O_RDONLY : O_RDWR;
        if (createContainer_) {
            flags |= O_CREAT;
        }
        rc = dfs_connect(pool_.c_str(),
                         system_.empty() ? nullptr : system_.c_str(),
                         container_.c_str(),
                         flags,
                         nullptr,
                         &dfs_);
        if (rc != 0) {
            cleanupDfs();
            throw std::runtime_error("dfs_connect failed: " + std::to_string(rc));
        }
        if (!objectClassHint_.empty()) {
            daos_oclass_id_t suggested = OC_UNKNOWN;
            rc = dfs_suggest_oclass(dfs_, objectClassHint_.c_str(), &suggested);
            if (rc != 0 || suggested == OC_UNKNOWN) {
                cleanupDfs();
                throw std::runtime_error("dfs_suggest_oclass failed for '" + objectClassHint_ +
                                         "': " + std::to_string(rc));
            }
            oclassId_ = suggested;
        }

        try {
            lanes_.reserve(numEventQueues_);
            for (size_t i = 0; i < numEventQueues_; ++i) {
                lanes_.push_back(std::make_unique<eventQueueLane>(i,
                                                                  dfs_,
                                                                  maxInflightPerQueue_,
                                                                  submissionBatchSize_,
                                                                  completionBatchSize_,
                                                                  pollTimeoutUs_,
                                                                  affinity[i]));
            }
        }
        catch (...) {
            lanes_.clear();
            cleanupDfs();
            throw;
        }
    }

    ~libDfsClient() override {
        // Lane destructors drain queued and in-flight operations before their
        // event queues are destroyed.
        lanes_.clear();
        cleanupDfs();
    }

    int
    open(std::string_view input_path, bool write, std::shared_ptr<iDfsObject> &object) override {
        object.reset();
        if (write && readOnly_) {
            return EROFS;
        }

        std::string path;
        try {
            path = normalizePath(input_path);
        }
        catch (const std::invalid_argument &) {
            return EINVAL;
        }

        scopedDfsObject opened;
        int rc;
        if (!write) {
            rc = dfs_lookup(dfs_, path.c_str(), O_RDONLY, opened.out(), nullptr, nullptr);
        } else {
            const size_t slash = path.rfind('/');
            const std::string parent_path =
                slash == std::string::npos ? std::string() : path.substr(0, slash);
            const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);

            scopedDfsObject parent;
            if (!parent_path.empty()) {
                rc = dfs_lookup(dfs_, parent_path.c_str(), O_RDWR, parent.out(), nullptr, nullptr);
                if (rc != 0) {
                    return rc;
                }
            }
            rc = dfs_open(dfs_,
                          parent.get(),
                          name.c_str(),
                          S_IFREG | S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                          O_RDWR | O_CREAT,
                          static_cast<daos_oclass_id_t>(oclassId_),
                          chunkSize_,
                          nullptr,
                          opened.out());
        }
        if (rc != 0) {
            return rc;
        }
        object = std::make_shared<libDfsObject>(opened.release());
        return 0;
    }

    int
    submitRead(const std::shared_ptr<iDfsObject> &object,
               uintptr_t data_ptr,
               size_t data_len,
               uint64_t offset,
               completion_t completion) override {
        return submit(object, data_ptr, data_len, offset, true, std::move(completion));
    }

    int
    submitWrite(const std::shared_ptr<iDfsObject> &object,
                uintptr_t data_ptr,
                size_t data_len,
                uint64_t offset,
                completion_t completion) override {
        if (readOnly_) {
            return EROFS;
        }
        return submit(object, data_ptr, data_len, offset, false, std::move(completion));
    }

    int
    exists(std::string_view input_path, bool &result) override {
        result = false;
        std::string path;
        try {
            path = normalizePath(input_path);
        }
        catch (const std::invalid_argument &) {
            return EINVAL;
        }

        scopedDfsObject object;
        const int rc = dfs_lookup(dfs_, path.c_str(), O_RDONLY, object.out(), nullptr, nullptr);
        if (rc == ENOENT) {
            return 0;
        }
        if (rc != 0) {
            return rc;
        }
        result = true;
        return 0;
    }

private:
    int
    submit(const std::shared_ptr<iDfsObject> &object,
           uintptr_t data_ptr,
           size_t data_len,
           uint64_t offset,
           bool read,
           completion_t completion) {
        if (!std::dynamic_pointer_cast<libDfsObject>(object) || !completion ||
            (data_len != 0 && data_ptr == 0) ||
            data_len > std::numeric_limits<daos_size_t>::max()) {
            return EINVAL;
        }
        if (data_len == 0) {
            completion(0, 0);
            return 0;
        }

        auto io = std::make_unique<pendingIo>();
        io->expectedSize = data_len;
        io->offset = offset;
        io->read = read;
        io->object = object;
        io->completion = std::move(completion);
        d_iov_set(&io->iov, reinterpret_cast<void *>(data_ptr), data_len);
        io->sgl.sg_nr = 1;
        io->sgl.sg_nr_out = 1;
        io->sgl.sg_iovs = &io->iov;

        const size_t first_lane = nextLane_.fetch_add(1, std::memory_order_relaxed) % lanes_.size();
        for (size_t i = 0; i < lanes_.size(); ++i) {
            const int rc = lanes_[(first_lane + i) % lanes_.size()]->enqueue(io);
            if (rc != EAGAIN) {
                return rc;
            }
        }
        return EAGAIN;
    }

    void
    cleanupDfs() {
        if (dfs_) {
            const int rc = dfs_disconnect(dfs_);
            if (rc != 0) {
                NIXL_ERROR << "dfs_disconnect failed: " << rc;
            }
            dfs_ = nullptr;
        }
        if (initialized_) {
            const int rc = dfs_fini();
            if (rc != 0) {
                NIXL_ERROR << "dfs_fini failed: " << rc;
            }
            initialized_ = false;
        }
    }

    std::string pool_;
    std::string container_;
    std::string system_;
    bool readOnly_;
    bool createContainer_;
    uint64_t chunkSize_;
    uint32_t oclassId_;
    std::string objectClass_;
    std::string objectClassHint_;
    size_t numEventQueues_;
    size_t maxInflightPerQueue_;
    size_t submissionBatchSize_;
    size_t completionBatchSize_;
    int64_t pollTimeoutUs_;
    bool initialized_ = false;
    dfs_t *dfs_ = nullptr;
    std::atomic<size_t> nextLane_{0};
    std::vector<std::unique_ptr<eventQueueLane>> lanes_;
};

} // namespace

std::shared_ptr<iDfsClient>
makeLibDfsClient(const nixl_b_params_t *custom_params) {
    return std::make_shared<libDfsClient>(custom_params);
}
