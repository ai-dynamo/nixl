/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_SPDK_SPDK_PROGRESS_ENGINE_H
#define NIXL_SRC_PLUGINS_SPDK_SPDK_PROGRESS_ENGINE_H

#include <atomic>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "nixl_params.h"
#include "nixl_types.h"
#include "spdk_backend.h"

extern "C" {
#include <spdk/bdev.h>
}

struct spdk_thread;

class nixlBackendInitParams;

class nixlSpdkProgressEngine {
public:
    explicit nixlSpdkProgressEngine(const nixlBackendInitParams *init_params);
    ~nixlSpdkProgressEngine();

    // Owns the SPDK runtime slot, the progress thread and the SPDK thread; the
    // queued work captures 'this', so the engine must stay put.
    nixlSpdkProgressEngine(const nixlSpdkProgressEngine &) = delete;
    nixlSpdkProgressEngine &
    operator=(const nixlSpdkProgressEngine &) = delete;

    [[nodiscard]] bool
    hasInitError() const noexcept {
        return initErr_.load(std::memory_order_acquire);
    }

    [[nodiscard]] nixl_status_t
    registerDram(nixlSpdkDramMD &md);
    [[nodiscard]] nixl_status_t
    deregisterDram(nixlSpdkDramMD &md);
    [[nodiscard]] nixl_status_t
    openBdev(nixlSpdkBdevMD &md);
    void
    closeBdev(nixlSpdkBdevMD &md);

    [[nodiscard]] nixl_status_t
    postXfer(nixlSpdkBackendReqH *req_h);
    [[nodiscard]] nixl_status_t
    checkXfer(nixlSpdkBackendReqH *req_h);
    void
    cancelRequest(nixlSpdkBackendReqH *req_h);

private:
    enum class ExecMode {
        CallerSerialized,
        BackendLocked,
        ProgressThread,
    };

    void
    parseParams(const nixlBackendInitParams *init_params);
    void
    runThread();
    [[nodiscard]] nixl_status_t
    initRuntime();
    void
    finiRuntime();
    // As finiRuntime(), for callers that already hold the shared-runtime lock.
    void
    finiRuntimeLocked();
    void
    pollOnce();
    // Drive the runtime's shared app thread, if this backend can claim it.
    void
    pollAppThread();
    template<std::predicate F>
    void
    pollUntil(F &&done);
    // Run fn in the SPDK thread context and return its result.
    template<std::invocable F>
    std::invoke_result_t<F>
    execute(F &&fn);
    // Run fn on the runtime's shared app thread, serialized against the other
    // backends in the process.
    template<std::invocable F>
    std::invoke_result_t<F>
    executeOnAppThread(F &&fn);
    // As execute(), additionally serialized against other callers when the
    // backend owns the locking (see ExecMode).
    template<std::invocable F>
    std::invoke_result_t<F>
    executeLocked(F &&fn);
    // Run fn in the SPDK execution context and wait for its result: handed to
    // the progress thread when there is one, otherwise run inline under the
    // backend lock.
    template<std::invocable F>
    std::invoke_result_t<F>
    executeSync(F &&fn);
    // Stores the callable for later consumption on the progress thread.
    void
    enqueueAsync(std::function<void()> fn);
    void
    drainQueue();
    void
    submitRequest(nixlSpdkBackendReqH *req_h);
    void
    submitOne(nixlSpdkIoContext *io);
    void
    completeOne(nixlSpdkBackendReqH *req_h, nixl_status_t status);
    void
    retireRequest(nixlSpdkBackendReqH *req_h);
    static void
    bdevComplete(struct spdk_bdev_io *bdev_io, bool success, void *cb_arg);
    static void
    ioWaitRetry(void *cb_arg);
    static void
    bdevEventCb(enum spdk_bdev_event_type type, struct spdk_bdev *bdev, void *event_ctx);

    std::string name_ = "nixl_spdk";
    std::string jsonConfig_;
    std::string jsonConfigFile_;
    std::string coreMask_;
    std::size_t msgMempoolSize_ = 0;
    std::atomic<bool> initErr_{false};
    // Whether this backend holds a reference on the process-wide SPDK runtime.
    bool ownsRuntimeRef_ = false;
    bool runtimeInitialized_ = false;
    bool stopThread_ = false;
    ExecMode execMode_ = ExecMode::BackendLocked;
    uint64_t threadDelayUs_ = 0;
    // Count of bdev I/Os submitted to the device and not yet retired. Only the
    // progress thread reads/writes it (submit and completion both run there), so
    // it needs no synchronization. The progress loop polls while it is non-zero
    // and blocks when it (and the work queue) are empty.
    std::size_t inFlight_ = 0;
    spdk_thread *spdkThread_ = nullptr;
    std::thread progressThread_;
    mutable std::mutex execMutex_;
    std::mutex queueMutex_;
    std::condition_variable queueCv_;
    std::vector<std::function<void()>> producerQueue_;
    std::vector<std::function<void()>> consumerQueue_;
};

#endif
