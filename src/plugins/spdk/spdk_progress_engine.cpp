/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "spdk_progress_engine.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <latch>
#include <optional>
#include <ranges>
#include <string_view>
#include <utility>
#include <variant>

#include "backend/backend_aux.h"
#include "common/backend.h"
#include "common/nixl_log.h"

extern "C" {
#include <spdk/accel.h>
#include <spdk/env.h>
#include <spdk/init.h>
#include <spdk/nvme_spec.h>
#include <spdk/rpc.h>
#include <spdk/thread.h>
}

namespace {

// SPDK's env, thread, accel and bdev subsystems are singletons within this copy
// of SPDK, so every backend in the process shares one runtime: the first to
// start brings it up and the last to go away tears it down. Each backend still
// owns its own SPDK thread, its own I/O channels and its own bdev configuration.
//
// SPDK also designates the first thread created as the "app thread" and posts
// control-plane work to it (the JSON config loader, bdev unregistration). It
// cannot be reassigned and is only freed by spdk_thread_lib_fini(), so the
// runtime owns it rather than any one backend: otherwise a second backend's
// config load would deadlock waiting on the first backend to poll, and the
// first backend's destruction would strand it for everyone else.
struct SharedRuntime {
    std::mutex mutex;
    // Serializes execution on appThread. Backends acquire it opportunistically
    // (try_lock) from their poll loops, so no backend can block another.
    std::mutex appMutex;
    // Atomic because pollAppThread() reads it outside g_runtime.mutex: a second
    // backend bringing the runtime up publishes it concurrently with an existing
    // backend's poll loop.
    std::atomic<spdk_thread *> appThread{nullptr};
    std::size_t refs = 0;
    bool envInitialized = false;
    bool threadLibInitialized = false;
    bool iobufInitialized = false;
    bool accelInitialized = false;
    bool bdevInitialized = false;
};

SharedRuntime g_runtime;

struct AsyncResult {
    bool done = false;
    int rc = 0;
};

void
asyncDone(int rc, void *ctx) {
    auto *result = static_cast<AsyncResult *>(ctx);
    result->rc = rc;
    result->done = true;
}

void
bdevInitDone(void *ctx, int rc) {
    auto *result = static_cast<AsyncResult *>(ctx);
    result->rc = rc;
    result->done = true;
}

void
bdevFiniDone(void *ctx) {
    auto *result = static_cast<AsyncResult *>(ctx);
    result->done = true;
}

void
finishDone(void *ctx) {
    auto *result = static_cast<AsyncResult *>(ctx);
    result->done = true;
}

// Bdev names, file paths and transport addresses come from the caller, so a
// stray quote or backslash would otherwise produce malformed JSON, or let a
// value inject additional keys into the generated config.
std::string
jsonEscape(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char c : value) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += std::format("\\u{:04x}", static_cast<unsigned>(c));
            } else {
                out += c;
            }
            break;
        }
    }
    return out;
}

// Build a single-bdev subsystem config from the convenience parameters
// (bdev_type + bdev_name + a type-specific source), so callers do not have to
// hand-write SPDK bdev RPC JSON for the common case. Returns "" when the
// convenience params are absent or the type is unknown.
std::string
buildBdevConfigJson(const nixl_b_params_t *params) {
    if (!params) {
        return "";
    }
    auto opt = [&](const char *key) -> std::string {
        auto v = nixl::getBackendParamOptional<std::string>(params, key);
        return v ? *v : std::string();
    };
    const std::string type = opt("bdev_type");
    const std::string name = opt("bdev_name");
    if (type.empty() || name.empty()) {
        return "";
    }

    // Numeric fields are emitted unquoted, so reject anything that is not a
    // plain decimal number rather than splicing it into the JSON verbatim.
    auto number = [&](const char *key, std::string_view fallback) -> std::string {
        const std::string value = opt(key);
        if (value.empty()) {
            return std::string(fallback);
        }
        if (!std::ranges::all_of(value, [](unsigned char c) { return std::isdigit(c) != 0; })) {
            NIXL_ERROR << "SPDK: parameter '" << key << "' must be a decimal number, got '" << value
                       << "'";
            return {};
        }
        return value;
    };

    std::string method, body;
    if (type == "malloc") {
        const std::string num_blocks = number("bdev_num_blocks", "");
        const std::string block_size = number("bdev_block_size", "512");
        if (num_blocks.empty() || block_size.empty()) {
            return "";
        }
        method = "bdev_malloc_create";
        body = std::format(R"("name": "{}", "num_blocks": {}, "block_size": {})",
                           jsonEscape(name),
                           num_blocks,
                           block_size);
    } else if (type == "aio") {
        method = "bdev_aio_create";
        body = std::format(R"("name": "{}", "filename": "{}")",
                           jsonEscape(name),
                           jsonEscape(opt("bdev_filename")));
        const std::string block_size = number("bdev_block_size", "");
        if (!block_size.empty()) {
            body += std::format(R"(, "block_size": {})", block_size);
        }
    } else if (type == "nvme") {
        // Transport-attached NVMe only. This plugin supplies its own env
        // implementation, which does no PCI enumeration and cannot translate a
        // virtual address to a physical one, so a local PCIe controller can
        // never be driven from here. Reject it up front instead of letting the
        // attach fail obscurely much later.
        //
        // Attaching a controller named <name> exposes namespaces as <name>n1,
        // <name>n2, ...; that is the name to use in BLK descriptors.
        const std::string trtype = opt("bdev_trtype");
        if (trtype.empty()) {
            NIXL_ERROR << "SPDK: bdev_type 'nvme' requires 'bdev_trtype' (RDMA or TCP)";
            return "";
        }
        if (std::ranges::equal(trtype, std::string_view("pcie"), [](char a, char b) {
                return std::tolower(static_cast<unsigned char>(a)) == b;
            })) {
            NIXL_ERROR << "SPDK: bdev_trtype 'PCIe' is not supported: this plugin does not "
                          "probe local PCI devices. Use an RDMA or TCP NVMe-oF target.";
            return "";
        }
        const std::string traddr = opt("bdev_traddr");
        const std::string subnqn = opt("bdev_subnqn");
        if (traddr.empty() || subnqn.empty()) {
            NIXL_ERROR << "SPDK: bdev_type 'nvme' requires 'bdev_traddr' and 'bdev_subnqn'";
            return "";
        }
        method = "bdev_nvme_attach_controller";
        body = std::format(R"("name": "{}", "trtype": "{}", "traddr": "{}", "subnqn": "{}")",
                           jsonEscape(name),
                           jsonEscape(trtype),
                           jsonEscape(traddr),
                           jsonEscape(subnqn));
        const std::string adrfam = opt("bdev_adrfam");
        if (!adrfam.empty()) {
            body += std::format(R"(, "adrfam": "{}")", jsonEscape(adrfam));
        }
        const std::string trsvcid = opt("bdev_trsvcid");
        if (!trsvcid.empty()) {
            body += std::format(R"(, "trsvcid": "{}")", jsonEscape(trsvcid));
        }
    } else {
        NIXL_ERROR << "SPDK: unknown bdev_type '" << type << "' (expected malloc, aio, or nvme)";
        return "";
    }
    return std::format(
        R"({{"subsystems": [{{"subsystem": "bdev", "config": [{{"method": "{}", "params": {{{}}}}}]}}]}})",
        method,
        body);
}

template<typename... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};

constexpr bool kOpenForWrite = true;
// spdk_thread_poll() takes a message budget and a timestamp; 0 means "no limit"
// and "read the current time" respectively.
constexpr uint32_t kPollAllMessages = 0;
constexpr uint64_t kPollUseCurrentTime = 0;

// The SPDK handles an I/O is issued against, resolved from the transfer target.
struct Device {
    spdk_bdev *bdev;
    spdk_bdev_desc *desc;
    spdk_io_channel *channel;
};

// Makes the calling thread impersonate an SPDK thread for a scope. Restoring
// from a destructor matters: the work run under it allocates, so an exception
// must not leave the caller permanently impersonating the SPDK thread.
class SpdkThreadScope {
public:
    explicit SpdkThreadScope(spdk_thread *thread) : prev_(spdk_get_thread()) {
        spdk_set_thread(thread);
    }

    ~SpdkThreadScope() {
        spdk_set_thread(prev_);
    }

    SpdkThreadScope(const SpdkThreadScope &) = delete;
    SpdkThreadScope &
    operator=(const SpdkThreadScope &) = delete;

private:
    spdk_thread *prev_;
};

// Set while this thread runs inside executeOnAppThread(), i.e. already holds
// g_runtime.appMutex. std::mutex::try_lock() on a mutex the caller already owns
// is undefined, so the poll path checks this instead of re-locking.
thread_local bool t_onAppThread = false;

class AppThreadScope {
public:
    AppThreadScope() : lock_(g_runtime.appMutex), scope_(g_runtime.appThread.load()) {
        t_onAppThread = true;
    }

    ~AppThreadScope() {
        t_onAppThread = false;
    }

    AppThreadScope(const AppThreadScope &) = delete;
    AppThreadScope &
    operator=(const AppThreadScope &) = delete;

private:
    std::scoped_lock<std::mutex> lock_;
    SpdkThreadScope scope_;
};

} // namespace

nixlSpdkProgressEngine::nixlSpdkProgressEngine(const nixlBackendInitParams *init_params) {
    parseParams(init_params);

    if (execMode_ == ExecMode::ProgressThread) {
        progressThread_ = std::thread(&nixlSpdkProgressEngine::runThread, this);
        std::unique_lock<std::mutex> lock(queueMutex_);
        queueCv_.wait(lock, [this]() { return runtimeInitialized_ || initErr_; });
    } else {
        const nixl_status_t status = initRuntime();
        if (status != NIXL_SUCCESS) {
            initErr_ = true;
        }
    }
}

nixlSpdkProgressEngine::~nixlSpdkProgressEngine() {
    if (execMode_ == ExecMode::ProgressThread) {
        {
            const std::scoped_lock lock(queueMutex_);
            stopThread_ = true;
        }
        queueCv_.notify_one();
        if (progressThread_.joinable()) {
            progressThread_.join();
        }
    } else {
        executeLocked([this]() { finiRuntime(); });
    }
}

void
nixlSpdkProgressEngine::parseParams(const nixlBackendInitParams *init_params) {
    const nixl_b_params_t *params = init_params ? init_params->customParams : nullptr;
    if (params) {
        if (auto value = nixl::getBackendParamOptional<std::string>(params, "json_config")) {
            jsonConfig_ = *value;
        }
        if (auto value = nixl::getBackendParamOptional<std::string>(params, "json_config_file")) {
            jsonConfigFile_ = *value;
        }
        if (auto value = nixl::getBackendParamOptional<std::string>(params, "spdk_name")) {
            name_ = *value;
        }
        if (auto value = nixl::getBackendParamOptional<std::string>(params, "core_mask")) {
            coreMask_ = *value;
        }
        if (auto value = nixl::getBackendParamOptional<size_t>(params, "msg_mempool_size")) {
            msgMempoolSize_ = *value;
        }
        // OBJ_SEG (NVMe-KV) uses the backend's bdev, named by 'bdev_name' (the
        // same parameter that names the convenience-config bdev).
        if (auto value = nixl::getBackendParamOptional<std::string>(params, "bdev_name")) {
            objBdevName_ = *value;
        }
    }
    // Fall back to the convenience parameters only when no explicit JSON was
    // supplied; explicit json_config / json_config_file always take precedence.
    if (jsonConfig_.empty() && jsonConfigFile_.empty()) {
        jsonConfig_ = buildBdevConfigJson(params);
    }
    threadDelayUs_ = init_params ? init_params->pthrDelay : 0;

    if (init_params && init_params->enableProgTh) {
        execMode_ = ExecMode::ProgressThread;
    } else if (init_params && init_params->syncMode == nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE) {
        execMode_ = ExecMode::CallerSerialized;
    } else {
        execMode_ = ExecMode::BackendLocked;
    }
}

void
nixlSpdkProgressEngine::runThread() {
    const nixl_status_t status = initRuntime();
    {
        const std::scoped_lock lock(queueMutex_);
        initErr_ = status != NIXL_SUCCESS;
        runtimeInitialized_ = status == NIXL_SUCCESS;
    }
    queueCv_.notify_all();
    if (status != NIXL_SUCCESS) {
        // initRuntime() already unwound itself; the destructor only joins this
        // thread.
        return;
    }

    while (true) {
        drainQueue();
        pollOnce();

        std::unique_lock<std::mutex> lock(queueMutex_);
        if (stopThread_) {
            break;
        }
        if (!producerQueue_.empty()) {
            continue;
        }
        if (inFlight_ > 0 || ioWaiting_ > 0) {
            // I/O is outstanding: the device may post a completion at any moment,
            // so keep polling. Entries parked on an io_wait queue count too --
            // only continued polling of the channel drives their retry callback.
            // threadDelayUs_ optionally throttles the poll to trade completion
            // latency for CPU; 0 means busy-poll.
            if (threadDelayUs_ > 0) {
                queueCv_.wait_for(lock, std::chrono::microseconds(threadDelayUs_));
            }
        } else {
            // Nothing queued and nothing in flight, so no completion can arrive.
            // Block until new work is enqueued instead of spinning a core.
            queueCv_.wait(lock, [this]() { return !producerQueue_.empty() || stopThread_; });
        }
    }

    drainQueue();

    // Drain before tearing anything down. cancelRequest() returns while its I/O
    // may still be on the device, and the completion callback dereferences
    // io->engine and io->reqH; finiRuntime() would close the bdev and run
    // spdk_bdev_finish() underneath it, and the handle would never be retired.
    pollUntil([this]() { return inFlight_ == 0 && ioWaiting_ == 0; });

    execute([this]() { finiRuntime(); });
}

nixl_status_t
nixlSpdkProgressEngine::initRuntime() {
    if (jsonConfig_.empty() && jsonConfigFile_.empty()) {
        NIXL_ERROR << "SPDK: a bdev configuration is required via the 'json_config' "
                      "(inline JSON) or 'json_config_file' (path) backend parameter";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Held across the whole of init so a second backend starting concurrently
    // cannot observe a half-built runtime. finiRuntimeLocked() is used on the
    // failure paths below because this thread already owns the lock.
    std::unique_lock<std::mutex> runtime_lock(g_runtime.mutex);

    // Taken before anything is brought up so that a failure part-way through
    // leaves this backend as the last reference and finiRuntimeLocked() unwinds
    // whatever did come up.
    ++g_runtime.refs;
    ownsRuntimeRef_ = true;

    if (!g_runtime.envInitialized) {
        spdk_env_opts env_opts;
        env_opts.opts_size = sizeof(env_opts);
        spdk_env_opts_init(&env_opts);
        env_opts.name = name_.c_str();
        if (!coreMask_.empty()) {
            env_opts.core_mask = coreMask_.c_str();
        }
        const int rc = spdk_env_init(&env_opts);
        if (rc != 0) {
            NIXL_ERROR << "SPDK: spdk_env_init failed: " << rc;
            finiRuntimeLocked();
            return NIXL_ERR_BACKEND;
        }
        g_runtime.envInitialized = true;
    }

    if (!g_runtime.threadLibInitialized) {
        const int rc = msgMempoolSize_ > 0 ?
            spdk_thread_lib_init_ext(nullptr, nullptr, 0, msgMempoolSize_) :
            spdk_thread_lib_init(nullptr, 0);
        if (rc != 0) {
            NIXL_ERROR << "SPDK: spdk_thread_lib_init failed: " << rc;
            finiRuntimeLocked();
            return NIXL_ERR_BACKEND;
        }
        g_runtime.threadLibInitialized = true;
    }

    // Created before any backend thread so that SPDK picks it, not a backend's
    // thread, as the app thread.
    if (!g_runtime.appThread) {
        g_runtime.appThread = spdk_thread_create("nixl_spdk_ctrl", nullptr);
        if (!g_runtime.appThread) {
            NIXL_ERROR << "SPDK: spdk_thread_create failed for the control thread";
            finiRuntimeLocked();
            return NIXL_ERR_BACKEND;
        }
    }

    spdkThread_ = spdk_thread_create(name_.c_str(), nullptr);
    if (!spdkThread_) {
        NIXL_ERROR << "SPDK: spdk_thread_create failed";
        finiRuntimeLocked();
        return NIXL_ERR_BACKEND;
    }

    // The subsystems below and the config loader both assume they run on the app
    // thread; g_runtime.mutex is held, so nobody else is driving it.
    executeOnAppThread([this]() {
        if (!g_runtime.iobufInitialized) {
            int iobuf_rc = spdk_iobuf_initialize();
            if (iobuf_rc != 0) {
                NIXL_ERROR << "SPDK: spdk_iobuf_initialize failed: " << iobuf_rc;
                initErr_ = true;
                return;
            }
            g_runtime.iobufInitialized = true;
        }

        // The bdev layer (and most bdev modules, e.g. malloc) rely on the accel
        // framework for data-path operations, so it must come up before bdev.
        if (!g_runtime.accelInitialized) {
            int accel_rc = spdk_accel_initialize();
            if (accel_rc != 0) {
                NIXL_ERROR << "SPDK: spdk_accel_initialize failed: " << accel_rc;
                initErr_ = true;
                return;
            }
            g_runtime.accelInitialized = true;
        }

        if (!g_runtime.bdevInitialized) {
            AsyncResult bdev_init;
            spdk_bdev_initialize(bdevInitDone, &bdev_init);
            pollUntil([&bdev_init]() { return bdev_init.done; });
            if (bdev_init.rc != 0) {
                NIXL_ERROR << "SPDK: spdk_bdev_initialize failed: " << bdev_init.rc;
                initErr_ = true;
                return;
            }
            g_runtime.bdevInitialized = true;
        }

        // Each backend applies its own configuration into the shared bdev
        // registry, so a second backend must name bdevs the first did not.
        std::string json;
        if (!jsonConfig_.empty()) {
            json = jsonConfig_;
        } else {
            std::ifstream json_file(jsonConfigFile_, std::ios::binary);
            if (!json_file) {
                NIXL_ERROR << "SPDK: failed to open JSON config file " << jsonConfigFile_;
                initErr_ = true;
                return;
            }
            json.assign((std::istreambuf_iterator<char>(json_file)),
                        std::istreambuf_iterator<char>());
        }
        // load_config (initialize_subsystems=false) applies the JSON in a single
        // pass at the current RPC state, running a method only when that state is
        // a subset of the method's registration mask. bdev_*_create are
        // RUNTIME-only, so the state must be exactly RUNTIME (not STARTUP|RUNTIME,
        // which would exclude them).
        spdk_rpc_set_state(SPDK_RPC_RUNTIME);
        AsyncResult config_load;
        spdk_subsystem_load_config(
            json.data(), static_cast<ssize_t>(json.size()), asyncDone, &config_load, true);
        pollUntil([&config_load]() { return config_load.done; });
        if (config_load.rc != 0) {
            NIXL_ERROR << "SPDK: failed to load JSON config: " << config_load.rc;
            initErr_ = true;
        }
    });

    if (initErr_) {
        finiRuntimeLocked();
        return NIXL_ERR_BACKEND;
    }
    runtimeInitialized_ = true;
    return NIXL_SUCCESS;
}

void
nixlSpdkProgressEngine::finiRuntime() {
    const std::scoped_lock lock(g_runtime.mutex);
    finiRuntimeLocked();
}

void
nixlSpdkProgressEngine::finiRuntimeLocked() {
    // The shared subsystems come down only with the last backend holding a
    // reference; everything else here is this backend's own state.
    const bool last = ownsRuntimeRef_ && g_runtime.refs == 1;

    if (spdkThread_) {
        // Release this backend's KV device before any bdev teardown.
        execute([this]() {
            if (kvChannel_) {
                spdk_put_io_channel(kvChannel_);
                kvChannel_ = nullptr;
            }
            if (kvDesc_) {
                spdk_bdev_close(kvDesc_);
                kvDesc_ = nullptr;
                kvBdev_ = nullptr;
            }
        });
        kvOpened_ = false;
    }

    if (last) {
        executeOnAppThread([this]() {
            if (g_runtime.bdevInitialized) {
                AsyncResult finish;
                spdk_bdev_finish(bdevFiniDone, &finish);
                pollUntil([&finish]() { return finish.done; });
                g_runtime.bdevInitialized = false;
            }
            if (g_runtime.accelInitialized) {
                AsyncResult finish;
                spdk_accel_finish(finishDone, &finish);
                pollUntil([&finish]() { return finish.done; });
                g_runtime.accelInitialized = false;
            }
            if (g_runtime.iobufInitialized) {
                AsyncResult finish;
                spdk_iobuf_finish(finishDone, &finish);
                pollUntil([&finish]() { return finish.done; });
                g_runtime.iobufInitialized = false;
            }
        });
    }

    if (spdkThread_) {
        execute([this]() {
            spdk_thread_exit(spdkThread_);
            pollUntil([this]() { return spdk_thread_is_exited(spdkThread_); });
        });
        spdk_thread_destroy(spdkThread_);
        spdkThread_ = nullptr;
    }

    // spdk_thread_lib_fini() requires every SPDK thread to be gone, so the app
    // thread goes away only once this backend has destroyed its own.
    if (last) {
        if (g_runtime.appThread) {
            executeOnAppThread([this]() {
                spdk_thread_exit(g_runtime.appThread);
                pollUntil([]() { return spdk_thread_is_exited(g_runtime.appThread); });
            });
            spdk_thread_destroy(g_runtime.appThread);
            g_runtime.appThread = nullptr;
        }
        if (g_runtime.threadLibInitialized) {
            spdk_thread_lib_fini();
            g_runtime.threadLibInitialized = false;
        }
        if (g_runtime.envInitialized) {
            spdk_env_fini();
            g_runtime.envInitialized = false;
        }
    }

    if (ownsRuntimeRef_) {
        --g_runtime.refs;
        ownsRuntimeRef_ = false;
    }
}

void
nixlSpdkProgressEngine::pollOnce() {
    if (spdkThread_) {
        spdk_thread_poll(spdkThread_, kPollAllMessages, kPollUseCurrentTime);
    }
    pollAppThread();
}

// The app thread has no dedicated poller: whichever backend is running drives
// it. try_lock rather than lock so that a backend with nothing to do for the app
// thread never stalls behind one that is holding it.
void
nixlSpdkProgressEngine::pollAppThread() {
    // Loaded once: another backend may publish or clear it while this runs.
    spdk_thread *const app_thread = g_runtime.appThread.load(std::memory_order_acquire);
    if (!app_thread) {
        return;
    }
    if (t_onAppThread) {
        spdk_thread_poll(app_thread, kPollAllMessages, kPollUseCurrentTime);
        return;
    }
    std::unique_lock<std::mutex> lock(g_runtime.appMutex, std::try_to_lock);
    if (lock.owns_lock()) {
        spdk_thread_poll(app_thread, kPollAllMessages, kPollUseCurrentTime);
    }
}

// Defined here rather than in the header: these are private and every call site
// is in this file, so a stray call from another TU fails to link.
template<std::predicate F>
void
nixlSpdkProgressEngine::pollUntil(F &&done) {
    // Deliberately no deadline: every caller is a bring-up, teardown or drain
    // step with no way to recover from a half-finished operation, so abandoning
    // the wait would leave the runtime in a worse state than continuing to poll.
    // Warn periodically instead, so a bdev module that never completes shows up
    // in the log rather than as a silent spin. None of these paths are on the
    // per-transfer hot path, so the clock read costs nothing that matters.
    constexpr auto kStallWarnInterval = std::chrono::seconds(30);
    auto next_warn = std::chrono::steady_clock::now() + kStallWarnInterval;

    while (!done()) {
        pollOnce();
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_warn) {
            NIXL_WARN << "SPDK: still waiting for an SPDK operation to complete after "
                      << kStallWarnInterval.count() << "s; the bdev configuration may be stuck";
            next_warn = now + kStallWarnInterval;
        }
    }
}

template<std::invocable F>
std::invoke_result_t<F>
nixlSpdkProgressEngine::execute(F &&fn) {
    const SpdkThreadScope scope(spdkThread_);
    return fn();
}

template<std::invocable F>
std::invoke_result_t<F>
nixlSpdkProgressEngine::executeOnAppThread(F &&fn) {
    const AppThreadScope scope;
    return fn();
}

template<std::invocable F>
std::invoke_result_t<F>
nixlSpdkProgressEngine::executeLocked(F &&fn) {
    if (execMode_ == ExecMode::BackendLocked) {
        const std::scoped_lock lock(execMutex_);
        return execute(std::forward<F>(fn));
    }
    return execute(std::forward<F>(fn));
}

template<std::invocable F>
std::invoke_result_t<F>
nixlSpdkProgressEngine::executeSync(F &&fn) {
    using Result = std::invoke_result_t<F>;

    if (execMode_ != ExecMode::ProgressThread) {
        return executeLocked(std::forward<F>(fn));
    }

    // Hand the work to the progress thread and block until it retires. The
    // latch is released from a destructor so an exception escaping fn() cannot
    // strand this thread waiting forever.
    std::latch done{1};

    struct Signal {
        std::latch &latch;

        ~Signal() {
            latch.count_down();
        }
    };

    // The captured locals outlive these lambdas only because this thread blocks
    // on the latch until the work has retired.
    if constexpr (std::is_void_v<Result>) {
        enqueueAsync([&fn, &done]() {
            const Signal signal{done};
            fn();
        });
        done.wait();
    } else {
        std::optional<Result> result;
        enqueueAsync([&fn, &done, &result]() {
            const Signal signal{done};
            result.emplace(fn());
        });
        done.wait();
        return std::move(*result);
    }
}

void
nixlSpdkProgressEngine::enqueueAsync(std::function<void()> fn) {
    {
        const std::scoped_lock lock(queueMutex_);
        producerQueue_.push_back(std::move(fn));
    }
    queueCv_.notify_one();
}

void
nixlSpdkProgressEngine::drainQueue() {
    {
        const std::scoped_lock lock(queueMutex_);
        consumerQueue_.swap(producerQueue_);
    }
    if (consumerQueue_.empty()) {
        return;
    }
    execute([this]() {
        for (auto &fn : consumerQueue_) {
            fn();
        }
    });
    consumerQueue_.clear();
}

nixl_status_t
nixlSpdkProgressEngine::registerDram(nixlSpdkDramMD &md) {
    return executeSync([&]() {
        // We map the DRAM directly for zero-copy DMA and intentionally do not
        // fall back to bounce buffers, so reject memory that cannot be
        // registered. spdk_mem_register requires page-aligned (4 KiB) address
        // and length on current SPDK.
        const int rc = spdk_mem_register(reinterpret_cast<void *>(md.addr), md.len);
        if (rc != 0) {
            NIXL_ERROR << "SPDK: spdk_mem_register failed (" << rc
                       << "); DRAM registered with the SPDK backend must be page-aligned "
                          "(4 KiB) in address and length";
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    });
}

nixl_status_t
nixlSpdkProgressEngine::deregisterDram(nixlSpdkDramMD &md) {
    return executeSync([&]() {
        const int rc = spdk_mem_unregister(reinterpret_cast<void *>(md.addr), md.len);
        if (rc != 0) {
            NIXL_ERROR << "SPDK: spdk_mem_unregister failed: " << rc;
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    });
}

nixl_status_t
nixlSpdkProgressEngine::openBdev(nixlSpdkBdevMD &md) {
    return executeSync([&]() {
        const int rc =
            spdk_bdev_open_ext(md.bdevName.c_str(), kOpenForWrite, bdevEventCb, &md, &md.desc);
        if (rc != 0) {
            NIXL_ERROR << "SPDK: failed to open bdev " << md.bdevName << ": " << rc;
            return NIXL_ERR_NOT_FOUND;
        }
        md.bdev = spdk_bdev_desc_get_bdev(md.desc);
        md.channel = spdk_bdev_get_io_channel(md.desc);
        if (!md.channel) {
            NIXL_ERROR << "SPDK: failed to get I/O channel for bdev " << md.bdevName;
            releaseBdevHandles(md.desc, md.bdev, md.channel);
            return NIXL_ERR_BACKEND;
        }
        if (!spdk_bdev_io_type_supported(md.bdev, SPDK_BDEV_IO_TYPE_READ) ||
            !spdk_bdev_io_type_supported(md.bdev, SPDK_BDEV_IO_TYPE_WRITE)) {
            NIXL_ERROR << "SPDK: bdev " << md.bdevName << " does not support read/write";
            releaseBdevHandles(md.desc, md.bdev, md.channel);
            return NIXL_ERR_NOT_SUPPORTED;
        }
        md.blockSize = spdk_bdev_get_block_size(md.bdev);
        md.writeUnitSize = spdk_bdev_get_write_unit_size(md.bdev);
        md.numBlocks = spdk_bdev_get_num_blocks(md.bdev);
        // Both are divisors in the descriptor validation on the transfer path;
        // a module reporting zero would raise SIGFPE on the first transfer.
        if (md.blockSize == 0 || md.writeUnitSize == 0) {
            NIXL_ERROR << "SPDK: bdev " << md.bdevName << " reports an invalid geometry: "
                       << "block size " << md.blockSize << ", write unit size " << md.writeUnitSize;
            releaseBdevHandles(md.desc, md.bdev, md.channel);
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    });
}

void
nixlSpdkProgressEngine::releaseBdevHandles(spdk_bdev_desc *&desc,
                                           spdk_bdev *&bdev,
                                           spdk_io_channel *&channel) {
    if (channel) {
        spdk_put_io_channel(channel);
        channel = nullptr;
    }
    if (desc) {
        spdk_bdev_close(desc);
        desc = nullptr;
    }
    bdev = nullptr;
}

void
nixlSpdkProgressEngine::closeBdev(nixlSpdkBdevMD &md) {
    executeSync([&]() { releaseBdevHandles(md.desc, md.bdev, md.channel); });
}

nixl_status_t
nixlSpdkProgressEngine::ensureKvBdev() {
    return executeSync([&]() {
        if (kvOpened_) {
            return NIXL_SUCCESS;
        }
        if (objBdevName_.empty()) {
            NIXL_ERROR << "SPDK: OBJ_SEG requires the 'bdev_name' parameter naming a "
                          "KV-capable bdev";
            return NIXL_ERR_INVALID_PARAM;
        }
        const int rc =
            spdk_bdev_open_ext(objBdevName_.c_str(), kOpenForWrite, kvBdevEventCb, this, &kvDesc_);
        if (rc != 0) {
            NIXL_ERROR << "SPDK: failed to open KV bdev " << objBdevName_ << ": " << rc;
            return NIXL_ERR_NOT_FOUND;
        }
        kvBdev_ = spdk_bdev_desc_get_bdev(kvDesc_);

        // OBJ_SEG only works on devices that accept NVMe passthru (to carry the
        // KV command) and identify as the NVMe KV command set.
        nixl_status_t supported = NIXL_SUCCESS;
        if (!spdk_bdev_io_type_supported(kvBdev_, SPDK_BDEV_IO_TYPE_NVME_IO)) {
            NIXL_ERROR << "SPDK: KV bdev " << objBdevName_
                       << " does not support NVMe passthru (SPDK_BDEV_IO_TYPE_NVME_IO)";
            supported = NIXL_ERR_NOT_SUPPORTED;
        } else if (spdk_bdev_get_nvme_csi(kvBdev_) != SPDK_NVME_CSI_KV) {
            NIXL_ERROR << "SPDK: KV bdev " << objBdevName_
                       << " does not identify as the NVMe KV command set";
            supported = NIXL_ERR_NOT_SUPPORTED;
        }
        if (supported != NIXL_SUCCESS) {
            spdk_bdev_close(kvDesc_);
            kvDesc_ = nullptr;
            kvBdev_ = nullptr;
            return supported;
        }

        kvChannel_ = spdk_bdev_get_io_channel(kvDesc_);
        if (!kvChannel_) {
            NIXL_ERROR << "SPDK: failed to get I/O channel for KV bdev " << objBdevName_;
            spdk_bdev_close(kvDesc_);
            kvDesc_ = nullptr;
            kvBdev_ = nullptr;
            return NIXL_ERR_BACKEND;
        }
        kvOpened_ = true;
        return NIXL_SUCCESS;
    });
}

nixl_status_t
nixlSpdkProgressEngine::postXfer(nixlSpdkBackendReqH *req_h) {
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // Reset completion state so a previously completed handle can be reposted,
    // then mark it submitted before handing it to the execution context. Both
    // happen on the caller thread, ahead of any progress-thread activity, so
    // cancelRequest() (also caller thread) sees a stable kSubmitted bit.
    req_h->reset();
    req_h->lifeState_.fetch_or(nixlSpdkBackendReqH::kSubmitted, std::memory_order_acq_rel);
    if (execMode_ == ExecMode::ProgressThread) {
        enqueueAsync([this, req_h]() { submitRequest(req_h); });
    } else {
        executeLocked([this, req_h]() {
            submitRequest(req_h);
            pollOnce();
        });
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlSpdkProgressEngine::checkXfer(nixlSpdkBackendReqH *req_h) {
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (execMode_ != ExecMode::ProgressThread &&
        (req_h->lifeState_.load(std::memory_order_acquire) & nixlSpdkBackendReqH::kDone) == 0) {
        executeLocked([this]() { pollOnce(); });
    }
    return req_h->status();
}

void
nixlSpdkProgressEngine::cancelRequest(nixlSpdkBackendReqH *req_h) {
    if (!req_h) {
        return;
    }
    // SPDK cannot un-submit an I/O that has already reached the device, so we do
    // not attempt a hard abort. Instead the handle is kept alive until its
    // outstanding I/O actually completes (retireRequest frees it then); the
    // caller must keep the registered DRAM buffer valid until completion.
    req_h->cancelled_.store(true, std::memory_order_release);
    const uint32_t prev =
        req_h->lifeState_.fetch_or(nixlSpdkBackendReqH::kReleased, std::memory_order_acq_rel);
    if (prev & nixlSpdkBackendReqH::kDone) {
        // Completion path already retired it; we are the last owner.
        delete req_h;
        return;
    }
    if (!(prev & nixlSpdkBackendReqH::kSubmitted)) {
        // Prepared but never posted: no completion will ever arrive for it.
        delete req_h;
    }
    // Otherwise I/O is in flight; retireRequest() frees it on completion.
}

void
nixlSpdkProgressEngine::submitRequest(nixlSpdkBackendReqH *req_h) {
    if (req_h->lifeState_.load(std::memory_order_acquire) & nixlSpdkBackendReqH::kReleased) {
        // Released before we reached the execution context; nothing is in
        // flight, so retire (and free) it here instead of submitting.
        retireRequest(req_h);
        return;
    }
    // outstanding_ carries a submission guard (see reset()), so a synchronous
    // completion from submitOne() cannot drop the count to zero while we are
    // still iterating req_h->ios_. We release the guard once submission is done.
    const size_t count = req_h->ios_.size();
    for (size_t i = 0; i < count; ++i) {
        submitOne(&req_h->ios_[i]);
    }
    if (req_h->outstanding_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        retireRequest(req_h);
    }
}

void
nixlSpdkProgressEngine::submitOne(nixlSpdkIoContext *io) {
    auto *req_h = io->reqH;
    io->engine = this;
    if (req_h->cancelled_.load(std::memory_order_acquire)) {
        completeOne(req_h, NIXL_ERR_NOT_ALLOWED);
        return;
    }

    // BLK_SEG issues read/write against the registered bdev's own handles;
    // OBJ_SEG carries an NVMe KV command over passthru on the shared KV device.
    // std::visit keeps the two exhaustive: a new target alternative will not
    // compile until it is handled here.
    const Device device = std::visit(
        overloaded{
            [](const nixlSpdkBlkTarget &blk) {
                return Device{blk.md->bdev, blk.md->desc, blk.md->channel};
            },
            [this](const nixlSpdkObjTarget &) { return Device{kvBdev_, kvDesc_, kvChannel_}; },
        },
        io->target);

    // The BLK handles come from the registration and the OBJ handles from this
    // engine, so neither is guaranteed here: OBJ metadata registered against a
    // different engine instance reaches this point with the KV device unopened,
    // and passthru would receive a null descriptor and channel.
    if (!device.bdev || !device.desc || !device.channel) {
        NIXL_ERROR << "SPDK: transfer target has no open device";
        completeOne(req_h, NIXL_ERR_BACKEND);
        return;
    }

    io->ioWaitQueued = false;
    io->waitEntry.bdev = device.bdev;
    io->waitEntry.cb_fn = ioWaitRetry;
    io->waitEntry.cb_arg = io;

    const int rc = std::visit(
        overloaded{
            [&](const nixlSpdkBlkTarget &blk) {
                return (req_h->operation_ == NIXL_READ) ? spdk_bdev_read(device.desc,
                                                                         device.channel,
                                                                         io->buf,
                                                                         blk.offset,
                                                                         io->nbytes,
                                                                         bdevComplete,
                                                                         io) :
                                                          spdk_bdev_write(device.desc,
                                                                          device.channel,
                                                                          io->buf,
                                                                          blk.offset,
                                                                          io->nbytes,
                                                                          bdevComplete,
                                                                          io);
            },
            [&](const nixlSpdkObjTarget &obj) {
                // Build a raw NVMe KV command (Store for WRITE, Retrieve for
                // READ) and forward it via bdev NVMe passthru. The key packs into
                // cdw2-3 (first 8 bytes) and cdw14-15 (next 8); cdw10.vsize is the
                // value size and cdw11.kl the key length.
                const std::span<const std::byte> key = obj.md->key();
                spdk_nvme_cmd cmd = {};
                cmd.opc = (req_h->operation_ == NIXL_WRITE) ? SPDK_NVME_OPC_KV_STORE :
                                                              SPDK_NVME_OPC_KV_RETRIEVE;
                cmd.cdw10_bits.kv.vsize = static_cast<uint32_t>(io->nbytes);
                cmd.cdw11_bits.kv.kl = static_cast<uint32_t>(key.size());
                cmd.cdw11_bits.kv.ro = 0;

                // Assemble the key as four dwords and assign them individually.
                // Each cdw member is 4 bytes wide, so copying 8 bytes through
                // cdw2 (or cdw14) would write through one member into the next.
                std::array<uint32_t, 4> kw{};
                std::memcpy(
                    kw.data(), key.data(), std::min(key.size(), kw.size() * sizeof(uint32_t)));
                cmd.cdw2 = kw[0];
                cmd.cdw3 = kw[1];
                cmd.cdw14 = kw[2];
                cmd.cdw15 = kw[3];
                return spdk_bdev_nvme_io_passthru(
                    device.desc, device.channel, &cmd, io->buf, io->nbytes, bdevComplete, io);
            },
        },
        io->target);

    if (rc == -ENOMEM) {
        int wait_rc = spdk_bdev_queue_io_wait(device.bdev, device.channel, &io->waitEntry);
        if (wait_rc == 0) {
            io->ioWaitQueued = true;
            // Counted as outstanding work: the retry only runs while the poll
            // loop keeps driving the channel, and without this the loop would
            // see no in-flight I/O, block on the queue, and never resume.
            ++ioWaiting_;
            return;
        }
        NIXL_ERROR << "SPDK: spdk_bdev_queue_io_wait failed: " << wait_rc;
        completeOne(req_h, NIXL_ERR_BACKEND);
        return;
    }
    if (rc != 0) {
        NIXL_ERROR << "SPDK: bdev I/O submit failed: " << rc;
        completeOne(req_h, rc == -EINVAL ? NIXL_ERR_INVALID_PARAM : NIXL_ERR_BACKEND);
        return;
    }
    // The I/O is now in flight on the device; bdevComplete will decrement this
    // when it retires. The progress thread polls hard while inFlight_ > 0 rather
    // than sleeping (a poll that returns no work does not mean the device is
    // idle - it may just not have posted the completion yet). Only the progress
    // thread touches inFlight_ (submit and completion both run on it), so it
    // needs no atomicity.
    ++inFlight_;
}

void
nixlSpdkProgressEngine::completeOne(nixlSpdkBackendReqH *req_h, nixl_status_t status) {
    if (status != NIXL_SUCCESS) {
        nixl_status_t expected = NIXL_IN_PROG;
        req_h->overallStatus_.compare_exchange_strong(expected, status, std::memory_order_acq_rel);
    }

    // fetch_sub returns the value before the decrement; reaching 1 means this
    // call took the count to zero, i.e. the last I/O retired the request.
    if (req_h->outstanding_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        retireRequest(req_h);
    }
}

void
nixlSpdkProgressEngine::retireRequest(nixlSpdkBackendReqH *req_h) {
    nixl_status_t expected = NIXL_IN_PROG;
    req_h->overallStatus_.compare_exchange_strong(
        expected, NIXL_SUCCESS, std::memory_order_acq_rel);

    // Mark completion and elect the single deleter: whichever of this path and
    // cancelRequest() sees the other's bit already set frees the handle.
    const uint32_t prev =
        req_h->lifeState_.fetch_or(nixlSpdkBackendReqH::kDone, std::memory_order_acq_rel);
    if (prev & nixlSpdkBackendReqH::kReleased) {
        delete req_h;
    }
}

void
nixlSpdkProgressEngine::bdevComplete(spdk_bdev_io *bdev_io, bool success, void *cb_arg) {
    auto *io = static_cast<nixlSpdkIoContext *>(cb_arg);
    // Capture the engine before completeOne(), which may free the request (and
    // hence io) when this is the last outstanding I/O.
    auto *engine = io->engine;
    auto *engine_req = io->reqH;
    spdk_bdev_free_io(bdev_io);
    --engine->inFlight_;
    engine->completeOne(engine_req, success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
}

void
nixlSpdkProgressEngine::ioWaitRetry(void *cb_arg) {
    auto *io = static_cast<nixlSpdkIoContext *>(cb_arg);
    io->ioWaitQueued = false;
    --io->engine->ioWaiting_;
    io->engine->submitOne(io);
}

void
nixlSpdkProgressEngine::bdevEventCb(enum spdk_bdev_event_type type,
                                    struct spdk_bdev *,
                                    void *event_ctx) {
    auto *md = static_cast<nixlSpdkBdevMD *>(event_ctx);
    NIXL_WARN << "SPDK: bdev event " << static_cast<int>(type) << " for " << md->bdevName;
}

void
nixlSpdkProgressEngine::kvBdevEventCb(enum spdk_bdev_event_type type,
                                      struct spdk_bdev *,
                                      void *event_ctx) {
    auto *engine = static_cast<nixlSpdkProgressEngine *>(event_ctx);
    NIXL_WARN << "SPDK: bdev event " << static_cast<int>(type) << " for KV bdev "
              << engine->objBdevName_;
}
