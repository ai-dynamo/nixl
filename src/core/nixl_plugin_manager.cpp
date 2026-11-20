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

#include "plugin_manager.h"
#include "nixl.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include <dlfcn.h>
#include <algorithm>
#include <exception>
#include <filesystem>
#include <dirent.h>
#include <unistd.h>  // For access() and F_OK
#include <fstream>
#include <string>
#include <map>

const std::string backendPluginPrefix = "libplugin_";
const std::string telemetryPluginPrefix = "libtelemetry_exporter_";
const std::string tracePluginPrefix = "libtrace_backend_";
const std::string kPluginSuffix = ".so";
const std::string kUcxPluginName = "UCX";
const std::string kUcxDeepBindVar = "NIXL_UCX_DEEPBIND";
const std::string prometheus_plugin_name = "prometheus";
const std::string prometheus_mp_plugin_name = "prometheus_mp";

// pluginHandle implementation
nixlBackendPluginHandle::nixlBackendPluginHandle(void *handle, nixlBackendPlugin *plugin)
    : nixlPluginHandle(handle),
      plugin_(plugin) {}

nixlBackendPluginHandle::~nixlBackendPluginHandle() {
    if (handle_) {
        // Call the plugin's cleanup function
        typedef void (*fini_func_t)();
        fini_func_t fini = (fini_func_t) dlsym(handle_, "nixl_plugin_fini");
        if (fini) {
            fini();
        }

        // Close the dynamic library
        dlclose(handle_);
        handle_ = nullptr;
        plugin_ = nullptr;
    }
}

nixlBackendEngine *
nixlBackendPluginHandle::createEngine(const nixlBackendInitParams *init_params) const {
    if (plugin_ && plugin_->create_engine) {
        return plugin_->create_engine(init_params);
    }
    return nullptr;
}

void
nixlBackendPluginHandle::destroyEngine(nixlBackendEngine *engine) const {
    if (plugin_ && plugin_->destroy_engine && engine) {
        plugin_->destroy_engine(engine);
    }
}

const char *
nixlBackendPluginHandle::getName() const {
    if (plugin_ && plugin_->get_plugin_name) {
        return plugin_->get_plugin_name();
    }
    return "unknown";
}

const char *
nixlBackendPluginHandle::getVersion() const {
    if (plugin_ && plugin_->get_plugin_version) {
        return plugin_->get_plugin_version();
    }
    return "unknown";
}

namespace {
// Backend plugin loader
std::shared_ptr<const nixlPluginHandle>
backendLoader(void *handle, const std::string &plugin_path) {
    // Get the initialization function
    typedef nixlBackendPlugin *(*init_func_t)();
    init_func_t init = (init_func_t)dlsym(handle, "nixl_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_plugin_init in " << plugin_path << ": " << dlerror();
        dlclose(handle);
        return nullptr;
    }

    // Call the initialization function
    nixlBackendPlugin *plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
        dlclose(handle);
        return nullptr;
    }

    // Check API version
    if (plugin->api_version != NIXL_PLUGIN_API_VERSION) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path << ": expected "
                   << NIXL_PLUGIN_API_VERSION << ", got " << plugin->api_version;
        dlclose(handle);
        return nullptr;
    }

    return std::make_shared<const nixlBackendPluginHandle>(handle, plugin);
}
} // namespace

nixlTelemetryPluginHandle::nixlTelemetryPluginHandle(void *handle, nixlTelemetryPlugin *plugin)
    : nixlPluginHandle(handle),
      plugin_(plugin) {}

nixlTelemetryPluginHandle::~nixlTelemetryPluginHandle() {
    if (handle_) {
        // Call the plugin's cleanup function
        typedef void (*fini_func_t)();
        fini_func_t fini = (fini_func_t)dlsym(handle_, "nixl_telemetry_plugin_fini");
        if (fini) {
            fini();
        }

        // Close the dynamic library
        dlclose(handle_);
        handle_ = nullptr;
        plugin_ = nullptr;
    }
}

std::unique_ptr<nixlTelemetryExporter>
nixlTelemetryPluginHandle::createExporter(
    const nixlTelemetryExporterInitParams &init_params) const {
    if (plugin_ && plugin_->create_exporter) {
        return plugin_->create_exporter(init_params);
    }
    return nullptr;
}

const char *
nixlTelemetryPluginHandle::getName() const {
    if (plugin_) {
        return plugin_->getName().c_str();
    }
    return "unknown";
}

const char *
nixlTelemetryPluginHandle::getVersion() const {
    if (plugin_) {
        return plugin_->getVersion().c_str();
    }
    return "unknown";
}

namespace {
// Telemetry plugin loader
std::shared_ptr<const nixlPluginHandle>
telemetryLoader(void *handle, const std::string &plugin_path) {
    // Get the initialization function
    typedef nixlTelemetryPlugin *(*init_func_t)();
    init_func_t init = (init_func_t)dlsym(handle, "nixl_telemetry_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_telemetry_plugin_init in " << plugin_path << ": "
                   << dlerror();
        dlclose(handle);
        return nullptr;
    }

    // Call the initialization function
    nixlTelemetryPlugin *plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
        dlclose(handle);
        return nullptr;
    }

    if (plugin->api_version != NIXL_TELEMETRY_PLUGIN_API_VERSION) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path << ": expected "
                   << static_cast<unsigned int>(NIXL_TELEMETRY_PLUGIN_API_VERSION) << ", got "
                   << static_cast<unsigned int>(plugin->api_version);
        dlclose(handle);
        return nullptr;
    }

    return std::make_shared<const nixlTelemetryPluginHandle>(handle, plugin);
}
} // namespace

nixlTracePluginHandle::nixlTracePluginHandle(void *handle, nixlTracePlugin *plugin)
    : nixlPluginHandle(handle),
      plugin_(plugin) {}

nixlTracePluginHandle::~nixlTracePluginHandle() {
    if (handle_) {
        typedef void (*fini_func_t)();
        fini_func_t fini = (fini_func_t)dlsym(handle_, "nixl_trace_plugin_fini");
        if (fini) {
            fini();
        }
        dlclose(handle_);
        handle_ = nullptr;
        plugin_ = nullptr;
    }
}

std::unique_ptr<nixl::trace::TraceBackend>
nixlTracePluginHandle::createBackend(const nixlTraceBackendInitParams &init_params) const {
    if (plugin_ && plugin_->create_backend) {
        return plugin_->create_backend(init_params);
    }
    return nullptr;
}

const char *
nixlTracePluginHandle::getName() const {
    if (plugin_) {
        return plugin_->getName().c_str();
    }
    return "unknown";
}

const char *
nixlTracePluginHandle::getVersion() const {
    if (plugin_) {
        return plugin_->getVersion().c_str();
    }
    return "unknown";
}

namespace {
// Trace plugin loader
std::shared_ptr<const nixlPluginHandle>
traceLoader(void *handle, const std::string &plugin_path) {
    typedef nixlTracePlugin *(*init_func_t)();
    init_func_t init = (init_func_t)dlsym(handle, "nixl_trace_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_trace_plugin_init in " << plugin_path << ": "
                   << dlerror();
        dlclose(handle);
        return nullptr;
    }

    nixlTracePlugin *plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
        dlclose(handle);
        return nullptr;
    }

    if (plugin->api_version != nixl_trace_plugin_api_version::V2) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path << ": expected "
                   << static_cast<unsigned int>(nixl_trace_plugin_api_version::V2) << ", got "
                   << static_cast<unsigned int>(plugin->api_version);
        dlclose(handle);
        return nullptr;
    }

    return std::make_shared<const nixlTracePluginHandle>(handle, plugin);
}
} // namespace

std::map<nixl_backend_t, std::string>
loadPluginList(const std::string &filename) {
    std::map<nixl_backend_t, std::string> plugins;
    std::ifstream file(filename);

    if (!file.is_open()) {
        NIXL_ERROR << "Failed to open plugin list file: " << filename;
        return plugins;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Find the equals sign
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string name = line.substr(0, pos);
            std::string path = line.substr(pos + 1);

            auto trim = [](std::string &s) {
                const char *whitespace = " \t\n\r\f\v";
                s.erase(0, s.find_first_not_of(whitespace));
                s.erase(s.find_last_not_of(whitespace) + 1);
            };
            trim(name);
            trim(path);
            // "NAME=" without a path is legitimate: it only registers the
            // name as discoverable. Reject entries without a name.
            if (name.empty()) {
                continue;
            }

            // Add to map
            plugins[name] = path;
        }
    }

    return plugins;
}

namespace {
/* ASCII-only upper casing: locale-independent, and plugin names are
 * expected to be ASCII. */
std::string
upperAscii(std::string s) {
    for (char &c : s) {
        if (c >= 'a' && c <= 'z') {
            c -= 'a' - 'A';
        }
    }
    return s;
}

/* Outcome of looking for an entry whose file name is exactly `filename`.
 * exists() is not enough: on case-insensitive file systems it succeeds for
 * case variants and would let one library load under two spellings. A
 * directory that cannot be enumerated is reported separately from one that
 * was enumerated and holds no such entry. */
using EntryLookup = nixlPluginManager::EntryLookup;

EntryLookup
lookupExactEntry(const std::string &dir, const std::string &filename) {
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec), end;
    if (ec) {
        return EntryLookup::CANNOT_LIST;
    }
    // increment(ec) makes the iterator equal to end when it fails, so the
    // error must be checked immediately rather than at the top of the next
    // iteration (which never runs).
    for (; it != end;) {
        if (it->path().filename().string() == filename) {
            return EntryLookup::MATCH;
        }
        it.increment(ec);
        if (ec) {
            return EntryLookup::CANNOT_LIST;
        }
    }
    return EntryLookup::NO_MATCH;
}

/* Whether the loader's compatibility branch applies: only for a directory
 * that cannot be enumerated at all, whose scan really saw the standard file
 * name (a whitespace variant never proves one), and whose standard path
 * still verifies. */
bool
standardPathFallback(EntryLookup lookup, bool scan_saw_standard, bool path_exists) {
    return lookup == EntryLookup::CANNOT_LIST && scan_saw_standard && path_exists;
}

bool
shouldDeepBindPlugin(const std::string &plugin_name) {
    // Case-insensitive so a UCX plugin registered under a different
    // spelling still gets the deep-bind treatment.
    if (upperAscii(plugin_name) != upperAscii(kUcxPluginName)) {
        return false;
    }

    try {
        /* TODO: check if RTLD_DEEPBIND is needed at all for UCX/NIXL */
        return nixl::config::getValueDefaulted<bool>(kUcxDeepBindVar, false);
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Invalid " << kUcxDeepBindVar
                  << " value, enabling RTLD_DEEPBIND: " << e.what();
        return true;
    }
}
} // namespace

std::shared_ptr<const nixlPluginHandle>
nixlPluginManager::loadPluginFromPath(const std::string &plugin_path,
                                      nixlPluginLoaderFunc loader,
                                      bool deepbind) {
    // Open the plugin file with RTLD_NODELETE to prevent glibc from physically unloading
    // the library on dlclose. This is required because plugins link dynamically against Abseil,
    // which uses thread_local and static initialization that are unsafe to unload dynamically
    // and trigger glibc bugs on older versions (e.g. Ubuntu 22.04 / glibc 2.35).
    int flags = RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE;
    if (deepbind) {
#ifdef RTLD_DEEPBIND
        flags |= RTLD_DEEPBIND;
#else
        NIXL_WARN << "RTLD_DEEPBIND requested for " << plugin_path
                  << " but is not supported on this platform";
#endif
    }

    void *handle = dlopen(plugin_path.c_str(), flags);
    if (!handle) {
        NIXL_INFO << "Failed to load plugin from " << plugin_path << ": " << dlerror();
        return nullptr;
    }

    return loader(handle, plugin_path);
}

void
nixlPluginManager::discoverPluginsFromList(const std::string &filename) {
    auto plugins = loadPluginList(filename);

    const std::lock_guard lock(mutex_);

    for (const auto& pair : plugins) {
        const std::string& name = pair.first;
        const std::string& path = pair.second;

        // Record the source even when the plugin is already loaded, so the
        // name survives a later unload.
        discovered_backend_plugins_[name].from_plugin_list = true;
        if (!path.empty()) {
            explicit_plugin_paths_[name] = path;
            NIXL_INFO << "Discovered backend plugin from list: " << name << " (" << path << ")";
        } else {
            NIXL_INFO << "Discovered backend plugin from list: " << name;
        }
    }
}

namespace {
std::string
getPluginDir() {
    // Environment variable takes precedence
    if (const auto plugin_dir = nixl::config::getValueOptional<std::string>("NIXL_PLUGIN_DIR")) {
        return *plugin_dir;
    }

    // By default, use the plugin directory relative to the binary
    Dl_info info;
    int ok = dladdr(reinterpret_cast<void *>(&getPluginDir), &info);
    if (!ok) {
        NIXL_ERROR << "Failed to get plugin directory from dladdr";
        return "";
    }
    return (std::filesystem::path(info.dli_fname).parent_path() / "plugins").string();
}
} // namespace

// PluginManager implementation
nixlPluginManager::nixlPluginManager() {
    // Force levels right before logging
#ifdef NIXL_USE_PLUGIN_FILE
    NIXL_DEBUG << "Loading plugins from file: " << NIXL_USE_PLUGIN_FILE;
    std::string plugin_file = NIXL_USE_PLUGIN_FILE;
    if (std::filesystem::exists(plugin_file)) {
        discoverPluginsFromList(plugin_file);
    }
#endif

    std::string plugin_dir = getPluginDir();
    if (!plugin_dir.empty()) {
        NIXL_DEBUG << "Loading plugins from: " << plugin_dir;
        uint64_t generation = 0;
        {
            const std::lock_guard lock(mutex_);
            plugin_dirs_.insert(plugin_dirs_.begin(), plugin_dir);
            generation = next_scan_generation_++;
            dir_scan_generations_[plugin_dir] = generation;
        }
        discoverPluginsFromDir(plugin_dir, generation);
    }

    registerBuiltinPlugins();
}

nixlPluginManager& nixlPluginManager::getInstance() {
    // Meyers singleton initialization is safe in multi-threaded environment.
    // Consult standard [stmt.dcl] chapter for details.
    static nixlPluginManager instance;

    return instance;
}

void
nixlPluginManager::addPluginDirectory(const std::string &directory) {
    if (directory.empty()) {
        NIXL_ERROR << "Cannot add empty plugin directory";
        return;
    }

    // Check if directory exists
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        NIXL_ERROR << "Plugin directory does not exist or is not readable: " << directory;
        return;
    }

    uint64_t generation = 0;
    {
        const std::lock_guard lock(mutex_);

        // Check if directory is already in the list
        for (const auto& dir : plugin_dirs_) {
            if (dir == directory) {
                NIXL_WARN << "Plugin directory already registered: " << directory;
                return;
            }
        }

        // Prioritize the new directory by inserting it at the beginning
        plugin_dirs_.insert(plugin_dirs_.begin(), directory);
        generation = next_scan_generation_++;
        dir_scan_generations_[directory] = generation;
    }

    discoverPluginsFromDir(directory, generation);
}

std::vector<std::string>
nixlPluginManager::getPluginDirectories() const {
    const std::lock_guard lock(mutex_);
    return plugin_dirs_;
}

void
nixlPluginManager::removePluginDirectoryForUnitTest(const std::string &directory) {
    const std::lock_guard lock(mutex_);

    plugin_dirs_.erase(std::remove(plugin_dirs_.begin(), plugin_dirs_.end(), directory),
                       plugin_dirs_.end());
    dir_scan_generations_.erase(directory);
    for (auto it = dir_override_paths_.begin(); it != dir_override_paths_.end();) {
        it->second.erase(directory);
        if (it->second.empty()) {
            it = dir_override_paths_.erase(it);
        } else {
            ++it;
        }
    }
    // Forget names this directory was the only source of; pluginlist-sourced
    // names survive.
    for (auto it = discovered_backend_plugins_.begin(); it != discovered_backend_plugins_.end();) {
        it->second.dirs.erase(directory);
        it->second.standard_dirs.erase(directory);
        if (!it->second.from_plugin_list && it->second.dirs.empty()) {
            it = discovered_backend_plugins_.erase(it);
        } else {
            ++it;
        }
    }
}

void
nixlPluginManager::discoverBackendPluginWithGenerationForUnitTest(const std::filesystem::path &path,
                                                                  const std::string &dir,
                                                                  uint64_t generation) {
    discoverBackendPlugin(path, dir, generation);
}

uint64_t
nixlPluginManager::pluginDirectoryScanGenerationForUnitTest(const std::string &dir) const {
    const std::lock_guard lock(mutex_);
    const auto it = dir_scan_generations_.find(dir);
    return it == dir_scan_generations_.end() ? 0 : it->second;
}

bool
nixlPluginManager::standardPathFallbackForUnitTest(EntryLookup lookup,
                                                   bool scan_saw_standard,
                                                   bool path_exists) {
    return standardPathFallback(lookup, scan_saw_standard, path_exists);
}

void
nixlPluginManager::setPreLoadHookForUnitTest(std::function<void()> hook) {
    const std::lock_guard lock(mutex_);
    pre_load_hook_for_test_ = std::move(hook);
}

void
nixlPluginManager::forgetDiscoveredPluginForUnitTest(const nixl_backend_t &name) {
    const std::lock_guard lock(mutex_);
    discovered_backend_plugins_.erase(name);
    dir_override_paths_.erase(name);
    explicit_plugin_paths_.erase(name);
}

std::string
nixlPluginManager::composePluginPath(const std::string &dir,
                                     const std::string &plugin_prefix,
                                     const std::string &plugin_name) {
    std::string plugin_path;
    if (dir.empty()) {
        return "";
    } else if (dir.back() == '/') {
        plugin_path = dir + plugin_prefix + plugin_name + ".so";
    } else {
        plugin_path = dir + "/" + plugin_prefix + plugin_name + ".so";
    }
    return plugin_path;
}

std::shared_ptr<const nixlBackendPluginHandle>
nixlPluginManager::loadBackendPlugin(const std::string &plugin_name,
                                     nixl_backend_t *canonical_name) {
    // One-shot test hook (swapped out under the lock, run without it): fires
    // after the caller's own name resolution and before this load's.
    std::function<void()> hook;
    {
        const std::lock_guard lock(mutex_);
        hook.swap(pre_load_hook_for_test_);
    }
    if (hook) {
        hook();
    }

    const std::lock_guard lock(mutex_);

    const std::string canonical = canonicalBackendNameLocked(plugin_name);
    if (canonical_name != nullptr) {
        *canonical_name = canonical;
    }
    if (canonical.empty()) {
        return nullptr;
    }
    const auto plugin_handle = loadBackendPluginImpl(canonical);
    if (plugin_handle) {
        loaded_backend_plugins_.try_emplace(canonical, plugin_handle);
    }
    return plugin_handle;
}

nixl_backend_t
nixlPluginManager::canonicalBackendName(const nixl_backend_t &name) const {
    const std::lock_guard lock(mutex_);
    return canonicalBackendNameLocked(name);
}

nixl_backend_t
nixlPluginManager::canonicalBackendNameLocked(const nixl_backend_t &name) const {
    // Trim ASCII whitespace from both ends.
    static constexpr const char *whitespace = " \t\n\r\f\v";
    const auto first = name.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return "";
    }
    const auto last = name.find_last_not_of(whitespace);
    const std::string trimmed = name.substr(first, last - first + 1);

    // An exact match always wins, so distinctly registered spellings (a
    // pluginlist may legitimately contain both "FOO" and "foo") keep loading
    // their own library.
    if (loaded_backend_plugins_.count(trimmed) != 0 || explicit_plugin_paths_.count(trimmed) != 0 ||
        discovered_backend_plugins_.count(trimmed) != 0) {
        return trimmed;
    }

    // The trimmed spelling may name a standard plugin file that a directory
    // scan has not published yet (scans run outside the lock); folding to
    // another spelling then would load the wrong plugin, while main probed
    // the requested spelling directly. The exact entry name decides, so a
    // case variant in a case-insensitive directory cannot capture the
    // request. Probing under the lock matches loadBackendPluginImpl, which
    // already dlopens while holding it.
    const std::string wanted_file = backendPluginPrefix + trimmed + kPluginSuffix;
    for (const auto &dir : plugin_dirs_) {
        if (lookupExactEntry(dir, wanted_file) == EntryLookup::MATCH) {
            return trimmed;
        }
    }

    // Callers may pass "ucx", "Ucx" or " UCX " for the UCX backend, so fall
    // back to a case-insensitive match against every known plugin name.
    const std::string wanted = upperAscii(trimmed);
    const auto matches = [&](const std::string &candidate) {
        return upperAscii(candidate) == wanted;
    };

    // Registration decides between same-fold spellings, never what happens to
    // be loaded: the loaded set grows over time, and preferring it would make
    // one agent's repeated alias request resolve differently once another
    // agent loads a case variant, slipping past the duplicate check. Loaded
    // spellings remain a last resort for plugins whose directories were
    // removed after they loaded.
    for (const auto &pair : explicit_plugin_paths_) {
        if (matches(pair.first)) {
            return pair.first;
        }
    }
    for (const auto &known : discovered_backend_plugins_) {
        if (matches(known.first)) {
            return known.first;
        }
    }
    for (const auto &pair : loaded_backend_plugins_) {
        if (matches(pair.first)) {
            return pair.first;
        }
    }
    return trimmed;
}

std::shared_ptr<const nixlBackendPluginHandle>
nixlPluginManager::loadBackendPluginImpl(const std::string &plugin_name) const {
    // Whitespace-only and empty names never name a plugin.
    if (plugin_name.empty()) {
        return nullptr;
    }

    // Check if the plugin is already loaded
    // Static Plugins are preloaded so return handle
    auto it = loaded_backend_plugins_.find(plugin_name);
    if (it != loaded_backend_plugins_.end()) {
        return it->second;
    }

    // Try the explicit path from the plugin list file first
    auto path_it = explicit_plugin_paths_.find(plugin_name);
    if (path_it != explicit_plugin_paths_.end()) {
        const std::string &plugin_path = path_it->second;
        if (std::filesystem::exists(plugin_path)) {
            auto plugin_handle =
                loadPluginFromPath(plugin_path, backendLoader, shouldDeepBindPlugin(plugin_name));
            if (plugin_handle) {
                return std::dynamic_pointer_cast<const nixlBackendPluginHandle>(plugin_handle);
            }
        }
    }

    // Try to load the plugin from all registered directories. The exact
    // directory entry decides, so on case-insensitive file systems a case
    // variant in a higher-priority directory cannot capture a request whose
    // exact file lives in a lower-priority one, and one library cannot be
    // cached under two spellings. A directory that can no longer be
    // enumerated at all (search permission without read) keeps serving a
    // standard file name its scan really saw, verified by path; a whitespace
    // variant never proves a standard name, and an enumeration that completed
    // without the entry always wins over the stale scan record. Suppress the
    // missing-file warning while fallback candidates remain (a plugin behind
    // a non-canonical file name has no standard file to find here).
    const auto override_it = dir_override_paths_.find(plugin_name);
    const bool has_override = override_it != dir_override_paths_.end();
    const std::string standard_file = backendPluginPrefix + plugin_name + kPluginSuffix;
    const auto discovered_it = discovered_backend_plugins_.find(plugin_name);
    for (const auto& dir : plugin_dirs_) {
        const std::string plugin_path = composePluginPath(dir, backendPluginPrefix, plugin_name);
        const EntryLookup lookup = lookupExactEntry(dir, standard_file);
        bool use_path = lookup == EntryLookup::MATCH;
        if (!use_path) {
            const bool scan_saw_standard = discovered_it != discovered_backend_plugins_.end() &&
                discovered_it->second.standard_dirs.count(dir) != 0;
            // The path probe runs only when the compatibility branch could
            // apply at all, never on an ordinary miss.
            if (lookup == EntryLookup::CANNOT_LIST && scan_saw_standard) {
                std::error_code ec;
                use_path = standardPathFallback(
                    lookup, scan_saw_standard, std::filesystem::exists(plugin_path, ec));
            }
        }
        if (!use_path) {
            if (!has_override) {
                NIXL_WARN << "Plugin file does not exist: " << plugin_path;
            }
            continue;
        }

        auto plugin_handle =
            loadPluginFromPath(plugin_path, backendLoader, shouldDeepBindPlugin(plugin_name));
        if (plugin_handle) {
            return std::dynamic_pointer_cast<const nixlBackendPluginHandle>(plugin_handle);
        }
    }

    // Last resort: a plugin discovered behind a non-canonical file name
    // (e.g. "libplugin_ Foo .so"). Walk the candidates in plugin_dirs_ order
    // so priority follows directory registration, never scan completion.
    if (has_override) {
        for (const auto &dir : plugin_dirs_) {
            const auto cand = override_it->second.find(dir);
            if (cand == override_it->second.end() || !std::filesystem::exists(cand->second)) {
                continue;
            }
            auto plugin_handle =
                loadPluginFromPath(cand->second, backendLoader, shouldDeepBindPlugin(plugin_name));
            if (plugin_handle) {
                return std::dynamic_pointer_cast<const nixlBackendPluginHandle>(plugin_handle);
            }
        }
    }

    // Failed to load the plugin
    NIXL_INFO << "Failed to load plugin '" << plugin_name << "' from any directory";
    return nullptr;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlPluginManager::loadTelemetryPlugin(const std::string &plugin_name) {
    const std::lock_guard lock(mutex_);

    // Check if the plugin is already loaded
    auto it = loaded_telemetry_plugins_.find(plugin_name);
    if (it != loaded_telemetry_plugins_.end()) {
        return it->second;
    }

    const std::string &file_name =
        plugin_name == prometheus_plugin_name ? prometheus_mp_plugin_name : plugin_name;
    it = loaded_telemetry_plugins_.find(file_name);
    if (it != loaded_telemetry_plugins_.end()) {
        const auto telemetry_plugin = it->second;
        loaded_telemetry_plugins_[plugin_name] = telemetry_plugin;
        return telemetry_plugin;
    }

    // Try to load the plugin from all registered directories
    for (const auto &dir : plugin_dirs_) {
        std::string plugin_path = composePluginPath(dir, telemetryPluginPrefix, file_name);
        if (plugin_path.empty()) {
            continue;
        }

        if (!std::filesystem::exists(plugin_path)) {
            NIXL_WARN << "Plugin file does not exist: " << plugin_path;
            continue;
        }

        auto plugin_handle = loadPluginFromPath(plugin_path, telemetryLoader);
        if (plugin_handle) {
            auto telemetry_plugin =
                std::dynamic_pointer_cast<const nixlTelemetryPluginHandle>(plugin_handle);
            loaded_telemetry_plugins_[file_name] = telemetry_plugin;
            loaded_telemetry_plugins_[plugin_name] = telemetry_plugin;
            return telemetry_plugin;
        }
    }

    NIXL_INFO << "Failed to load plugin '" << plugin_name << "' from any directory";
    return nullptr;
}

std::shared_ptr<const nixlTracePluginHandle>
nixlPluginManager::loadTracePlugin(const std::string &plugin_name) {
    const std::lock_guard lock(mutex_);

    // Check if the plugin is already loaded
    auto it = loaded_trace_plugins_.find(plugin_name);
    if (it != loaded_trace_plugins_.end()) {
        return it->second;
    }

    // Try to load the plugin from all registered directories
    for (const auto &dir : plugin_dirs_) {
        std::string plugin_path = composePluginPath(dir, tracePluginPrefix, plugin_name);
        if (plugin_path.empty() || !std::filesystem::exists(plugin_path)) {
            continue;
        }

        auto plugin_handle = loadPluginFromPath(plugin_path, traceLoader);
        if (plugin_handle) {
            auto trace_plugin =
                std::dynamic_pointer_cast<const nixlTracePluginHandle>(plugin_handle);
            loaded_trace_plugins_[plugin_name] = trace_plugin;
            return trace_plugin;
        }
    }

    NIXL_INFO << "Failed to load trace plugin '" << plugin_name << "' from any directory";
    return nullptr;
}

namespace {
static bool
startsWith(const std::string &str, const std::string &prefix) {
    return str.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), str.begin());
}

static bool
endsWith(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() && std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

static std::string
extractPluginName(const std::string &filename, const std::string &prefix) {
    return filename.substr(prefix.size(), filename.size() - prefix.size() - kPluginSuffix.size());
}
} // namespace

void
nixlPluginManager::discoverBackendPlugin(const std::filesystem::path &path,
                                         const std::string &dir_key,
                                         uint64_t generation) {
    const std::string filename = path.filename().string();
    if (startsWith(filename, backendPluginPrefix) && endsWith(filename, kPluginSuffix)) {
        std::string plugin_name = extractPluginName(filename, backendPluginPrefix);

        // Trim ASCII whitespace. A discovered name is only usable when it
        // equals its trimmed form, because loading composes the file name
        // from the plugin name; "libplugin_ .so" is skipped, and
        // "libplugin_ Foo .so" is registered by its real path.
        static constexpr const char *whitespace = " \t\n\r\f\v";
        const auto first = plugin_name.find_first_not_of(whitespace);
        if (first == std::string::npos) {
            return;
        }
        const auto last = plugin_name.find_last_not_of(whitespace);
        const std::string trimmed = plugin_name.substr(first, last - first + 1);

        const std::lock_guard lock(mutex_);
        // Discard the scan when the directory was deregistered, or
        // deregistered and re-registered, while this scan was in flight.
        const auto gen_it = dir_scan_generations_.find(dir_key);
        if (gen_it == dir_scan_generations_.end() || gen_it->second != generation) {
            return;
        }

        // Record the source even when the plugin is already loaded, so the
        // name survives a later unload. A non-canonical file name additionally
        // records its real path as a fallback candidate, without overriding a
        // pluginlist-specified library or the standard file name, which the
        // loader prefers. dir_key is the registered directory string, so
        // discovery, loading and removal all share one key regardless of
        // trailing slashes or other spelling differences.
        discovered_backend_plugins_[trimmed].dirs.insert(dir_key);
        if (trimmed != plugin_name) {
            const auto inserted = dir_override_paths_[trimmed].emplace(dir_key, path.string());
            if (!inserted.second) {
                NIXL_WARN << "Multiple non-canonical files for plugin '" << trimmed << "' in "
                          << dir_key << "; keeping " << inserted.first->second;
            }
        } else {
            // The scan saw the standard file name itself; only that proves
            // the on-disk spelling for the unlistable-directory fallback in
            // the loader.
            discovered_backend_plugins_[trimmed].standard_dirs.insert(dir_key);
        }
        NIXL_INFO << "Discovered backend plugin: " << trimmed;
    }
}

void
nixlPluginManager::discoverTelemetryPlugin(const std::string &filename) {
    if (startsWith(filename, telemetryPluginPrefix) && endsWith(filename, kPluginSuffix)) {
        std::string plugin_name = extractPluginName(filename, telemetryPluginPrefix);
        NIXL_INFO << "Discovered telemetry plugin: " << plugin_name;
    }
}

void
nixlPluginManager::discoverTracePlugin(const std::string &filename) {
    if (startsWith(filename, tracePluginPrefix) && endsWith(filename, kPluginSuffix)) {
        std::string plugin_name = extractPluginName(filename, tracePluginPrefix);
        NIXL_INFO << "Discovered trace plugin: " << plugin_name;
    }
}

void
nixlPluginManager::discoverPluginsFromDir(const std::string &dir, uint64_t generation) {
    std::error_code ec;
    std::filesystem::directory_iterator dir_iter(dir, ec);
    if (ec) {
        NIXL_ERROR << "Error accessing directory(" << dir << "): " << ec.message();
        return;
    }

    for (const auto& entry : dir_iter) {
        std::string filename = entry.path().filename().string();
        discoverBackendPlugin(entry.path(), dir, generation);
        discoverTelemetryPlugin(filename);
        discoverTracePlugin(filename);
    }
}

void
nixlPluginManager::unloadBackendPluginForUnitTest(const nixl_backend_t &plugin_name) {
    // Resolve to the canonical name so a plugin loaded through one spelling
    // can be unloaded through another.
    const nixl_backend_t canonical = canonicalBackendName(plugin_name);

    // Do not unload static plugins
    for (const auto &splugin : getBackendStaticPlugins()) {
        if (splugin.name == canonical) {
            return;
        }
    }

    const std::lock_guard lock(mutex_);
    loaded_backend_plugins_.erase(canonical);
}

nixl_status_t
nixlPluginManager::getBackendParams(const nixl_backend_t &type,
                                    nixl_mem_list_t &mems,
                                    nixl_b_params_t &params) const {
    const std::lock_guard lock(mutex_);

    if (const auto plugin = loadBackendPluginImpl(canonicalBackendNameLocked(type))) {
        mems = plugin->getBackendMems();
        params = plugin->getBackendOptions();
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_NOT_FOUND;
}

std::shared_ptr<const nixlBackendPluginHandle>
nixlPluginManager::getBackendPlugin(const nixl_backend_t &plugin_name) {
    const std::lock_guard lock(mutex_);

    auto it = loaded_backend_plugins_.find(canonicalBackendNameLocked(plugin_name));
    if (it != loaded_backend_plugins_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlPluginManager::getTelemetryPlugin(const nixl_telemetry_plugin_t &plugin_name) {
    const std::lock_guard lock(mutex_);
    auto it = loaded_telemetry_plugins_.find(plugin_name);
    if (it != loaded_telemetry_plugins_.end()) {
        return it->second;
    }
    return nullptr;
}

nixl_b_params_t
nixlBackendPluginHandle::getBackendOptions() const {
    nixl_b_params_t params;
    if (plugin_ && plugin_->get_backend_options) {
        return plugin_->get_backend_options();
    }
    return params; // Return empty params if not implemented
}

nixl_mem_list_t
nixlBackendPluginHandle::getBackendMems() const {
    nixl_mem_list_t mems;
    if (plugin_ && plugin_->get_backend_mems) {
        return plugin_->get_backend_mems();
    }
    return mems; // Return empty mems if not implemented
}

std::vector<nixl_backend_t>
nixlPluginManager::getLoadedBackendPluginNames() {
    const std::lock_guard lock(mutex_);

    std::vector<nixl_backend_t> names;
    for (const auto &pair : loaded_backend_plugins_) {
        names.push_back(pair.first);
    }
    return names;
}

std::vector<nixl_backend_t>
nixlPluginManager::getAvailBackendPluginNames() {
    const std::lock_guard lock(mutex_);

    std::vector<nixl_backend_t> names;
    for (const auto &pair : loaded_backend_plugins_) {
        names.push_back(pair.first);
    }
    for (const auto &name : discovered_backend_plugins_) {
        // Skip discovered plugins that are already loaded to avoid duplicates
        if (loaded_backend_plugins_.find(name.first) == loaded_backend_plugins_.end()) {
            names.push_back(name.first);
        }
    }
    return names;
}

std::vector<nixl_telemetry_plugin_t>
nixlPluginManager::getLoadedTelemetryPluginNames() {
    const std::lock_guard lock(mutex_);

    std::vector<nixl_telemetry_plugin_t> names;
    for (const auto &pair : loaded_telemetry_plugins_) {
        names.push_back(pair.first);
    }
    return names;
}

void
nixlPluginManager::registerBackendStaticPlugin(const std::string &name,
                                               nixlStaticPluginCreatorFunc creator) {
    const std::lock_guard lock(mutex_);

    nixlBackendStaticPluginInfo info;
    info.name = name;
    info.createFunc = creator;
    backend_static_plugins_.push_back(info);

    //Static Plugins are considered pre-loaded
    nixlBackendPlugin* plugin = info.createFunc();
    NIXL_INFO << "Loading static plugin: " << name;
    if (plugin) {
        // Register the loaded plugin
        auto plugin_handle = std::make_shared<const nixlBackendPluginHandle>(nullptr, plugin);
        loaded_backend_plugins_[name] = plugin_handle;
    }
}

void
nixlPluginManager::registerTelemetryStaticPlugin(const std::string &name,
                                                 nixlTelemetryStaticPluginCreatorFunc creator) {
    const std::lock_guard lock(mutex_);

    nixlTelemetryStaticPluginInfo info;
    info.name = name;
    info.createFunc = creator;
    telemetry_static_plugins_.push_back(info);

    // Static Plugins are considered pre-loaded
    nixlTelemetryPlugin *plugin = info.createFunc();
    NIXL_INFO << "Loading static plugin: " << name;
    if (plugin) {
        // Register the loaded plugin
        auto plugin_handle = std::make_shared<const nixlTelemetryPluginHandle>(nullptr, plugin);
        loaded_telemetry_plugins_[name] = plugin_handle;
    }
}

const std::vector<nixlBackendStaticPluginInfo> &
nixlPluginManager::getBackendStaticPlugins() {
    return backend_static_plugins_;
}

const std::vector<nixlTelemetryStaticPluginInfo> &
nixlPluginManager::getTelemetryStaticPlugins() {
    return telemetry_static_plugins_;
}

#define NIXL_REGISTER_STATIC_PLUGIN(plugin_type, name)              \
    extern nixl##plugin_type##Plugin *createStatic##name##Plugin(); \
    register##plugin_type##StaticPlugin(#name, createStatic##name##Plugin);

void nixlPluginManager::registerBuiltinPlugins() {
#ifdef STATIC_PLUGIN_LIBFABRIC
    NIXL_REGISTER_STATIC_PLUGIN(Backend, LIBFABRIC)
#endif

#ifdef STATIC_PLUGIN_UCX
    NIXL_REGISTER_STATIC_PLUGIN(Backend, UCX)
#endif

#ifdef STATIC_PLUGIN_GDS
#ifndef DISABLE_GDS_BACKEND
    NIXL_REGISTER_STATIC_PLUGIN(Backend, GDS)
#endif
#endif

#ifdef STATIC_PLUGIN_GDS_MT
    NIXL_REGISTER_STATIC_PLUGIN(Backend, GDS_MT)
#endif

#ifdef STATIC_PLUGIN_AIS
#ifndef DISABLE_AIS_BACKEND
    NIXL_REGISTER_STATIC_PLUGIN(Backend, AIS)
#endif
#endif

#ifdef STATIC_PLUGIN_POSIX
    NIXL_REGISTER_STATIC_PLUGIN(Backend, POSIX)
#endif

#ifdef STATIC_PLUGIN_GPUNETIO
    NIXL_REGISTER_STATIC_PLUGIN(Backend, GPUNETIO)
#endif

#ifdef STATIC_PLUGIN_OBJ
    NIXL_REGISTER_STATIC_PLUGIN(Backend, OBJ)
#endif

#ifdef STATIC_PLUGIN_REDIS
    NIXL_REGISTER_STATIC_PLUGIN(Backend, REDIS)
#endif

#ifdef STATIC_PLUGIN_MOONCAKE
    NIXL_REGISTER_STATIC_PLUGIN(Backend, MOONCAKE)
#endif

#ifdef STATIC_PLUGIN_HF3FS
    NIXL_REGISTER_STATIC_PLUGIN(Backend, HF3FS)
#endif

#ifdef STATIC_PLUGIN_INFINIA
    NIXL_REGISTER_STATIC_PLUGIN(Backend, INFINIA)
#endif
#ifdef STATIC_PLUGIN_DOCA_MEMOS
    NIXL_REGISTER_STATIC_PLUGIN(Backend, DOCA_MEMOS)
#endif

    NIXL_REGISTER_STATIC_PLUGIN(Telemetry, BUFFER)
    NIXL_REGISTER_STATIC_PLUGIN(Telemetry, NOP)
}
