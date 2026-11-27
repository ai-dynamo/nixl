/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace nixlbench {

// A private, unique directory keeps concurrent runs and prior manifests separate.
// Only a fully registered key set is published; failed setup removes its own files.
class MemosKeyManifest {
    std::string directory_;
    std::string temporary_;
    std::string final_;
    std::ofstream stream_;
    bool published_ = false;

    void
    cleanup() noexcept {
        if (stream_.is_open()) {
            stream_.close();
        }
        if (!published_) {
            if (!temporary_.empty()) {
                ::unlink(temporary_.c_str());
            }
            ::rmdir(directory_.c_str());
        }
    }

public:
    explicit MemosKeyManifest(const std::string &parent = ".")
        : directory_(parent + "/memos-keys-XXXXXX") {
        if (::mkdtemp(directory_.data()) == nullptr) {
            throw std::runtime_error("cannot create unique MEMOS manifest directory");
        }
        try {
            temporary_ = directory_ + "/memos-keys.txt.tmp";
            final_ = directory_ + "/memos-keys.txt";
            stream_.open(temporary_);
            if (!stream_) {
                throw std::runtime_error("cannot create temporary MEMOS key manifest");
            }
        }
        catch (...) {
            cleanup();
            throw;
        }
    }

    MemosKeyManifest(const MemosKeyManifest &) = delete;
    MemosKeyManifest &
    operator=(const MemosKeyManifest &) = delete;

    ~MemosKeyManifest() {
        cleanup();
    }

    void
    append(const std::string &key) {
        if (published_) {
            throw std::runtime_error("MEMOS key manifest is already published");
        }
        stream_ << key << '\n';
        if (!stream_) {
            throw std::runtime_error("cannot write MEMOS key manifest");
        }
    }

    void
    publish() {
        if (published_) {
            throw std::runtime_error("MEMOS key manifest is already published");
        }
        stream_.close();
        if (!stream_) {
            throw std::runtime_error("cannot close MEMOS key manifest");
        }
        if (::rename(temporary_.c_str(), final_.c_str()) != 0) {
            throw std::runtime_error("cannot publish MEMOS key manifest");
        }
        published_ = true;
    }

    const std::string &
    path() const {
        return final_;
    }
};

} // namespace nixlbench
