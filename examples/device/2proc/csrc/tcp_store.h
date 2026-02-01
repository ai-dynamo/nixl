/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <memory>

namespace c10d {
class TCPStore;
}

namespace tcp_store {

/**
 * Simple wrapper around PyTorch's TCPStore for metadata exchange.
 */
class TCPStore {
public:
    /**
     * Create a TCPStore instance.
     *
     * @param host Host address (e.g., "127.0.0.1")
     * @param port Port number (e.g., 9998)
     * @param is_master Whether this process should start the server
     * @param timeout_ms Timeout in milliseconds (default: 30000)
     */
    TCPStore(const std::string &host, int port, bool is_master, int timeout_ms = 30000);
    ~TCPStore();

    // Prevent copying
    TCPStore(const TCPStore &) = delete;
    TCPStore &operator=(const TCPStore &) = delete;

    /**
     * Set a key-value pair in the store.
     *
     * @param key Key to set
     * @param value Value to store (binary safe)
     */
    void
    set(const std::string &key, const std::string &value);

    /**
     * Get a value from the store.
     *
     * @param key Key to retrieve
     * @return Value associated with the key
     */
    std::string
    get(const std::string &key);

    /**
     * Wait for a key to be available in the store.
     *
     * @param key Key to wait for
     * @param timeout_ms Timeout in milliseconds
     * @return true if key is available, false on timeout
     */
    bool
    wait(const std::string &key, int timeout_ms = 30000);

    /**
     * Delete a key from the store.
     *
     * @param key Key to delete
     */
    void
    delete_key(const std::string &key);

private:
    std::shared_ptr<c10d::TCPStore> store_;
};

} // namespace tcp_store
