/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_RUNTIME_ASIO_RUNTIME_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_RUNTIME_ASIO_RUNTIME_H

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <asio.hpp>

#include "runtime.h"

enum class asio_msg_type_t : char {
    INTEGER = 'i',
    INTEGER_ARRAY = 'a',
    STRING = 's',
    REDUCE = 'r',
    BARRIER = 'b'
};

struct xferBenchAsioIncoming {
    asio_msg_type_t type;
    std::string data;
};

class xferBenchAsioRT : public xferBenchRT {
public:
    xferBenchAsioRT(const std::string &ip, const std::uint16_t port)
        : endpoint_(asio::ip::address::from_string(ip), asio::ip::port_type(port)),
          timer_(context_, asio::chrono::seconds(25)),
          acceptor_(attemptAcceptor()),
          thread_(&xferBenchAsioRT::main, this) {
        setSize(2);
        setRank(int(!bool(acceptor_)));
    }

    ~xferBenchAsioRT() {
        context_.stop();
        thread_.join();
    }

    int
    sendInt(int *buffer, int dest_rank) override {
        assert(1 - getRank() == dest_rank);
        postSend(asio_msg_type_t::INTEGER, buffer, sizeof(int));
        return 0;
    }

    int
    recvInt(int *buffer, int src_rank) override {
        assert(1 - getRank() == src_rank);
        recvWait(asio_msg_type_t::INTEGER, [&](const std::string &data) {
            assert(data.size() == sizeof(int));
            std::memcpy(buffer, data.data(), data.size());
            return true;
        });
        return 0;
    }

    int
    broadcastInt(int *buffer, size_t count, int root_rank) override {
        assert(count <= (1 << 28));
        if (getRank() == root_rank) {
            postSend(asio_msg_type_t::INTEGER_ARRAY, buffer, count * sizeof(int));
        } else {
            recvWait(asio_msg_type_t::INTEGER_ARRAY, [&](const std::string &data) {
                if (data.size() == (count * sizeof(int))) {
                    std::memcpy(buffer, data.data(), data.size());
                    return true;
                }
                return false;
            });
        }
        return 0;
    }

    int
    sendChar(char *buffer, size_t count, int dest_rank) override {
        assert(count <= (1 << 30));
        assert(1 - getRank() == dest_rank);
        postSend(asio_msg_type_t::STRING, buffer, count);
        return 0;
    }

    int
    recvChar(char *buffer, size_t count, int src_rank) override {
        assert(count <= (1 << 30));
        assert(1 - getRank() == src_rank);
        recvWait(asio_msg_type_t::STRING, [&](const std::string &data) {
            // Replicate std::min from ETCD runtime.
            std::memcpy(buffer, data.data(), std::min(data.size(), count));
            return true;
        });
        return 0;
    }

    int
    reduceSumDouble(double *local_value, double *global_value, int dest_rank) override {
        assert(dest_rank < getSize());
        if (getRank() != dest_rank) {
            postSend(asio_msg_type_t::REDUCE, local_value, sizeof(double));
            *global_value = -1.0;
        } else {
            recvWait(asio_msg_type_t::REDUCE, [&](const std::string &data) {
                if (data.size() == sizeof(double)) {
                    std::memcpy(global_value, data.data(), data.size());
                    return true;
                }
                return false;
            });
            *global_value += *local_value;
        }
        return 0;
    }

    int
    barrier(const std::string &barrier_id) override {
        const std::string barrier_name = barrier_id + "/" + std::to_string(++barrier_);
        postSend(asio_msg_type_t::BARRIER, barrier_name.data(), barrier_name.size());
        recvWait(asio_msg_type_t::BARRIER,
                 [&](const std::string &data) { return data == barrier_name; });
        return 0;
    }

private:
    void
    main() {
        try {
            connectSocket();
            while (!context_.stopped()) {
                context_.run();
            }
        }
        catch (...) {
            const std::unique_lock lock(mutex_);
            exception_ = std::current_exception();
            cond_.notify_all();
        }
    }

    void
    connectSocket() {
        const auto sock = std::make_shared<asio::ip::tcp::socket>(context_);

        auto handler = [this, sock](const asio::error_code &ec) {
            if (ec.value()) {
                throw std::runtime_error("ASIO Connection failed: " + ec.message());
            }
            socket_ = sock;
            timer_.cancel();
            recvHead();
            if (!outgoing_.empty()) {
                sendImpl();
            }
        };

        auto timeout = [this](const asio::error_code &ec) {
            if (!socket_) {
                throw std::runtime_error("ASIO Connection timeout: " + ec.message());
            }
        };

        if (acceptor_) {
            acceptor_->async_accept(*sock, handler);
        } else {
            sock->async_connect(endpoint_, handler);
        }
        timer_.async_wait(timeout);
    }

    [[nodiscard]] std::unique_ptr<asio::ip::tcp::acceptor>
    attemptAcceptor() noexcept {
        auto result = std::make_unique<asio::ip::tcp::acceptor>(context_);

        result->open(endpoint_.protocol());
        result->set_option(asio::ip::tcp::acceptor::reuse_address(true));

        try {
            result->bind(endpoint_);
        }
        catch (const std::exception &) {
            std::cout << "ASIO runtime bind() failed -- using connect() instead" << std::endl;
            return {}; // Use connect() instead of listen() and accept() on bind error.
        }

        result->listen();
        return result;
    }

    void
    postSend(const asio_msg_type_t type, const void *data, const std::size_t size) {
        const std::size_t head = size | (size_t(type) << 56);
        const std::string buffer =
            std::string(reinterpret_cast<const char *>(&head), sizeof(head)) +
            std::string(reinterpret_cast<const char *>(data), size);
        {
            const std::unique_lock lock(mutex_);

            if (exception_) {
                std::rethrow_exception(exception_);
            }
        }
        asio::post(context_, [this, buffer]() {
            const bool was_empty = outgoing_.empty();
            outgoing_.emplace_back(buffer);

            if (socket_ && was_empty) {
                sendImpl();
            }
        });
    }

    void
    sendImpl() {
        assert(!outgoing_.empty());

        asio::async_write(*socket_,
                          asio::buffer(outgoing_.front()),
                          [this](const asio::error_code &ec, const std::size_t done) {
                              if (ec.value()) {
                                  throw std::runtime_error("ASIO Write failed: " + ec.message());
                              }
                              outgoing_.pop_front();

                              if (!outgoing_.empty()) {
                                  sendImpl();
                              }
                          });
    }

    void
    recvHead() {
        asio::async_read(*socket_,
                         asio::buffer(&head_, sizeof(head_)),
                         [this](const asio::error_code &ec, const std::size_t done) {
                             if (ec.value()) {
                                 throw std::runtime_error("ASIO Read head failed: " + ec.message());
                             }
                             temp_.type = asio_msg_type_t(head_ >> 56);
                             temp_.data.clear();
                             temp_.data.resize(head_ & 0x1ffffffffull);
                             recvData();
                         });
    }

    void
    recvData() {
        asio::async_read(*socket_,
                         asio::buffer(temp_.data.data(), temp_.data.size()),
                         [this](const asio::error_code &ec, const std::size_t done) {
                             if (ec.value()) {
                                 throw std::runtime_error("ASIO Read data failed: " + ec.message());
                             }
                             // Lock scope
                             {
                                 const std::unique_lock lock(mutex_);
                                 incoming_.emplace_back(temp_);
                                 cond_.notify_one();
                             }
                             recvHead();
                         });
    }

    template<typename F>
    void
    recvWait(const asio_msg_type_t type, F &&f) {
        std::unique_lock lock(mutex_);
        if (exception_) {
            std::rethrow_exception(exception_);
        }

        // We don't care about scanning the whole list every time because
        // the list is expected to be very small and not checked frequently.

        if (!cond_.wait_for(lock, std::chrono::seconds(15), [&]() {
                for (auto it = incoming_.begin(); it != incoming_.end(); ++it) {
                    if ((it->type == type) && f(it->data)) {
                        incoming_.erase(it);
                        return true;
                    }
                }
                return false;
            })) {
            throw std::runtime_error("ASIO Receive timeout");
        }

        if (exception_) {
            std::rethrow_exception(exception_);
        }
    }

    const asio::ip::tcp::endpoint endpoint_;

    std::atomic<unsigned> barrier_{0};

    std::mutex mutex_;
    std::condition_variable cond_;
    std::list<xferBenchAsioIncoming> incoming_;
    std::exception_ptr exception_;

    size_t head_;
    xferBenchAsioIncoming temp_;
    std::list<std::string> outgoing_;
    asio::io_context context_;
    asio::steady_timer timer_; // For connection establishment
    const std::unique_ptr<asio::ip::tcp::acceptor> acceptor_; // Can be nullptr
    std::shared_ptr<asio::ip::tcp::socket> socket_;

    std::thread thread_;
};

#endif
