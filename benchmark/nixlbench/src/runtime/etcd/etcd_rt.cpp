/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <thread>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <etcd/Client.hpp>
#include <etcd/Response.hpp>
#include "etcd_rt.h"

#define ETCD_EP_DEFAULT "http://localhost:2379"

// ETCD Runtime implementation
xferBenchEtcdRT::xferBenchEtcdRT(const std::string& etcd_endpoints, const int size,
                                 int *terminate_input) {

    std::string use_etcd_ep = ETCD_EP_DEFAULT;

    // Parse command line arguments to get ETCD endpoints
    if (!etcd_endpoints.empty()) {
        use_etcd_ep = etcd_endpoints;
    }

    namespace_prefix = "xferbench/"; // Namespace for XFER benchmark
    barrier_gen = 0;
    terminate = terminate_input;

    // Connect to ETCD
    try {
        std::cout << "Connecting to ETCD at " << use_etcd_ep << std::endl;
        client = std::make_unique<etcd::Client>(use_etcd_ep);
    } catch (const std::exception& e) {
        std::cerr << "Failed to connect to ETCD: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    // Registration process - get a unique rank
    std::string lock_key = makeKey("lock");

    // Try to acquire a lock for registration
    auto lock_response = client->lock(lock_key).get();
    if (lock_response.error_code() != 0) {
        std::cerr << "Failed to acquire lock: " << lock_response.error_message() << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the current size - number of processes that have registered
    auto size_response = client->get(makeKey("size")).get();
    if (size_response.error_code() == 0) {
        my_rank = std::stoi(size_response.value().as_string());
    } else {
        my_rank = 0;
    }

    // Set the global size to the sum of initiator and target devices
    global_size = size;

    // Update registration information
    client->put(makeKey("size"), std::to_string(my_rank + 1)).get();
    client->put(makeKey("rank", my_rank), "active").get();

    // Release the lock
    client->unlock(lock_response.lock_key()).get();

    // Set the parent class rank and size
    setRank(my_rank);
    setSize(global_size);

    if (my_rank >= global_size) {
        std::cerr << "Rank " << my_rank
                  << " is greater than or equal to global size "
                  << global_size << std::endl;
        client->rmdir(makeKey(""), true).get();
        exit(EXIT_FAILURE);
    }
    std::cout << "ETCD Runtime: Registered as rank " << my_rank
	      << " item " << my_rank + 1 << " of "
	      << global_size << std::endl;
}

xferBenchEtcdRT::~xferBenchEtcdRT() {
    // All ranks delete, as some could be missing if ETCD state is confused
    client->rmdir(makeKey(""), true).get();
}

int xferBenchEtcdRT::getRank() const {
    return my_rank;
}

int xferBenchEtcdRT::getSize() const {
    return global_size;
}

std::string xferBenchEtcdRT::makeTypedKey(const std::string& operation, int src,
                                          int dst, xferBenchEtcdMsgType type) {
    std::stringstream ss;
    ss << namespace_prefix << operation << "+"
       << (type == XFER_BENCH_ETCD_MSG_TYPE_INT ? "int_data" : "char_data") << "/"
       << "src=" << src << "/"
       << "dst=" << dst;
    return ss.str();
}

int xferBenchEtcdRT::sendInt(int* buffer, int dest_rank) {
    try {
        // Create the message key
        std::string msg_key = makeTypedKey("msg", my_rank, dest_rank, XFER_BENCH_ETCD_MSG_TYPE_INT);
        std::string ack_key = msg_key + "/ack";

        // Store the integer value directly as a string
        std::string value_str = std::to_string(*buffer);
        client->put(msg_key, value_str).get();

        int retries = 0;
        bool ack_received = false;

        while (!ack_received && should_retry(retries)) {
            auto ack_response = client->get(ack_key).get();
            if (ack_response.error_code() == 0 && ack_response.value().as_string() == "received") {
                ack_received = true;
                // Clean up the acknowledgment
                client->rm(ack_key).get();
            } else {
                // Wait and retry
                std::this_thread::sleep_for(std::chrono::seconds(1));
                retries++;
            }
        }

        if (!ack_received) {
            std::cerr << "Timeout waiting for int data acknowledgment from rank " << dest_rank << std::endl;
            return -1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error sending int data: " << e.what() << std::endl;
        return -1;
    }
}

int xferBenchEtcdRT::recvInt(int* buffer, int src_rank) {
    // Create the message key
    std::string msg_key = makeTypedKey("msg", src_rank, my_rank, XFER_BENCH_ETCD_MSG_TYPE_INT);
    std::string ack_key = msg_key + "/ack";

    // Poll until the data is available (blocking)
    int retries = 0;
    bool data_received = false;

    while (!data_received && should_retry(retries)) {
        auto response = client->get(msg_key).get();
        if (response.error_code() == 0) {
            // Get the value directly as a string
            std::string value_str = response.value().as_string();

            try {
                // Convert string to integer
                *buffer = std::stoi(value_str);

                // Send acknowledgment
                client->put(ack_key, "received").get();
                data_received = true;

                // Delete the message after we've acknowledged it
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Brief delay
                client->rm(msg_key).get();

                return 0;
            } catch (const std::exception& e) {
                std::cerr << "Error converting value to integer: " << e.what() << std::endl;
                return -1;
            }
        } else {
            // Wait and retry
            std::this_thread::sleep_for(std::chrono::seconds(1));
            retries++;
        }
    }

    if (!data_received) {
        std::cerr << "Timeout waiting for int data from rank " << src_rank << std::endl;
        return -1;
    }

    return 0;
}


int xferBenchEtcdRT::sendChar(char* buffer, size_t count, int dest_rank) {
    try {
        // Create the message key and data key
        std::string msg_key = makeTypedKey("msg", my_rank, dest_rank, XFER_BENCH_ETCD_MSG_TYPE_CHAR);
        std::string data_key = msg_key + "/data";
        std::string ack_key = msg_key + "/ack";

        // Store raw character data directly
        std::string data_str(buffer, buffer + count);
        client->put(data_key, data_str).get();

        // Store metadata
        std::string meta = std::to_string(my_rank) + ":" + std::to_string(dest_rank) + ":" + std::to_string(count);
        client->put(msg_key, meta).get();

        int retries = 0;
        bool ack_received = false;

        while (!ack_received && should_retry(retries)) {
            auto ack_response = client->get(ack_key).get();
            if (ack_response.error_code() == 0 && ack_response.value().as_string() == "received") {
                ack_received = true;
                // Clean up the acknowledgment
                client->rm(ack_key).get();
            } else {
                // Wait and retry
                std::this_thread::sleep_for(std::chrono::seconds(1));
                retries++;
            }
        }

        if (!ack_received) {
            std::cerr << "Timeout waiting for char data acknowledgment from rank " << dest_rank << std::endl;
            return -1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error sending char data: " << e.what() << std::endl;
        return -1;
    }
}

int xferBenchEtcdRT::recvChar(char* buffer, size_t count, int src_rank) {
    // Create the message keys
    std::string msg_key = makeTypedKey("msg", src_rank, my_rank, XFER_BENCH_ETCD_MSG_TYPE_CHAR);
    std::string data_key = msg_key + "/data";
    std::string ack_key = msg_key + "/ack";

    // Poll until the data is available (blocking)
    int retries = 0;
    bool data_received = false;

    while (!data_received && should_retry(retries)) {
        // First check if metadata exists
        auto meta_response = client->get(msg_key).get();
        if (meta_response.error_code() == 0) {
            // Now get the actual data
            auto data_response = client->get(data_key).get();
            if (data_response.error_code() == 0) {
                // Get the raw character data
                std::string data_str = data_response.value().as_string();

                // Copy to the buffer
                size_t copy_size = std::min(data_str.size(), count);
                std::copy(data_str.begin(), data_str.begin() + copy_size, buffer);

                // Send acknowledgment
                client->put(ack_key, "received").get();
                data_received = true;

                // Delete the message after we've acknowledged it
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Brief delay
                client->rm(data_key).get();
                client->rm(msg_key).get();

                return 0;
            }
        }

        // Wait and retry
        std::this_thread::sleep_for(std::chrono::seconds(1));
        retries++;
    }

    if (!data_received) {
        std::cerr << "Timeout waiting for char data from rank " << src_rank << std::endl;
        return -1;
    }

    return 0;
}

int xferBenchEtcdRT::reduceSumDouble(double *local_value, double *global_value, int dest_rank) {
    try {
        // Use a random ID for this reduction operation
        std::string reduce_id = std::to_string(std::time(nullptr)) + "-" + std::to_string(std::rand());
        std::string reduce_key = makeKey("reduce/" + reduce_id);
        std::string value_key = reduce_key + "/rank-" + std::to_string(my_rank);

        // Contribute our value directly as a string
        std::stringstream ss;
        ss << std::fixed << std::setprecision(16) << *local_value;
        client->put(value_key, ss.str()).get();

        // If we are the destination rank, collect and reduce
        if (my_rank == dest_rank) {
            // Initialize the global value with our value
            *global_value = *local_value;

            // Wait for all contributions
            int received = 0;
            int expected = global_size - 1; // Excluding ourselves
            int retries = 0;

            while (received < expected && should_retry(retries, 30)) {
                auto response = client->ls(reduce_key).get();
                if (response.error_code() == 0) {
                    for (const auto& kv : response.keys()) {
                        // Skip our own contribution
                        std::string key = kv;
                        if (key == value_key) {
                            continue;
                        }

                        // Get the contribution data as a string
                        auto get_response = client->get(key).get();
                        if (get_response.error_code() == 0) {
                            // Convert string directly to double
                            std::string value_str = get_response.value().as_string();
                            double contrib_value = std::stod(value_str);

                            // Add to the global value
                            *global_value += contrib_value;

                            // Remove this contribution
                            client->rm(key).get();
                            received++;
                        }
                    }
                }

                if (received < expected) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    retries++;
                }
            }

            // Clean up
            client->rmdir(reduce_key, true).get();

            if (received < expected) {
                std::cerr << "Timeout waiting for reduction contributions (got " << received
                          << "/" << expected << " contributions)" << std::endl;
                return -1;
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in reduce operation: " << e.what() << std::endl;
        return -1;
    }
}

int xferBenchEtcdRT::barrier(const std::string& barrier_id) {
    int count = 0;
    std::string barrier_suffix = "barrier/" + barrier_id + "/" + std::to_string(barrier_gen);
    std::string barrier_key = namespace_prefix + barrier_suffix + "/";
    barrier_gen++; // In case same barrier is reused too quickly

    try {
        auto resp = client->put(barrier_key + std::to_string(my_rank), "1").get();
        if (!resp.is_ok()) {
            throw std::runtime_error("put");
        }

        for (auto retries = 0; should_retry(retries); ++retries) {
            auto resp = client->ls(barrier_key).get();
            if (resp.error_code()) {
                break;
            }

            count = resp.keys().size();
            if (count == 0 || // rank 0 observed completion and deleted the barrier
                count == (int)global_size) {
                if (my_rank == 0) {
                    // Only one of them needs to cleanup
                    client->rmdir(barrier_key, true).get();
                }
                return 0;
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        throw std::runtime_error("wait");
    } catch (const std::exception& e) {
        std::cerr << "Error in barrier " << e.what() << " " << barrier_key
            << " rank " << my_rank << " completed "
            << count << "/" << global_size << " ranks)" << std::endl;
    }

    return -1;
}

int xferBenchEtcdRT::broadcastInt(int* buffer, size_t count, int root_rank) {
    try {
        // Create a unique key for this broadcast operation
        std::string bcast_key  = makeKey("bcast/int", root_rank);
        std::string barrier_id = "bcast_int_" + std::to_string(root_rank);

        // First phase: root process puts the value in etcd
        if (my_rank == root_rank) {
            // Serialize array of integers using binary representation
            std::string value_str(reinterpret_cast<char*>(buffer), count * sizeof(int));
            client->put(bcast_key, value_str).get();
        }

        // Synchronize to ensure the value is written before non-root processes try to read it
        barrier(barrier_id + "_write");

        // Second phase: non-root processes read the value
        if (my_rank != root_rank) {
            int retries = 0;
            bool data_received = false;

            while (!data_received && should_retry(retries, 10)) {
                auto response = client->get(bcast_key).get();
                if (response.error_code() == 0) {
                    std::string value_str = response.value().as_string();
                    try {
                        // Deserialize binary data directly into the buffer
                        if (value_str.size() >= count * sizeof(int)) {
                            std::memcpy(buffer, value_str.data(), count * sizeof(int));
                            data_received = true;
                        } else {
                            std::cerr << "Received data size (" << value_str.size()
                                      << ") is smaller than expected (" << count * sizeof(int) << ")" << std::endl;
                            retries++;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error deserializing broadcast data: " << e.what() << std::endl;
                        return -1;
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    retries++;
                }
            }

            if (!data_received) {
                std::cerr << "Failed to read broadcast data from rank "
                          << root_rank << std::endl;
                return -1;
            }
        }

        // Synchronize to ensure all processes have read the value before cleaning up
        barrier(barrier_id + "_read");

        // Clean up
        if (my_rank == root_rank) {
            client->rm(bcast_key).get();
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in broadcast operation: " << e.what() << std::endl;
        return -1;
    }
}
