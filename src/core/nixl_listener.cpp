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

#include <fcntl.h>
#include <iostream>
#include "nixl.h"
#include "common/nixl_time.h"
#include "common/str_tools.h"
#include "agent_data.h"
#include "common/nixl_log.h"
#if HAVE_ETCD
#include <etcd/Client.hpp>
#endif // HAVE_ETCD

int connectToIP(std::string ip_addr, int port) {

    struct sockaddr_in listenerAddr;
    listenerAddr.sin_port   = htons(port);
    listenerAddr.sin_family = AF_INET;

    int ret_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (ret_fd == -1) {
        return -1;
    }

    if (inet_pton(AF_INET, ip_addr.c_str(),
                  &listenerAddr.sin_addr) <= 0) {
        close(ret_fd);
        return -1;
    }

    //make connect block for now to avoid ambiguity in send right after
    int orig_flags = fcntl(ret_fd, F_GETFL, 0);
    int new_flags = orig_flags ^ O_NONBLOCK;

    fcntl(ret_fd, F_SETFL, new_flags);

    if (connect(ret_fd, (struct sockaddr*)&listenerAddr,
                    sizeof(listenerAddr)) < 0) {
        perror("async connect");
        close(ret_fd);
        return -1;
    }

    //make nonblocking again
    fcntl(ret_fd, F_SETFL, orig_flags);

    return ret_fd;
}

size_t sendCommMessage(int fd, std::string msg){
    size_t bytes;
    bytes = send(fd, msg.c_str(), msg.size(), 0);
    if(bytes < 0) {
        std::cerr << "Cannot send on socket to fd " << fd << std::endl;
    }
    return bytes;
}

ssize_t recvCommMessage(int fd, std::string &msg){
    char buffer[16384];
    ssize_t one_recv_bytes = 0;
    ssize_t recv_bytes = 0;
    msg = std::string("");

    do {
        one_recv_bytes = recv(fd, buffer, sizeof(buffer), 0);
        if (one_recv_bytes == -1){
            if(errno == EAGAIN || errno == EWOULDBLOCK) return recv_bytes;
            std::cerr << "Cannot recv on socket fd " << fd << std::endl;
            return one_recv_bytes;
        }
        msg.append(buffer, one_recv_bytes);
        recv_bytes += one_recv_bytes;
    } while(one_recv_bytes > 0);

    return recv_bytes;
}

#if HAVE_ETCD
// Helper function to create etcd key
static std::string makeEtcdKey(const std::string& agent_name,
                                const std::string& namespace_prefix,
                                const std::string& metadata_type) {
    std::stringstream ss;
    ss << namespace_prefix << "/" << agent_name << "/" << metadata_type;
    return ss.str();
}

// Store metadata in etcd
static nixl_status_t storeMetadataInEtcd(const std::string& agent_name,
                                   const std::string& namespace_prefix,
                                   std::unique_ptr<etcd::Client>& client,
                                   const std::string& metadata_type,
                                   const nixl_blob_t& metadata) {
    // Check if etcd client is available
    if (!client) {
        std::cerr << "ETCD client not available" << std::endl;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        // Create key for metadata
        std::string metadata_key = makeEtcdKey(agent_name, namespace_prefix, metadata_type);

        // Store metadata in etcd
        etcd::Response response = client->put(metadata_key, metadata).get();

        if (response.is_ok()) {
            std::cout << "Successfully stored " << metadata_type << " in etcd with key: " << metadata_key << std::endl;
            return NIXL_SUCCESS;
        } else {
            std::cerr << "Failed to store " << metadata_type << " in etcd: " << response.error_message() << std::endl;
            return NIXL_ERR_BACKEND;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error sending " << metadata_type << " to etcd: " << e.what() << std::endl;
        return NIXL_ERR_BACKEND;
    }
}

// Fetch metadata from etcd
static nixl_status_t fetchMetadataFromEtcd(const std::string& agent_name,
                                     const std::string& namespace_prefix,
                                     std::unique_ptr<etcd::Client>& client,
                                     const std::string& metadata_type,
                                     nixl_blob_t& metadata) {
    // Check if etcd client is available
    if (!client) {
        std::cerr << "ETCD client not available" << std::endl;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        // Create key for agent's metadata
        std::string metadata_key = makeEtcdKey(agent_name, namespace_prefix, metadata_type);

        // Fetch metadata from etcd
        etcd::Response response = client->get(metadata_key).get();

        if (response.is_ok()) {
            metadata = response.value().as_string();
            std::cout << "Successfully fetched " << metadata_type << " for agent: " << agent_name << std::endl;
            return NIXL_SUCCESS;
        } else {
            std::cerr << "Failed to fetch " << metadata_type << " from etcd: " << response.error_message() << std::endl;
            return NIXL_ERR_BACKEND;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error fetching " << metadata_type << " from etcd: " << e.what() << std::endl;
        return NIXL_ERR_BACKEND;
    }
}

// Remove metadata from etcd
static nixl_status_t removeMetadataFromEtcd(const std::string& agent_name,
                                      const std::string& namespace_prefix,
                                      std::unique_ptr<etcd::Client>& client,
                                      const std::string& metadata_type) {
    // Check if etcd client is available
    if (!client) {
        std::cerr << "ETCD client not available" << std::endl;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        // Create key for metadata
        std::string metadata_key = makeEtcdKey(agent_name, namespace_prefix, metadata_type);

        // Remove metadata from etcd
        etcd::Response response = client->rm(metadata_key).get();

        if (response.is_ok()) {
            std::cout << "Successfully removed " << metadata_type << " from etcd for agent: " << agent_name << std::endl;
            return NIXL_SUCCESS;
        } else {
            std::cerr << "Warning: Failed to remove " << metadata_type << " from etcd: " << response.error_message() << std::endl;
            return NIXL_ERR_BACKEND;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error removing " << metadata_type << " from etcd: " << e.what() << std::endl;
        return NIXL_ERR_BACKEND;
    }
}

// Create etcd client with specified endpoints or from environment variable
static std::unique_ptr<etcd::Client> createEtcdClient() {
    try {
        // First check if NIXL_ETCD_ENDPOINTS environment variable is set
        char* env_endpoints = getenv("NIXL_ETCD_ENDPOINTS");
        if (env_endpoints && strlen(env_endpoints) > 0) {
            NIXL_INFO << "Using etcd endpoints from environment: " << env_endpoints;
        } else {
            // Fall back to provided endpoints if environment variable is not set
            NIXL_ERROR << "No etcd endpoints provided";
            return nullptr;
        }

        // Create and return new etcd client
        return std::make_unique<etcd::Client>(std::string(env_endpoints));
    } catch (const std::exception& e) {
        NIXL_ERROR << "Error creating etcd client: " << e.what();
        return nullptr;
    }
}
#endif // HAVE_ETCD

void nixlAgentData::commWorker(nixlAgent* myAgent){

    while(!(commThreadStop)) {

        std::vector<nixl_comm_req_t> work_queue;

        // first, accept new connections
        int new_fd = 0;

#if HAVE_ETCD
        auto etcdclient = std::unique_ptr<etcd::Client>(nullptr);
        std::string namespace_prefix = getenv("NIXL_ETCD_NAMESPACE");
        std::string etcd_endpoints = getenv("NIXL_ETCD_ENDPOINTS");

        while(new_fd != -1 && !etcd_endpoints.empty()) {
#else
        while(new_fd != -1) {
#endif // HAVE_ETCD
            new_fd = listener->acceptClient();
            nixl_socket_peer_t accepted_client;

            if(new_fd != -1){
                // need to convert fd to IP address and add to client map
                sockaddr_in client_address;
                socklen_t client_addrlen = sizeof(client_address);
                if (getpeername(new_fd, (sockaddr*)&client_address, &client_addrlen) == 0) {
                    char client_ip[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &client_address.sin_addr, client_ip, INET_ADDRSTRLEN);
                    accepted_client.first = std::string(client_ip);
                    accepted_client.second = client_address.sin_port;
                } else {
                    throw std::runtime_error("getpeername failed");
                }
                remoteSockets[accepted_client] = new_fd;

                // make new socket nonblocking
                int new_flags = fcntl(new_fd, F_GETFL, 0) | O_NONBLOCK;

                if (fcntl(new_fd, F_SETFL, new_flags) == -1)
                    throw std::runtime_error("fcntl accept");

            }
        }

#if HAVE_ETCD
        if(!etcdclient && useEtcd) {
            etcdclient = createEtcdClient();
            NIXL_INFO << "Created etcd client to " << etcd_endpoints;
        }
#endif // HAVE_ETCD

        // second, do agent commands
        getCommWork(work_queue);

        for(nixl_comm_req_t request: work_queue) {

            nixl_comm_t req_command = std::get<0>(request);
            std::string req_ip = std::get<1>(request);
            int req_port = std::get<2>(request);
            std::string my_MD = std::get<3>(request);

            nixl_socket_peer_t req_sock = std::make_pair(req_ip, req_port);

            // use remote IP for socket lookup
            auto client = remoteSockets.find(req_sock);
            int client_fd;

            switch(req_command) {
                case SOCK_SEND:
                {
                    // not connected
                    if(client == remoteSockets.end()) {
                        int new_client = connectToIP(req_ip, req_port);
                        if(new_client == -1) {
                            std::cerr << "Listener thread could not connect to IP " << req_ip << " and port " << req_port << std::endl;
                            break;
                        }
                        remoteSockets[req_sock] = new_client;
                        client_fd = new_client;
                    } else {
                        client_fd = client->second;
                    }

                    sendCommMessage(client_fd, std::string("NIXLCOMM:LOAD" + my_MD));
                    break;
                }
                case SOCK_FETCH:
                {
                    if(client == remoteSockets.end()) {
                        int new_client = connectToIP(req_ip, req_port);
                        if(new_client == -1) {
                            std::cerr << "Listener thread could not connect to IP " << req_ip;
                            break;
                        }
                        remoteSockets[req_sock] = new_client;
                        client_fd = new_client;
                    } else
                        client_fd = client->second;

                    sendCommMessage(client_fd, std::string("NIXLCOMM:SEND"));
                    break;
                }
                case SOCK_INVAL:
                {
                    if(client == remoteSockets.end()) {
                        // improper usage
                        throw std::runtime_error("invalidate on closed socket\n");
                    }
                    client_fd = client->second;
                    sendCommMessage(client_fd, std::string("NIXLCOMM:INVL") + name);
                    break;
                }
#if HAVE_ETCD
                // ETCD operations using existing methods
                case ETCD_SEND:
                {
                    if (!useEtcd) {
                        throw std::runtime_error("ETCD is not enabled");
                    }

                    // Parse request parameters
                    std::string metadata_type = "metadata";

                    if (req_ip.find("/partial_metadata") != std::string::npos) {
                        metadata_type = "partial_metadata";
                    }

                    // Use local storeMetadataInEtcd function
                    nixl_status_t ret = storeMetadataInEtcd(name, namespace_prefix, etcdclient, metadata_type, my_MD);
                    if (ret != NIXL_SUCCESS) {
                        std::cerr << "Failed to store metadata in etcd: " << ret << std::endl;
                    }
                    break;
                }
                case ETCD_FETCH:
                {
                    if (!useEtcd) {
                        throw std::runtime_error("ETCD is not enabled");
                    }

                    // First try a direct get
                    nixl_blob_t remote_metadata;
                    nixl_status_t ret = fetchMetadataFromEtcd(req_ip, namespace_prefix, etcdclient, "metadata", remote_metadata);

                    if (ret == NIXL_SUCCESS) {
                        // Load the metadata
                        std::string remote_agent;
                        ret = myAgent->loadRemoteMD(remote_metadata, remote_agent);
                        if (ret == NIXL_SUCCESS) {
                            std::cout << "Successfully loaded metadata for agent: " << req_ip << std::endl;
                        } else {
                            std::cerr << "Failed to load remote metadata: " << ret << std::endl;
                        }
                    } else {
                        // Key not found, set up a watch
                        std::cout << "Metadata not found, setting up watch for agent: " << req_ip << std::endl;

                        try {
                            // Create key for agent's metadata
                            std::string metadata_key = makeEtcdKey(req_ip, namespace_prefix, "metadata");

                            // Get current index to watch from
                            etcd::Response response = etcdclient->get(metadata_key).get();
                            int64_t watch_index = response.index();
                            // Set up watch
                            etcd::Response watch_response = etcdclient->watch(metadata_key, watch_index).get();

                            if (watch_response.is_ok()) {
                                std::string remote_md = watch_response.value().as_string();
                                std::string remote_agent;
                                ret = myAgent->loadRemoteMD(remote_md, remote_agent);
                                if (ret != NIXL_SUCCESS) {
                                    std::cerr << "Failed to load remote metadata from watch: " << ret << std::endl;
                                } else {
                                    std::cout << "Successfully loaded metadata from watch for agent: " << req_ip << std::endl;
                                }
                            } else {
                                std::cerr << "Watch failed: " << watch_response.error_message() << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "Error watching etcd: " << e.what() << std::endl;
                        }
                    }
                    break;
                }
                case ETCD_INVAL:
                {
                    if (!useEtcd) {
                        throw std::runtime_error("ETCD is not enabled");
                    }

                    // The agent name comes in req_ip
                    try {
                        std::string agent = req_ip;

                        // Remove main metadata
                        nixl_status_t status1 = removeMetadataFromEtcd(agent, namespace_prefix, etcdclient, "metadata");
                        if (status1 != NIXL_SUCCESS) {
                            std::cerr << "Warning: Failed to remove metadata from etcd for agent: " << agent << std::endl;
                        } else {
                            std::cout << "Successfully removed metadata from etcd for agent: " << agent << std::endl;
                        }

                        // Remove partial metadata
                        nixl_status_t status2 = removeMetadataFromEtcd(agent, namespace_prefix, etcdclient, "partial_metadata");
                        if (status2 != NIXL_SUCCESS) {
                            std::cerr << "Warning: Failed to remove partial metadata from etcd for agent: " << agent << std::endl;
                        } else {
                            std::cout << "Successfully removed partial metadata from etcd for agent: " << agent << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error removing metadata from etcd: " << e.what() << std::endl;
                    }
                    break;
                }
#endif // HAVE_ETCD
                default:
                {
                    throw std::runtime_error("Impossible command\n");
                    break;
                }
            }
        }

        // third, do remote commands
        auto socket_iter = remoteSockets.begin();
        while (socket_iter != remoteSockets.end()) {
            std::string commands;
            std::vector<std::string> command_list;
            nixl_status_t ret;

            ssize_t recv_bytes = recvCommMessage(socket_iter->second, commands);

            if(recv_bytes == 0 || recv_bytes == -1) {
                socket_iter++;
                continue;
            }

            command_list = str_split_substr(commands, "NIXLCOMM:");

            for(std::string command : command_list) {

                if(command.size() < 4) continue;

                // always just 4 chars:
                std::string header = command.substr(0, 4);

                if(header == "LOAD") {
                    std::string remote_md = command.substr(4);
                    std::string remote_agent;
                    ret = myAgent->loadRemoteMD(remote_md, remote_agent);
                    if(ret != NIXL_SUCCESS) {
                        throw std::runtime_error("loadRemoteMD in listener thread failed, critically failing\n");
                    }
                    // not sure what to do with remote_agent
                } else if(header == "SEND") {
                    nixl_blob_t my_MD;
                    myAgent->getLocalMD(my_MD);

                    sendCommMessage(socket_iter->second, std::string("NIXLCOMM:LOAD" + my_MD));
                } else if(header == "INVL") {
                    std::string remote_agent = command.substr(4);
                    myAgent->invalidateRemoteMD(remote_agent);
                    break;
                } else {
                    throw std::runtime_error("Received socket message with bad header" + header + ", critically failing\n");
                }
            }

            socket_iter++;
        }

        nixlTime::us_t start = nixlTime::getUs();
        while( (start + config.lthrDelay) > nixlTime::getUs()) {
            std::this_thread::yield();
        }
    }

    // Close remaining connections
    for (auto &[remote, fd] : remoteSockets) {
        shutdown(fd, SHUT_RDWR);
        close(fd);
    }
}

void nixlAgentData::enqueueCommWork(std::tuple<nixl_comm_t, std::string, int, std::string> request){
    std::lock_guard<std::mutex> lock(commLock);
    commQueue.push_back(request);
}


void nixlAgentData::getCommWork(std::vector<nixl_comm_req_t> &req_list){
    std::lock_guard<std::mutex> lock(commLock);
    req_list = commQueue;
    commQueue.clear();
}
