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

void nixlAgent::commWorker(){

    while(!(data->commThreadStop)) {

        std::vector<nixl_comm_req_t> work_queue;

        //first, accept new connections
        int new_fd = 0;

        while(new_fd != -1) {
            new_fd = data->listener->acceptClient();
            std::string accepted_client;

            if(new_fd != -1){
                //need to convert fd to IP address and add to client map
                sockaddr_in client_address;
                socklen_t client_addrlen = sizeof(client_address);
                if (getpeername(new_fd, (sockaddr*)&client_address, &client_addrlen) == 0) {
                    char client_ip[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &client_address.sin_addr, client_ip, INET_ADDRSTRLEN);
                    accepted_client = std::string(client_ip);
                } else {
                    throw std::runtime_error("getpeername failed");
                }
                data->remoteSockets[accepted_client] = new_fd;

                //make new socket nonblocking
                int new_flags = fcntl(new_fd, F_GETFL, 0) | O_NONBLOCK;

                if (fcntl(new_fd, F_SETFL, new_flags) == -1)
                    throw std::runtime_error("fcntl accept");

            }
        }

        //second, do agent commands
        data->getCommWork(work_queue);

        for(nixl_comm_req_t request: work_queue) {

            nixl_comm_t req_command = std::get<0>(request);
            std::string req_ip = std::get<1>(request);
            int req_port = std::get<2>(request);

            //use remote IP for socket lookup
            auto client = data->remoteSockets.find(req_ip);
            int client_fd;

            switch(req_command) {
                case SOCK_SEND:
                {
                    //not connected
                    if(client == data->remoteSockets.end()) {
                        int new_client = connectToIP(req_ip, req_port);
                        if(new_client == -1) {
                            std::cerr << "Listener thread could not connect to IP " << req_ip;
                            break;
                        }
                        data->remoteSockets[req_ip] = new_client;
                        client_fd = new_client;
                    } else {
                        client_fd = client->second;
                    }

                    nixl_blob_t my_MD;
                    getLocalMD(my_MD);

                    sendCommMessage(client_fd, std::string("NIXLCOMM:SENT" + my_MD));
                    break;
                }
                case SOCK_FETCH:
                {
                    if(client == data->remoteSockets.end()) {
                        int new_client = connectToIP(req_ip, req_port);
                        if(new_client == -1) {
                            std::cerr << "Listener thread could not connect to IP " << req_ip;
                            break;
                        }
                        data->remoteSockets[req_ip] = new_client;
                        client_fd = new_client;
                    } else
                        client_fd = client->second;

                    sendCommMessage(client_fd, std::string("NIXLCOMM:SEND"));
                    break;
                }
                case SOCK_INVAL:
                {
                    if(client == data->remoteSockets.end()) {
                        //improper usage
                        throw std::runtime_error("invalidate on closed socket\n");
                    }
                    client_fd = client->second;
                    sendCommMessage(client_fd, std::string("NIXLCOMM:INVL"));
                    close(client_fd);
                    data->remoteSockets.erase(req_ip);
                    break;
                }
                default:
                {
                    throw std::runtime_error("Impossible command\n");
                    break;
                }
            }
        }

        //third, do remote commands
        for (const auto& [ip_address, socketClient] : data->remoteSockets ) {
            std::string commands;
            std::vector<std::string> command_list;
            nixl_status_t ret;

            ssize_t recv_bytes = recvCommMessage(socketClient, commands);

            if(recv_bytes == 0 || recv_bytes == -1) continue;

            command_list = str_split_substr(commands, "NIXLCOMM:");

            for(std::string command : command_list) {

                if(command.size() < 4) continue;

                //always just 4 chars:
                std::string header = command.substr(0, 4);

                if(header == "SENT") {
                    std::string remote_md = command.substr(4);
                    std::string remote_agent;
                    ret = loadRemoteMD(remote_md, remote_agent);
                    if(ret != NIXL_SUCCESS) {
                        throw std::runtime_error("loadRemoteMD in listener thread failed, critically failing\n");
                    }
                    //not sure what to do with remote_agent
                } else if(header == "SEND") {
                    nixl_blob_t my_MD;
                    getLocalMD(my_MD);

                    sendCommMessage(socketClient, std::string("NIXLCOMM:SENT" + my_MD));
                } else if(header == "INVL") {
                    close(socketClient);
                    data->remoteSockets.erase(ip_address);
                } else {
                    throw std::runtime_error("Received socket message with bad header" + header + ", critically failing\n");
                }
            }
        }

        //relatively large delay for now
        nixlTime::us_t start = nixlTime::getUs();
        while( (start + data->config.lthrDelay) > nixlTime::getUs()) {
            std::this_thread::yield();
        }
    }
}

void nixlAgentData::enqueueCommWork(std::tuple<nixl_comm_t, std::string, int> request){
    std::lock_guard<std::mutex> lock(commLock);
    commQueue.push_back(request);
}


void nixlAgentData::getCommWork(std::vector<nixl_comm_req_t> &req_list){
    std::lock_guard<std::mutex> lock(commLock);
    req_list = commQueue;
    commQueue.clear();
}
