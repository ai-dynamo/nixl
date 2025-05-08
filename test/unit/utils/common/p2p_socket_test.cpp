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
#include <algorithm>
#include <iostream>
#include <string>

#include "stream/metadata_stream.h"

void run_server()
{
    nixlMDStreamListener listener(8082);
    listener.startListenerForClient();
    std::cout << listener.recvFromClient();
    return;
}

void run_client()
{
    nixlMDStreamClient client("127.0.0.1", 8082);
    client.connectListener();
    std::string data = "Hello NixL MD listener\n";
    client.sendData(data);
    std::cout << "Sent Data to listener " << data << "\n";
    std::string ack = client.recvData();
    std::cout << "Received from server: " << ack << "\n";

    return;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Enter client/server\n";
        exit(-1);
    }

    std::string arg1 = argv[1];
    std::string server = "server";
    std::transform(arg1.begin(), arg1.end(), arg1.begin(), ::tolower);

    if (arg1 == server)
        run_server();
    else
        run_client();

    return 0;
}
