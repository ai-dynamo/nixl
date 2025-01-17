#include "internal/metadata_stream.h"
#include <iostream>
#include <string>
#include <algorithm>

void run_server() {
    nixlMDStreamListener listener(8082);
    listener.startListenerForClients();
    return;
}

void run_client() {
    nixlMDStreamClient client("127.0.0.1", 8082);
    client.connectListener();
    std::string data = "Hello NixL MD listener\n";
    client.sendData(data);
    std::cout << "Sent Data to listener " << data << "\n";
    std::string ack = client.recvData();
    std::cout << "Received from server: " << ack << "\n";

    return;
}

int main (int argc, char *argv[]) {
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
