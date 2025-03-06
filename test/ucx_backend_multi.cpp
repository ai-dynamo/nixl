#include <iostream>
#include <cassert>
#include <thread>

#include "backend_tester.h"
#include "ucx_backend.h"

// Temporarily while fixing CI/CD pipeline
#define USE_PTHREAD false

volatile bool ready[2]  = {false, false};
volatile bool done[2]  = {false, false};
volatile bool disconnect[2]  = {false, false};
std::string conn_info[2];

void test_thread(int id)
{
    nixlUcxInitParams init_params;
    nixlBackendEngine* ucxw;
    nixlBackendTester* ucx_tester;
    nixl_status_t ret;

    std::string my_name("Agent1");
    std::string other("Agent2");

    if(id){
        my_name = "Agent2";
        other = "Agent1";
    }

    init_params.localAgent = my_name;
    init_params.enableProgTh = USE_PTHREAD;

    std::cout << my_name << " Started\n";

    ucxw = (nixlBackendEngine*) new nixlUcxEngine (&init_params);
    ucx_tester = new nixlBackendTester(ucxw);

    if(!USE_PTHREAD) ucx_tester->progress();

    conn_info[id] = ucx_tester->getConnInfo();

    ready[id] = true;
    //wait for other
    while(!ready[!id]);

    ret = ucx_tester->loadRemoteConnInfo(other, conn_info[!id]);
    assert(ret == NIXL_SUCCESS);

    //one-sided connect
    if(!id)
        ret = ucx_tester->connect(other);

    assert(ret == NIXL_SUCCESS);

    done[id] = true;
    while(!done[!id])
        if(!USE_PTHREAD && id) ucx_tester->progress();

    std::cout << "Thread passed with id " << id << "\n";

    //test one-sided disconnect
    if(!id)
        ucx_tester->disconnect(other);

    disconnect[id] = true;
    //wait for other
    while(!disconnect[!id]);

    if(!USE_PTHREAD) ucx_tester->progress();

    std::cout << "Thread disconnected with id " << id << "\n";

    delete ucx_tester;
}

int main()
{
    std::cout << "Multithread test start \n";
    std::thread th1(test_thread, 0);
    std::thread th2(test_thread, 1);

    th1.join();
    th2.join();
}
