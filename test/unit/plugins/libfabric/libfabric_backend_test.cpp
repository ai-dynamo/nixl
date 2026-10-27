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

// Regression test: a peer that still holds a connection entry must answer a
// re-establishing side's handshake (vllm#62144).

#include <chrono>
#include <iostream>
#include <string>

#include "libfabric_backend.h"
#include "test_utils.h"

using namespace std;
using namespace std::chrono;

// Well under the 60 s handshake deadline
static const seconds kMaxHandshakeWait(45);

static nixlLibfabricEngine *
createEngine(const string &name) {
    nixlBackendInitParams init;
    nixl_b_params_t custom_params;

    init.enableProgTh = true;
    init.pthrDelay = 100;
    init.localAgent = name;
    init.customParams = &custom_params;
    init.type = "LIBFABRIC";

    auto *engine = new nixlLibfabricEngine(&init);
    nixl_exit_on_failure(!engine->getInitErr(), "Failed to initialize engine " + name);
    return engine;
}

int
main() {
    auto *a = createEngine("AgentA");
    auto *b = createEngine("AgentB");

    string conn_a, conn_b;
    nixl_exit_on_failure(a->getConnInfo(conn_a) == NIXL_SUCCESS, "getConnInfo(A) failed");
    nixl_exit_on_failure(b->getConnInfo(conn_b) == NIXL_SUCCESS, "getConnInfo(B) failed");
    nixl_exit_on_failure(a->loadRemoteConnInfo("AgentB", conn_b) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(A, B) failed");
    nixl_exit_on_failure(b->loadRemoteConnInfo("AgentA", conn_a) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(B, A) failed");

    auto t0 = steady_clock::now();
    nixl_status_t status = a->connect("AgentB");
    nixl_exit_on_failure(status == NIXL_SUCCESS,
                         "Initial connect A->B failed (the peer never answered the handshake)");
    auto initial_ms = duration_cast<milliseconds>(steady_clock::now() - t0).count();
    nixl_exit_on_failure(duration_cast<seconds>(steady_clock::now() - t0) < kMaxHandshakeWait,
                         "Initial handshake took close to the 60 s deadline");

    t0 = steady_clock::now();
    status = b->connect("AgentA");
    nixl_exit_on_failure(status == NIXL_SUCCESS, "Initial connect B->A failed");

    // One-sided teardown: B is never told and keeps its entry for A
    status = a->disconnect("AgentB");
    nixl_exit_on_failure(status == NIXL_SUCCESS, "disconnect(A -> B) failed");

    // A re-establishes: fresh connection, so its handshake again carries conn info
    status = a->loadRemoteConnInfo("AgentB", conn_b);
    nixl_exit_on_failure(status == NIXL_SUCCESS, "loadRemoteConnInfo(A, B) after teardown failed");
    t0 = steady_clock::now();
    status = a->connect("AgentB");
    auto elapsed = duration_cast<milliseconds>(steady_clock::now() - t0).count();
    nixl_exit_on_failure(status == NIXL_SUCCESS,
                         "Re-establishment after unilateral teardown failed: the peer still "
                         "holds a live entry and did not answer the fresh handshake "
                         "(vllm#62144 deadlock)");
    nixl_exit_on_failure(milliseconds(elapsed) < kMaxHandshakeWait,
                         "Re-establishment degraded to near the 60 s handshake timeout");

    cout << "libfabric_backend_test OK: initial handshake " << initial_ms
         << " ms, re-establishment " << elapsed << " ms" << endl;

    // A holder that registered a peer but never called connect() still owes the
    // reply: its connection is DISCONNECTED by default until connect(), and only
    // a removed registration (an actual disconnect) means no reply.
    auto *c = createEngine("AgentC");
    auto *d = createEngine("AgentD");
    string conn_c, conn_d;
    nixl_exit_on_failure(c->getConnInfo(conn_c) == NIXL_SUCCESS, "getConnInfo(C) failed");
    nixl_exit_on_failure(d->getConnInfo(conn_d) == NIXL_SUCCESS, "getConnInfo(D) failed");
    nixl_exit_on_failure(c->loadRemoteConnInfo("AgentD", conn_d) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(C, D) failed");
    nixl_exit_on_failure(d->loadRemoteConnInfo("AgentC", conn_c) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(D, C) failed");
    nixl_exit_on_failure(c->connect("AgentD") == NIXL_SUCCESS,
                         "Initial connect C->D failed (D's handshake should arrive without D "
                         "ever connecting)");
    status = c->disconnect("AgentD");
    nixl_exit_on_failure(status == NIXL_SUCCESS, "disconnect(C -> D) failed");
    status = c->loadRemoteConnInfo("AgentD", conn_d);
    nixl_exit_on_failure(status == NIXL_SUCCESS, "loadRemoteConnInfo(C, D) after teardown failed");
    t0 = steady_clock::now();
    status = c->connect("AgentD");
    elapsed = duration_cast<milliseconds>(steady_clock::now() - t0).count();
    nixl_exit_on_failure(status == NIXL_SUCCESS,
                         "Re-establishment from a registered-but-never-connected peer failed: "
                         "the holder's reply was dropped");
    nixl_exit_on_failure(milliseconds(elapsed) < kMaxHandshakeWait,
                         "Re-establishment from an unconnected holder hit the handshake deadline");
    cout << "libfabric_backend_test OK: unconnected-holder re-establishment " << elapsed << " ms"
         << endl;

    c->disconnect("AgentD");
    d->disconnect("AgentC");
    // Delete the replyer (D) first: its CM thread must be gone before the peer
    // it replies to frees its receive buffers.
    delete d;
    delete c;

    a->disconnect("AgentB");
    b->disconnect("AgentA");
    // Same ordering: B is the holder replying to A.
    delete b;
    delete a;
    return 0;
}
