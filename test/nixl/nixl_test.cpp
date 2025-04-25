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
#include <iostream>
#include <string>
#include <algorithm>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"
#define NUM_TRANSFERS 1
#define SIZE 1024

/**
 * This test does p2p from using PUT.
 * intitator -> target so the metadata and
 * desc list needs to move from
 * target to initiator
 */

bool allBytesAre(void* buffer, size_t size, uint8_t value) {
    uint8_t* byte_buffer = static_cast<uint8_t*>(buffer); // Cast void* to uint8_t*
    // Iterate over each byte in the buffer
    for (size_t i = 0; i < size; ++i) {
        if (byte_buffer[i] != value) {
            return false; // Return false if any byte doesn't match the value
        }
    }
    return true; // All bytes match the value
}

int main(int argc, char *argv[]) {
    int                     target_port;
    nixl_status_t           ret = NIXL_SUCCESS;
    void                    *addr[NUM_TRANSFERS];
    std::string             role;
    const char              *target_ip;
    nixl_blob_t             remote_desc;
    nixl_blob_t             tgt_metadata;
    nixl_blob_t             tgt_md_init;
    int                     status = 0;
    bool                    rc = false;

    /** NIXL declarations */
    /** Agent and backend creation parameters */
    nixl_b_params_t params;
    nixlBlobDesc    buf[NUM_TRANSFERS];
    nixlBackendH    *ucx;

    /** Serialization/Deserialization object to create a blob */
    nixlSerDes *serdes        = new nixlSerDes();
    nixlSerDes *remote_serdes = new nixlSerDes();

    /** Descriptors and Transfer Request */
    nixl_reg_dlist_t  dram_for_ucx(DRAM_SEG);
    nixlXferReqH      *treq;

    /** Argument Parsing */
    if (argc < 4) {
        std::cout <<"Enter the required arguments\n" << std::endl;
        std::cout <<"<Role> " <<"Target IP> <Target Port>"
                  << std::endl;
        exit(-1);
    }

    role = std::string(argv[1]);
    target_ip   = argv[2];
    target_port = std::stoi(argv[3]);
    std::transform(role.begin(), role.end(), role.begin(), ::tolower);

    if (!role.compare("initiator") && !role.compare("target")) {
            std::cerr << "Invalid role. Use 'initiator' or 'target'."
                      << "Currently "<< role <<std::endl;
            return 1;
    }

    nixlAgentConfig cfg(true);
    if ( role == "target" ) {
        cfg = nixlAgentConfig(true, true, target_port);
    } else {
        //use default port
        cfg = nixlAgentConfig(true, true);
    }


    /*** End - Argument Parsing */

    /** Common to both Initiator and Target */
    std::cout << "Starting Agent for "<< role << "\n";
    nixlAgent     agent(role, cfg);
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    for (int i = 0; i < NUM_TRANSFERS; i++) {
        addr[i] = calloc(1, SIZE);
        if (role != "target") {
            memset(addr[i], 0xbb, SIZE);
            std::cout << "Allocating for initiator : "
                      << addr[i] << ", "
                      << "Setting to 0xbb "
                      << std::endl;
        } else {
            memset(addr[i], 0, SIZE);
            std::cout << "Allocating for target : "
                      << addr[i] << ", "
                      << "Setting to 0 " << std::endl;
        }
        buf[i].addr  = (uintptr_t)(addr[i]);
        buf[i].len   = SIZE;
        buf[i].devId = 0;
        dram_for_ucx.addDesc(buf[i]);
    }

    /** Register memory in both initiator and target */
    agent.registerMem(dram_for_ucx, &extra_params);
    if (role == "target") {
        agent.getLocalMD(tgt_metadata);
    }

    std::cout << " Start Control Path metadata exchanges \n";
    if (role == "target") {
        std::cout << " Desc List from Target to Initiator\n";
        dram_for_ucx.print();

        /** Only send desc list */
        assert(dram_for_ucx.trim().serialize(serdes) == NIXL_SUCCESS);

        std::cout << " Wait for initiator and then send xfer descs\n";

        std::string message = serdes->exportStr();

        do{
            ret = agent.genNotif("initiator", message, &extra_params);
        } while(ret != NIXL_SUCCESS);

        std::cout << " End Control Path metadata exchanges \n";

        std::cout << " Start Data Path Exchanges \n";
        std::cout << " Waiting to receive Data from Initiator\n";

        while (!rc) {
            //Only works with progress thread now, as backend is protected
            /** Sanity Check */
            for (int i = 0; i < NUM_TRANSFERS; i++) {
                rc = allBytesAre(addr[i], SIZE, 0xbb);
                if (!rc)
                    break;
            }
        }
        if (!rc)
            std::cerr << " UCX Transfer failed, buffers are different\n";
        else
            std::cout << " Transfer completed and Buffers match with Initiator\n"
                      <<"  UCX Transfer Success!!!\n";

    } else {

        std::cout << " Exchange metadata with Target \n";
        nixl_opt_args_t md_extra_params;
        md_extra_params.ipAddr = target_ip;
        md_extra_params.port = target_port;

        agent.fetchRemoteMD("target", &md_extra_params);

        agent.sendLocalMD(&md_extra_params);

        nixl_notifs_t notifs;

        while(notifs.size() == 0) {
            ret = agent.getNotifs(notifs, &extra_params);
            assert(ret >= 0);
        }
        std::string rrstr = notifs["target"][0];
        assert(rrstr.size() > 0);

        remote_serdes->importStr(rrstr);

        std::cout << " Verify Deserialized Target's Desc List at Initiator\n";
        nixl_xfer_dlist_t dram_target_ucx(remote_serdes);
        nixl_xfer_dlist_t dram_initiator_ucx = dram_for_ucx.trim();
        dram_target_ucx.print();

        std::cout << " End Control Path metadata exchanges \n";
        std::cout << " Start Data Path Exchanges \n\n";
        std::cout << " Create transfer request with UCX backend\n ";

        // Need to do this in a loop with NIXL_ERR_NOT_FOUND
        // UCX AM with desc list is faster than listener thread can recv/load MD with sockets
        // Will be deprecated with ETCD or callbacks
        do {
            ret = agent.createXferReq(NIXL_WRITE, dram_initiator_ucx, dram_target_ucx,
                                      "target", treq, &extra_params);
        } while (ret == NIXL_ERR_NOT_FOUND);

        if (ret != NIXL_SUCCESS) {
            std::cerr << "Error creating transfer request " << ret <<"\n";
            exit(-1);
        }

        std::cout << " Post the request with UCX backend\n ";
        status = agent.postXferReq(treq);
        std::cout << " Initiator posted Data Path transfer\n";
        std::cout << " Waiting for completion\n";

        while (status != NIXL_SUCCESS) {
            status = agent.getXferStatus(treq);
            assert(status >= 0);
        }
        std::cout << " Completed Sending Data using UCX backend\n";
        agent.releaseXferReq(treq);
        agent.invalidateLocalMD(&md_extra_params);
    }

    std::cout <<"Cleanup.. \n";
    agent.deregisterMem(dram_for_ucx, &extra_params);
    for (int i = 0; i < NUM_TRANSFERS; i++) {
        free(addr[i]);
    }
    delete serdes;
    delete remote_serdes;

    return 0;
}
