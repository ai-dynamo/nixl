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
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <cstring>

#define NUM_TRANSFERS 250
#define SIZE 10485760


std::string generate_timestamped_filename(const std::string& base_name) {
    std::time_t t = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp),
                  "%Y%m%d%H%M%S", std::localtime(&t));
    std::string return_val = base_name + std::string(timestamp);

    return return_val;
}

int main(int argc, char *argv[])
{
        nixl_status_t           ret = NIXL_SUCCESS;
        void                    *addr[NUM_TRANSFERS];
        std::string             role;
        int                     status = 0;
        int                     i;
        int                     fd[NUM_TRANSFERS];

        nixlAgentConfig         cfg(true);
        nixl_b_params_t         params;
        nixlStringDesc          buf[NUM_TRANSFERS];
        nixlStringDesc          ftrans[NUM_TRANSFERS];
        nixlBackendH	        *gds;

        nixl_reg_dlist_t        vram_for_gds(VRAM_SEG);
        nixl_reg_dlist_t        file_for_gds(FILE_SEG, false);
        nixlXferReqH            *treq;
        std::string             name;

        std::cout << "Starting Agent for " << "GDS Test Agent" << "\n";
        nixlAgent agent("GDSTester", cfg);

        gds          = agent.createBackend("GDS", params);
        if (gds == nullptr) {
		std::cerr <<"Error creating a new backend\n";
		exit(-1);
	}
        /** Argument Parsing */
        if (argc < 2) {
                std::cout <<"Enter the required arguments\n" << std::endl;
                std::cout <<"Directory path " << std::endl;
                exit(-1);
        }

        for (i = 0; i < NUM_TRANSFERS; i++) {
                cudaMalloc((void **)&addr[i], SIZE);
                cudaMemset(addr[i], 'A', SIZE);
		name = generate_timestamped_filename("testfile");
	        std::string path = std::string(argv[1]);
		name = path +"/"+ name +"_"+ std::to_string(i);

		std::cout << "Opening File " << name << std::endl;
		fd[i] = open(name.c_str(), O_RDWR|O_CREAT, 0744);
	        if (fd[i] < 0) {
		   std::cerr<<"Error: "<<strerror(errno)<<std::endl;
		   std::cerr<<"Open call failed to open file\n";
	           std::cerr<<"Cannot run tests\n";
	           return 1;
		}
	        std::cout << "Opened File " << name << std::endl;

		std::cout << "Allocating for src buffer : "
                          << addr[i] << ","
                          << "Setting to As "
                          << std::endl;
                /* If O_CREATE is specified, mode flags need to be specified */
                /* Use 0744 as mode */
                buf[i].addr   = (uintptr_t)(addr[i]);
                buf[i].len    = SIZE;
                buf[i].devId  = 0;
                vram_for_gds.addDesc(buf[i]);

                ftrans[i].addr  = 0; // this is offset
                ftrans[i].len   = SIZE;
                ftrans[i].devId = fd[i];
                file_for_gds.addDesc(ftrans[i]);
        }
        agent.registerMem(file_for_gds, gds);
        agent.registerMem(vram_for_gds, gds);

        nixl_xfer_dlist_t vram_for_gds_list = vram_for_gds.trim();
        nixl_xfer_dlist_t file_for_gds_list = file_for_gds.trim();

        ret = agent.createXferReq(vram_for_gds_list, file_for_gds_list,
                                  "GDSTester", "", NIXL_WRITE, treq);
        if (ret != NIXL_SUCCESS) {
                std::cerr << "Error creating transfer request\n" << ret;
                exit(-1);
        }

        std::cout << " Post the request with GDS backend\n ";
        status = agent.postXferReq(treq);
        std::cout << " GDS File IO has been posted\n";
        std::cout << " Waiting for completion\n";

        while (status != NIXL_SUCCESS) {
            status = agent.getXferStatus(treq);
            assert(status >= 0);
        }
        std::cout <<" Completed writing data using GDS backend\n";
        agent.invalidateXferReq(treq);

        std::cout <<"Cleanup.. \n";
        agent.deregisterMem(vram_for_gds, gds);
        agent.deregisterMem(file_for_gds, gds);

        return 0;
}
