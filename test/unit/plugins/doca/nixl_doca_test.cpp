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

#include "common.h"

static void checkCudaError(cudaError_t result, const char *message) {
	if (result != cudaSuccess) {
		std::cerr << message << " (Error code: " << result << " - "
				   << cudaGetErrorString(result) << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char *argv[])
{
	std::string             role;
	void                    *addr_initiator[NUM_TRANSFERS];
	void                    *addr_target[NUM_TRANSFERS];
	nixlAgentConfig         cfg(true);
	nixl_b_params_t         params;
	nixlBlobDesc            buf_initiator[NUM_TRANSFERS];
	nixlBlobDesc            buf_target[NUM_TRANSFERS];
	nixlBackendH            *doca_initiator;
	nixlBackendH            *doca_target;
	nixlBlobDesc			desc;
	nixlXferReqH            *treq;
	std::string             name_plugin = "DOCA";
	nixl_opt_args_t extra_params_initiator;
	nixl_opt_args_t extra_params_target;
	nixl_reg_dlist_t dram_for_doca_initiator(DRAM_SEG);
	nixl_reg_dlist_t dram_for_doca_target(DRAM_SEG);
	cudaStream_t stream_initiator, stream_target;
	nixl_status_t status;
	std::string name;
	nixl_blob_t metadata_target;
	nixl_blob_t metadata_target_connect;
	nixl_blob_t metadata_initiator;
	nixl_blob_t metadata_initiator_connect;
	nixlSerDes *serdes_target    = new nixlSerDes();
	nixlSerDes *serdes_target_connect    = new nixlSerDes();
	nixlSerDes *serdes_initiator = new nixlSerDes();
	nixlSerDes *serdes_initiator_connect = new nixlSerDes();
	int leastPriority, greatestPriority;

	params["network_devices"] = "mlx5_0";
	params["gpu_devices"] = "8A:00.0";

	checkCudaError(cudaSetDevice(0), "Failed to set device");
	cudaFree(0);
	checkCudaError(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority), "Failed to get GPU stream priorities");
	checkCudaError(cudaStreamCreateWithPriority(&stream_initiator, cudaStreamNonBlocking, greatestPriority), "Failed to create CUDA stream");
	checkCudaError(cudaStreamCreateWithPriority(&stream_target, cudaStreamNonBlocking, leastPriority), "Failed to create CUDA stream");

	launch_warmup_kernel(stream_initiator, 0, 0);
	launch_warmup_kernel(stream_target, 0, 0);

	std::cout << "Starting Agent for DOCA Test\n";

	/* ********************************* Initiator ********************************* */

	nixlAgent agent_initiator("doca_initiator", cfg);
	agent_initiator.createBackend(name_plugin, params, doca_initiator);
	if (doca_initiator == nullptr) {
		std::cerr <<"Error creating a new backend\n";
		exit(-1);
	}
	extra_params_initiator.backends.push_back(doca_initiator);

	std::cout << "DOCA Backend initiator created\n";

	for (int i = 0; i < NUM_TRANSFERS; i++) {
		checkCudaError(cudaMalloc(&addr_initiator[i], SIZE), "Failed to allocate CUDA buffer 0");
		checkCudaError(cudaMemset(addr_initiator[i], 0, SIZE), "Failed to memset CUDA buffer 0");
		buf_initiator[i].addr  = (uintptr_t)(addr_initiator[i]);
		buf_initiator[i].len   = SIZE;
		buf_initiator[i].devId = 0;
		dram_for_doca_initiator.addDesc(buf_initiator[i]);
		std::cout << "GPU alloc buffer " << i << "\n";
	}
	agent_initiator.registerMem(dram_for_doca_initiator, &extra_params_initiator);
	std::cout << "DOCA initiator registerMem local\n";

	/* ********************************* Target ********************************* */

	nixlAgent agent_target("doca_target", cfg);
	agent_target.createBackend(name_plugin, params, doca_target);
	if (doca_target == nullptr) {
		std::cerr <<"Error creating a new backend\n";
		exit(-1);
	}
	extra_params_target.backends.push_back(doca_target);

	std::cout << "DOCA Backend target created\n";

	/* As this is a unit test single peer, fake the remote memory with different local memory */
	for (int i = 0; i < NUM_TRANSFERS; i++) {
		checkCudaError(cudaMalloc(&addr_target[i], SIZE), "Failed to allocate CUDA buffer 0");
		checkCudaError(cudaMemset(addr_target[i], 0, SIZE), "Failed to memset CUDA buffer 0");
		buf_target[i].addr  = (uintptr_t)(addr_target[i]);
		buf_target[i].len   = SIZE;
		buf_target[i].devId = 0;
		dram_for_doca_target.addDesc(buf_target[i]);
		std::cout << "GPU alloc buffer " << i << "\n";
	}
	std::cout << "DOCA registerMem remote\n";
	agent_target.registerMem(dram_for_doca_target, &extra_params_target);

	/* ********************************* Single process handshake ********************************* */

	agent_target.getLocalMD(metadata_target);
	agent_initiator.getLocalMD(metadata_initiator);

	//Init target for remote
	assert(serdes_target->addStr("AgentMD", metadata_target) == NIXL_SUCCESS);
	assert(dram_for_doca_target.trim().serialize(serdes_target) == NIXL_SUCCESS);

	//Init initiator for remote
	assert(serdes_initiator->addStr("AgentMD", metadata_initiator) == NIXL_SUCCESS);
	assert(dram_for_doca_initiator.trim().serialize(serdes_initiator) == NIXL_SUCCESS);

	/* Fake target -> initiator socket send/recv */
	std::cout << "Connect Initiator\n";
	serdes_initiator_connect->importStr(serdes_target->exportStr());
	metadata_initiator_connect = serdes_initiator_connect->getStr("AgentMD");
	assert (metadata_initiator_connect != "");
	agent_initiator.loadRemoteMD(metadata_initiator_connect, name);

	/* Fake initiator -> target socket send/recv */
	std::cout << "Connect Target\n";
	serdes_target_connect->importStr(serdes_initiator->exportStr());
	metadata_target_connect = serdes_target_connect->getStr("AgentMD");
	assert (metadata_target_connect != "");
	agent_target.loadRemoteMD(metadata_target_connect, name);

	/* ********************************* Create initiator -> target Xfer req ********************************* */

	nixl_xfer_dlist_t dram_initiator_doca = dram_for_doca_initiator.trim();
	nixl_xfer_dlist_t dram_target_doca(serdes_initiator_connect);

	extra_params_initiator.customParam = (uintptr_t)stream_initiator;
	status = agent_initiator.createXferReq(NIXL_WRITE, dram_initiator_doca, dram_target_doca, "doca_target", treq, &extra_params_initiator);
	if (status != NIXL_SUCCESS) {
		std::cerr << "Error creating transfer request\n";
		exit(-1);
	}

	std::cout << "Launch initiator send kernel on stream\n";
	launch_initiator_send_kernel(stream_initiator, buf_initiator[0].addr, buf_initiator[0].len);
	std::cout << "Post the request with DOCA backend\n ";
	status = agent_initiator.postXferReq(treq);
	std::cout << "Waiting for completion\n";
	/* Weird behaviour: even if stream non-blocking with lowest priority
	 * still it prevents other cuda kernels to be launched in parallel
	 * while it's still active.
	 * Can't put it befeore initiator or postXfer.
	 * Nosense CUDA driver issue...
	 */
	std::cout << "Launch target wait kernel on stream\n";
	launch_target_wait_kernel(stream_target, buf_target[0].addr, buf_target[0].len);

	while (status != NIXL_SUCCESS) {
		status = agent_initiator.getXferStatus(treq);
		assert(status >= 0);
	}
	std::cout <<" Completed writing data using DOCA backend\n";
	agent_initiator.releaseXferReq(treq);

	std::cout << "Closing.. \n";
	cudaStreamSynchronize(stream_initiator);
	cudaStreamDestroy(stream_initiator);
	cudaStreamSynchronize(stream_target);
	cudaStreamDestroy(stream_target);

	std::cout << "Memory cleanup.. \n";
	agent_initiator.deregisterMem(dram_for_doca_initiator, &extra_params_initiator);
	agent_target.deregisterMem(dram_for_doca_target, &extra_params_target);

	/** Argument Parsing */
	// if (argc < 2) {
	//     std::cout <<"Enter the required arguments\n" << std::endl;
	//     std::cout <<"Directory path " << std::endl;
	//     exit(-1);
	// }

		return 0;
}
