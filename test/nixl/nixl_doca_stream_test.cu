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

#define CUDA_THREADS 512
#define TRANSFER_NUM_BUFFER 32
#define SIZE 1024
#define INITIATOR_VALUE 0xbb
#define VOLATILE(x) (*(volatile typeof(x) *)&(x))
#define INITIATOR_THRESHOLD_NS 50000 //50us
#define USE_NVTX 1

#if USE_NVTX
#include <nvtx3/nvToolsExt.h>


const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define DEVICE_GET_TIME(globaltimer) asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer))

#define PUSH_RANGE(name,cid) { \
	int color_id = cid; \
	color_id = color_id%num_colors;\
	nvtxEventAttributes_t eventAttrib = {0}; \
	eventAttrib.version = NVTX_VERSION; \
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
	eventAttrib.colorType = NVTX_COLOR_ARGB; \
	eventAttrib.color = colors[color_id]; \
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
	eventAttrib.message.ascii = name; \
	nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

static void checkCudaError(cudaError_t result, const char *message) {
	if (result != cudaSuccess) {
		std::cerr << message << " (Error code: " << result << " - "
				   << cudaGetErrorString(result) << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

__global__ void target_kernel(uintptr_t addr, uint8_t val)
{
	uint8_t ok = 1;
	uintptr_t buffer_addr = addr + (threadIdx.x * SIZE);

	printf(">>>>>>> CUDA target waiting on buffer %d addr %lx size %d\n",
			threadIdx.x, buffer_addr, (uint32_t)SIZE);

	while(VOLATILE(((uint8_t*)buffer_addr)[0]) == 0);

	for (int i = 0; i < (int)SIZE; i++) {
		if (((uint8_t*)buffer_addr)[i] != val) {
			printf(">>>>>>> CUDA target byte %x is wrong\n", i);
			ok = 1;
		}
	}
	if (ok == 1)
		printf(">>>>>>> CUDA target, all bytes received! val=%d\n", val);
	else
		printf(">>>>>>> CUDA target, not all received bytes are ok!\n");
}

int launch_target_wait_kernel(cudaStream_t stream, uintptr_t addr, uint8_t val)
{
	cudaError_t result = cudaSuccess;

	/* Check no previous CUDA errors */
	result = cudaGetLastError();
	if (result != cudaSuccess) {
		fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
		return -1;
	}

	target_kernel<<<1, TRANSFER_NUM_BUFFER, 0, stream>>>(addr, val);
	result = cudaGetLastError();
	if (result != cudaSuccess) {
		fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
		return -1;
	}

	return 0;
}

__global__ void initiator_kernel(uintptr_t addr)
{
	unsigned long long start, end;
	// Each block updates a buffer in this transfer
	uintptr_t block_address = (addr + (blockIdx.x * SIZE));

	/* Simulate a longer CUDA kernel to process initiator data */
	DEVICE_GET_TIME(start);

	for (int i = threadIdx.x; i < SIZE; i+=blockDim.x)
		((uint8_t*)block_address)[i] = INITIATOR_VALUE;

	__syncthreads();

	do {
		DEVICE_GET_TIME(end);
	} while (end - start < INITIATOR_THRESHOLD_NS);
}

int launch_initiator_send_kernel(cudaStream_t stream, uintptr_t addr)
{
	cudaError_t result = cudaSuccess;

	/* Check no previous CUDA errors */
	result = cudaGetLastError();
	if (result != cudaSuccess) {
		fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
		return -1;
	}

	// Block = # buffers x transfer
	initiator_kernel<<<TRANSFER_NUM_BUFFER, CUDA_THREADS, 0, stream>>>(addr);
	result = cudaGetLastError();
	if (result != cudaSuccess) {
		fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
		return -1;
	}

	return 0;
}

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

std::string recvFromTarget(int port) {
	nixlMDStreamListener listener(port);
	listener.setupListenerSync();
	listener.acceptClient();
	return listener.recvFromClient();
}

void sendToInitiator(const char *ip, int port, std::string data) {
	nixlMDStreamClient client(ip, port);
	client.connectListenerSync();
	client.sendData(data);
}

int main(int argc, char *argv[]) {
	int                     peer_port;
	nixl_status_t           ret = NIXL_SUCCESS;
	uint8_t                 *data_address;
	std::string             role;
	std::string             processing;
	const char              *peer_ip;
	nixl_blob_t             remote_desc;
	nixl_blob_t             metadata;
	nixl_blob_t             remote_metadata;
	int                     status = 0;
	static std::string target("target");
	static std::string initiator("initiator");

	/** NIXL declarations */
	/** Agent and backend creation parameters */
	nixlAgentConfig cfg(true);
	nixl_b_params_t params;
	nixlBlobDesc    buf[TRANSFER_NUM_BUFFER];
	nixlBackendH    *doca;
	cudaStream_t    stream;
	/** Serialization/Deserialization object to create a blob */
	nixlSerDes *serdes        = new nixlSerDes();
	nixlSerDes *remote_serdes = new nixlSerDes();

	/** Descriptors and Transfer Request */
	nixl_reg_dlist_t  dram_for_doca(DRAM_SEG);
	nixlXferReqH      *treq;
	nixl_notifs_t notifs;

	/** Argument Parsing */
	if (argc < 5) {
		std::cout <<"Enter the required arguments\n" << std::endl;
		std::cout <<"<Role> <Peer IP> <Peer Port> <CPU or GPU processing>"
				  << std::endl;
		exit(-1);
	}

	role = std::string(argv[1]);
	std::transform(role.begin(), role.end(), role.begin(), ::tolower);
	if (!role.compare(initiator) && !role.compare(target)) {
			std::cerr << "Invalid role. Use 'initiator' or 'target'."
					  << "Currently "<< role <<std::endl;
			return 1;
	}

	peer_ip   = argv[2];
	peer_port = std::stoi(argv[3]);
	processing = std::string(argv[4]);
	std::transform(processing.begin(), processing.end(), processing.begin(), ::tolower);
	if (!processing.compare("cpu") && !processing.compare("gpu")) {
			std::cerr << "Invalid type of processing. Use 'cpu' or 'gpu'."
					  << "Currently "<< processing <<std::endl;
			return 1;
	}

	/*** End - Argument Parsing */
	checkCudaError(cudaSetDevice(0), "Failed to set device");
	cudaFree(0);

	/** Common to both Initiator and Target */
	std::cout << "Starting Agent for "<< role << "\n";
	nixlAgent     agent(role, cfg);
	params["network_devices"] = "mlx5_0";
	params["gpu_devices"] = "0";
	PUSH_RANGE("createBackend", 0)
	agent.createBackend("DOCA", params, doca);
	POP_RANGE

	nixl_opt_args_t extra_params;
	extra_params.backends.push_back(doca);

	checkCudaError(cudaMalloc(&data_address, SIZE * TRANSFER_NUM_BUFFER), "Failed to allocate CUDA buffer 0");
	checkCudaError(cudaMemset((void*)data_address, 0, SIZE * TRANSFER_NUM_BUFFER), "Failed to memset CUDA buffer 0");

	if (role != target) {
		std::cout << "Allocating for initiator : "
				  << TRANSFER_NUM_BUFFER << " buffers "
				  << SIZE << " Bytes each "
				  << (void*)data_address << " address "
				  << std::endl;
	} else {
		std::cout << "Allocating for target : "
				  << TRANSFER_NUM_BUFFER << " buffers "
				  << SIZE << " Bytes each "
				  << (void*)data_address << " address "
				  << std::endl;
	}

	for (int i = 0; i < TRANSFER_NUM_BUFFER; i++) {
		buf[i].addr  = (uintptr_t)(data_address + (i * SIZE));
		buf[i].len   = SIZE;
		buf[i].devId = 0;
		dram_for_doca.addDesc(buf[i]);
	}
	/** Register memory in both initiator and target */
	agent.registerMem(dram_for_doca, &extra_params);
	agent.getLocalMD(metadata);

	std::cout << " Start Control Path metadata exchanges \n";
	if (role == target) {
		bool found = false;
		//Not used
		#if USE_FETCH_REMOTE_MD
			std::string message = serdes->exportStr();
			if (agent.genNotif(initiator, message, &extra_params) != NIXL_SUCCESS) {
				std::cout << "Can't send notif " << message << std::endl;
			}
		#else
			nixlMDStreamClient client(peer_ip, peer_port);
			client.connectListenerSync();
			nixlMDStreamListener listener(peer_port);
			listener.setupListenerSync();
		
			std::cout << " Desc List from Target to Initiator\n";
			dram_for_doca.print();

			//Send local MD to remote initiator
			assert(serdes->addStr("AgentMD", metadata) == NIXL_SUCCESS);
			assert(dram_for_doca.trim().serialize(serdes) == NIXL_SUCCESS);
			client.sendData(serdes->exportStr());
			std::cout << " End Control Path metadata exchanges \n";

			//Wait to receive remote MD from remote initiator
			//Not required by DOCA Backend but needed by Agent to populate remoteBackends array
			//Without this step, can't call genNotif() from target as the remoteBackends doesn't
			//have populated initiator entry.
			listener.acceptClient();
			std::string rrstr = listener.recvFromClient();
			remote_serdes->importStr(rrstr);
			remote_metadata = remote_serdes->getStr("AgentMD");
			assert (remote_metadata != "");
			agent.loadRemoteMD(remote_metadata, initiator);

			std::cout << " Serialize Metadata to string and Send to Initiator\n";
			std::cout << " \t -- To be handled by runtime - currently sent via a TCP Stream\n";
		#endif

		//First recv notif: initiator ack it connected correctly
		do {
			nixl_status_t ret = agent.getNotifs(notifs);
		} while(notifs.size() == 0);

		for (const auto& n : notifs) {
			for (size_t idx = 0; idx < n.second.size(); idx++) {
				std::cout << "Received message from " << n.first << " msg: " << n.second[idx] << " at " << idx << std::endl;

				if (n.first == initiator && n.second[idx] == "connected") {
					std::cout << "Received correct message from " << n.first << " msg: " << n.second[idx] << " at " << idx << std::endl;
					break;
				}
			}
		}

		std::cout << " Start Data Path Exchanges \n";
		std::cout << " Waiting to receive Data from Initiator\n";

		checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Failed to create CUDA stream");
	
		//Second recv notif: initiator ack data has been sent
		do {
			for (const auto& n : notifs) {
				for (size_t idx = 0; idx < n.second.size(); idx++) {
					if (n.first == initiator && n.second[idx] == "sent") {
						std::cout << "Received correct message from " << n.first << " msg: " << n.second[idx] << " at " << idx << std::endl;
						launch_target_wait_kernel(stream, (uintptr_t)(data_address), INITIATOR_VALUE);
						cudaStreamSynchronize(stream);
						std::cout << " DOCA Transfer completed -- first!\n";
						found = true;
						break;
					}
				}
			}
			nixl_status_t ret = agent.getNotifs(notifs);
		} while (found == false);

		notifs.clear();

		//First send notif: target processed previously sent data
		std::string msg = "processed";
		ret = agent.genNotif(initiator, msg, &extra_params);
		if(ret != NIXL_SUCCESS) {
			std::cerr << "Target genNotif error " << ret << "\n";
		}
		found = false;

		std::cout << " Waiting for second 'sent' notif\n";
		//Third recv notif: sent
		do {
			for (const auto& n : notifs) {
				for (size_t idx = 0; idx < n.second.size(); idx++) {
					if (n.first == initiator && n.second[idx] == "sent") {
						std::cout << "Received correct message from " << n.first << " msg: " << n.second[idx] << " at " << idx << std::endl;
						launch_target_wait_kernel(stream, (uintptr_t)(data_address), INITIATOR_VALUE+1);
						cudaStreamSynchronize(stream);
						std::cout << " DOCA Transfer completed -- second!\n";
						found = true;
						break;
					}
				}
			}
			nixl_status_t ret = agent.getNotifs(notifs);
			} while (found == false);

		cudaStreamDestroy(stream);
	} else {
		std::cout << " Wait for metadata from Target \n";
		std::cout << " \t -- To be handled by runtime - currently received via a TCP Stream\n";
		
		//Not used
		#if USE_FETCH_REMOTE_MD
			nixl_opt_args_t md_extra_params;
			md_extra_params.ipAddr = peer_ip;
			md_extra_params.port = peer_port;
			agent.fetchRemoteMD(target, &md_extra_params);
			agent.sendLocalMD(&md_extra_params);

			do {
				nixl_status_t ret = agent.getNotifs(notifs);
			} while(notifs.size() == 0);

			for (const auto &notif : notifs[target]) {
				remote_serdes->importStr(notif);
			}

			for (const auto& n : notifs) {
				if (n.first == target && n.second[0] == "connected") {
					std::cout << "Received correct message from " << n.first << " msg: " << n.second[0] << std::endl;
					break;
				} else {
					std::cout << "Received wrong message from " << n.first << " msg: " << n.second[0] << std::endl;
				}
			}
		#else
			//Wait for remote target connection
			nixlMDStreamListener listener(peer_port);
			listener.setupListenerSync();
			listener.acceptClient();
			std::string rrstr = listener.recvFromClient();
			remote_serdes->importStr(rrstr);
			remote_metadata = remote_serdes->getStr("AgentMD");
			assert (remote_metadata != "");
			agent.loadRemoteMD(remote_metadata, target);

			//Wait target to open listener
			sleep(2);
			//Send to remote target local connection info.
			//Not needed by DOCA backend, required by NIXL Agent (see above)
			nixlMDStreamClient client(peer_ip, peer_port);
			client.connectListenerSync();
			assert(serdes->addStr("AgentMD", metadata) == NIXL_SUCCESS);
			assert(dram_for_doca.trim().serialize(serdes) == NIXL_SUCCESS);
			client.sendData(serdes->exportStr());
			std::cout << " End Control Path metadata exchanges \n";
		#endif

		//First send notif: connected
		std::string msg = "connected";
		ret = agent.genNotif(target, msg);
		if(ret != NIXL_SUCCESS) {
			std::cerr << "Target genNotif error " << ret << "\n";
		}

		std::cout << " Verify Deserialized Target's Desc List at Initiator\n";
		nixl_xfer_dlist_t dram_target_doca(remote_serdes);
		nixl_xfer_dlist_t dram_initiator_doca = dram_for_doca.trim();
		dram_target_doca.print();
		std::cout << " Got metadata from " << target << " \n";
		std::cout << " Create transfer request with DOCA backend\n ";

		PUSH_RANGE("createXferReq", 1)

		//Create Xfer request with notification
		if (processing.compare("gpu") == 0)
	        checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Failed to create CUDA stream");

		if (processing.compare("gpu") == 0) {
			extra_params.customParam.resize(sizeof(uintptr_t));
			*((uintptr_t*) extra_params.customParam.data()) = (uintptr_t)stream;
		}
		extra_params.notifMsg = "sent";
		extra_params.hasNotif = true;
		ret = agent.createXferReq(NIXL_WRITE, dram_initiator_doca, dram_target_doca,
						target, treq, &extra_params);
		if (ret != NIXL_SUCCESS) {
			std::cerr << "Error creating transfer request\n";
			exit(-1);
		}
		POP_RANGE

		std::cout << "Launch initiator send kernel on stream\n";

		/* Synthetic simulation of GPU processing data before sending */
		if (processing.compare("gpu") == 0) {
			std::cout << " Prepare data, GPU mode, transfer 1" << std::endl;
			PUSH_RANGE("InitData", 2)
			launch_initiator_send_kernel(stream, (uintptr_t)(data_address));
			POP_RANGE

			std::cout << " Post the request with DOCA backend transfer 1" << std::endl;
			PUSH_RANGE("postXferReq", 3)
			status = agent.postXferReq(treq);
			assert(status >= NIXL_SUCCESS);
			POP_RANGE
		} else {
			/* Synthetic simulation of CPU processing data before sending */
			std::cout << "First xfer, prepare data, CPU mode, transfer 1" << std::endl;
			PUSH_RANGE("InitData", 2)
			cudaMemset((void*)data_address, INITIATOR_VALUE, TRANSFER_NUM_BUFFER * SIZE);
			POP_RANGE

			std::cout << " Post the request with DOCA backend transfer 1" << std::endl;
			PUSH_RANGE("postXferReq", 3)
			status = agent.postXferReq(treq);
			assert(status >= NIXL_SUCCESS);
			POP_RANGE

			std::cout << " Waiting for completion\n";
			PUSH_RANGE("getXferStatus", 4)
			while (status != NIXL_SUCCESS) {
				status = agent.getXferStatus(treq);
				assert(status >= NIXL_SUCCESS);
			}
			POP_RANGE

			std::cout << "Second xfer, prepare data, CPU mode, transfer 2" << std::endl;
			PUSH_RANGE("InitData", 2)
			cudaMemset((void*)data_address, INITIATOR_VALUE + 1, TRANSFER_NUM_BUFFER * SIZE);
			POP_RANGE

			//First recv notif: target processed previously sent data
			do {
				nixl_status_t ret = agent.getNotifs(notifs);
			} while(notifs.size() == 0);

			for (const auto& n : notifs) {
				for (size_t idx = 0; idx < n.second.size(); idx++) {
					if (n.first == target && n.second[idx] == "processed") {
						std::cout << "Received correct message from " << n.first << " msg: " << n.second[idx] << " at " << idx << std::endl;
						break;
					}
				}
			}

			//Repost same treq with different data in buffers
			std::cout << " Post the request with DOCA backend transfer 2" << std::endl;
			PUSH_RANGE("postXferReq", 3)
			status = agent.postXferReq(treq);
			assert(status >= NIXL_SUCCESS);
			POP_RANGE

			std::cout << " Waiting for completion\n";
			PUSH_RANGE("getXferStatus", 4)
			while (status != NIXL_SUCCESS) {
				status = agent.getXferStatus(treq);
				assert(status >= NIXL_SUCCESS);
			}
			POP_RANGE
		}

		std::cout << "Releasing request " << std::endl;
		agent.releaseXferReq(treq);
	
		if (processing.compare("gpu") == 0) {
			cudaStreamSynchronize(stream);
			cudaStreamDestroy(stream);
		}
	}

	std::cout <<"Cleanup.. \n";
	
	agent.deregisterMem(dram_for_doca, &extra_params);
	// cudaFree(data_address);

	if (role == "target")
		delete serdes;
	else
		delete remote_serdes;

	std::cout <<"Exit.. \n";

	return 0;
}
