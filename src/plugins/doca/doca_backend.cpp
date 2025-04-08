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
#include "doca_backend.h"
#include "serdes/serdes.h"
#include <cassert>
#include <stdexcept>

DOCA_LOG_REGISTER(NIXL::DOCA);

/****************************************
 * DOCA request management
*****************************************/

void nixlDocaEngine::_requestInit(void *request)
{
	/* Initialize request in-place (aka "placement new")*/
	new(request) nixlDocaBckndReq;
}

void nixlDocaEngine::_requestFini(void *request)
{
	/* Finalize request */
	nixlDocaBckndReq *req = (nixlDocaBckndReq*)request;
	req->~nixlDocaBckndReq();
}

/****************************************
 * Constructor/Destructor
*****************************************/

static doca_error_t
open_doca_device_with_ibdev_name(const uint8_t *value, size_t val_size, struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	char buf[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
	char val_copy[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
	doca_error_t res;
	size_t i;

	/* Set default return value */
	*retval = NULL;

	/* Setup */
	if (val_size > DOCA_DEVINFO_IBDEV_NAME_SIZE) {
		DOCA_LOG_ERR("Value size too large. Failed to locate device");
		return DOCA_ERROR_INVALID_VALUE;
	}
	memcpy(val_copy, value, val_size);

	res = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to load doca devices list. Doca_error value");
		return res;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		res = doca_devinfo_get_ibdev_name(dev_list[i], buf, DOCA_DEVINFO_IBDEV_NAME_SIZE);
		if (res == DOCA_SUCCESS && strncmp(buf, val_copy, val_size) == 0) {
			/* If any special capabilities are needed */
			/* if device can be opened */
			res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_destroy_list(dev_list);
				return res;
			}
		}
	}

	DOCA_LOG_ERR("Matching device not found");

	res = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_destroy_list(dev_list);
	return res;
}

nixlDocaEngine::nixlDocaEngine (const nixlBackendInitParams* init_params)
: nixlBackendEngine (init_params)
{
	std::vector<std::string> ndevs, tmp_gdevs; /* Empty vector */
	doca_error_t result;
	nixl_b_params_t* custom_params = init_params->customParams;

	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		throw std::invalid_argument("Can't initialize doca log");

	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		throw std::invalid_argument("Can't initialize doca log");

	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		throw std::invalid_argument("Can't initialize doca log");

	if (custom_params->count("network_devices") !=0 )
		ndevs = str_split((*custom_params)["network_devices"], " ");
	// Temporary: will extend to more NICs in a dedicated PR
	if (ndevs.size() > 1)
		throw std::invalid_argument("Only 1 network device is allowed");

	std::cout << "DOCA network devices:" << std::endl;
	for (const std::string& str : ndevs) {
		std::cout << str << " ";
	}
	std::cout << std::endl;

	if (custom_params->count("gpu_devices") == 0)
		throw std::invalid_argument("At least 1 GPU device must be specified");
	// Temporary: will extend to more GPUs in a dedicated PR
	if (custom_params->count("gpu_devices") > 1)
		throw std::invalid_argument("Only 1 GPU device is allowed");

	std::cout << "DOCA GPU devices:" << std::endl;
	tmp_gdevs = str_split((*custom_params)["gpu_devices"], " ");
	for (auto &cuda_id : tmp_gdevs) {
		gdevs.push_back(std::pair((uint32_t)std::stoi(cuda_id), nullptr));
		std::cout << "cuda_id " << cuda_id << " int " << std::stoi(cuda_id) << "\n";
	}
	std::cout << std::endl;

	/* Open DOCA device */
	result = open_doca_device_with_ibdev_name((const uint8_t *)(ndevs[0].c_str()),
						  ndevs[0].size(),
						  &(ddev));
	if (result != DOCA_SUCCESS) {
		throw std::invalid_argument("Failed to open DOCA device");
	}

	char pciBusId[DOCA_DEVINFO_IBDEV_NAME_SIZE];
	for (auto &item : gdevs) {
		std::cout << "item.first " << item.first << "\n";
		cudaDeviceGetPCIBusId(pciBusId, DOCA_DEVINFO_IBDEV_NAME_SIZE, item.first);
		result = doca_gpu_create(pciBusId, &item.second);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create DOCA GPU device: %s", doca_error_get_descr(result));
		}
	}

	/* Create DOCA RDMA instance */
	result = doca_rdma_create(ddev, &(rdma));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA RDMA: %s", doca_error_get_descr(result));
	}

	/* Convert DOCA RDMA to general DOCA context */
	rdma_ctx = doca_rdma_as_ctx(rdma);
	if (rdma_ctx == NULL) {
		result = DOCA_ERROR_UNEXPECTED;
		DOCA_LOG_ERR("Failed to convert DOCA RDMA to DOCA context: %s", doca_error_get_descr(result));
	}

	/* Set permissions to DOCA RDMA */
	result = doca_rdma_set_permissions(rdma, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set permissions to DOCA RDMA: %s", doca_error_get_descr(result));
	}

	// /* Set gid_index to DOCA RDMA if it's provided */
	// if (cfg->is_gid_index_set) {
	// 	/* Set gid_index to DOCA RDMA */
	// 	result = doca_rdma_set_gid_index(rdma, cfg->gid_index);
	// 	if (result != DOCA_SUCCESS) {
	// 		DOCA_LOG_ERR("Failed to set gid_index to DOCA RDMA: %s", doca_error_get_descr(result));
	// 	}
	// }

	/* Set send queue size to DOCA RDMA */
	result = doca_rdma_set_send_queue_size(rdma, RDMA_SEND_QUEUE_SIZE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set send queue size to DOCA RDMA: %s", doca_error_get_descr(result));
	}

	/* Setup datapath of RDMA CTX on GPU */
	result = doca_ctx_set_datapath_on_gpu(rdma_ctx, gdevs[0].second);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set datapath on GPU: %s", doca_error_get_descr(result));
	}

	/* Set receive queue size to DOCA RDMA */
	result = doca_rdma_set_recv_queue_size(rdma, RDMA_RECV_QUEUE_SIZE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set receive queue size to DOCA RDMA: %s", doca_error_get_descr(result));
	}

	/* Set GRH to DOCA RDMA */
	result = doca_rdma_set_grh_enabled(rdma, true);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set GRH to DOCA RDMA: %s", doca_error_get_descr(result));
	}

	result = doca_ctx_start(rdma_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start RDMA context: %s", doca_error_get_descr(result));
	}

	result = doca_rdma_get_gpu_handle(rdma, &(rdma_gpu));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to get RDMA GPU handler: %s", doca_error_get_descr(result));
	}

	result = doca_rdma_export(rdma, &(connection_details), &(conn_det_len), &connection);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to export RDMA with connection details");
	}

	//GDRCopy
	result = doca_gpu_mem_alloc(gdevs[0].second, sizeof(struct docaXferReqGpu) * DOCA_XFER_REQ_MAX,
		4096,
		DOCA_GPU_MEM_TYPE_GPU_CPU,
		(void **)&xferReqRingGpu,
		(void **)&xferReqRingCpu);
	if (result != DOCA_SUCCESS || xferReqRingGpu == NULL || xferReqRingCpu == NULL) {
		DOCA_LOG_ERR("Function doca_gpu_mem_alloc returned %s", doca_error_get_descr(result));
	}

	cudaMemset(xferReqRingGpu, 0, sizeof(struct docaXferReqGpu) * DOCA_XFER_REQ_MAX);

	// We may need a GPU warmup with relevant DOCA engine kernels
	doca_kernel_write(0, rdma_gpu, nullptr, 0);
	doca_kernel_read(0, rdma_gpu, nullptr, 0);

	xferRingPos = 0;
	firstXferRingPos = 0;
}

nixl_mem_list_t nixlDocaEngine::getSupportedMems () const {
	nixl_mem_list_t mems;
	mems.push_back(DRAM_SEG);
	mems.push_back(VRAM_SEG);
	return mems;
}

// Through parent destructor the unregister will be called.
nixlDocaEngine::~nixlDocaEngine () {
	doca_error_t result;

	// per registered memory deregisters it, which removes the corresponding metadata too
	// parent destructor takes care of the desc list
	// For remote metadata, they should be removed here
	if (this->initErr) {
		// Nothing to do
		return;
	}

	doca_gpu_mem_free(gdevs[0].second, xferReqRingGpu);

	result = doca_ctx_stop(rdma_ctx);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to stop RDMA context: %s", doca_error_get_descr(result));

	result = doca_rdma_destroy(rdma);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to destroy DOCA RDMA: %s", doca_error_get_descr(result));

	result = doca_dev_close(ddev);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to close DOCA device: %s", doca_error_get_descr(result));

	result = doca_gpu_destroy(gdevs[0].second);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to close DOCA GPU device: %s", doca_error_get_descr(result));
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlDocaEngine::checkConn(const std::string &remote_agent) {
	 if(remoteConnMap.find(remote_agent) == remoteConnMap.end()) {
		return NIXL_ERR_NOT_FOUND;
	}
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::endConn(const std::string &remote_agent) {
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::getConnInfo(std::string &str) const {
	str = nixlSerDes::_bytesToString(connection_details, conn_det_len);
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::connect(const std::string &remote_agent) {
	/* Already connected to remote QP at loadRemoteConnInfo time */
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::disconnect(const std::string &remote_agent) {
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::loadRemoteConnInfo (const std::string &remote_agent,
												 const std::string &remote_conn_info)
{
	doca_error_t result;
	nixlDocaConnection conn;
	size_t size = remote_conn_info.size();
	//TODO: eventually std::byte?
	char* addr = new char[size];

	std::cout << "loadRemoteConnInfo " << remote_agent << "\n";
	if(remoteConnMap.find(remote_agent) != remoteConnMap.end()) {
		return NIXL_ERR_INVALID_PARAM;
	}

	nixlSerDes::_stringToBytes((void*) addr, remote_conn_info, size);
	result = doca_rdma_connect(rdma, addr, size, connection);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_rdma_connect failed: %s", doca_error_get_descr(result));
		return NIXL_ERR_BACKEND;
	}

	conn.remoteAgent = remote_agent;
	conn.connected = true;

	std::cout << "Connected agent " << remote_agent << "\n";
	remoteConnMap[remote_agent] = conn;

	delete[] addr;

	return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/
nixl_status_t nixlDocaEngine::registerMem (const nixlBlobDesc &mem,
										  const nixl_mem_t &nixl_mem,
										  nixlBackendMD* &out)
{
	nixlDocaPrivateMetadata *priv = new nixlDocaPrivateMetadata;
	// uint64_t rkey_addr;
	// size_t rkey_size;
	uint32_t permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING;
	doca_error_t result;
	struct ibv_pd *pd;
	uint32_t mkey;

	auto it = std::find_if(gdevs.begin(), gdevs.end(),
							[&mem](std::pair<uint32_t, struct doca_gpu*> &x)
							{ return x.first == mem.devId; }
						);
	if (it == gdevs.end()) {
		std::cout << "Can't register memory for unknown device " << mem.devId << std::endl;
		return NIXL_ERR_INVALID_PARAM;
	}

	result = doca_mmap_create(&(priv->mem.mmap));
	if (result != DOCA_SUCCESS)
		return NIXL_ERR_BACKEND;

	// RELAXED ORDERING
	result = doca_mmap_set_permissions(priv->mem.mmap, permissions);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_mmap_set_memrange(priv->mem.mmap, (void*)mem.addr, (size_t)mem.len);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_mmap_add_dev(priv->mem.mmap, ddev);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_mmap_start(priv->mem.mmap);
	if (result != DOCA_SUCCESS)
		goto error;

	/* export mmap for rdma */
	result = doca_mmap_export_rdma(priv->mem.mmap,
						ddev,
						(const void **)&(priv->mem.export_mmap),
						&(priv->mem.export_len));
	if (result != DOCA_SUCCESS)
		goto error;

	priv->mem.addr = (void*)mem.addr;
	priv->mem.len = mem.len;
	priv->remoteMmapStr = nixlSerDes::_bytesToString((void*) priv->mem.export_mmap, priv->mem.export_len);

	/* Local buffer array */
	result = doca_buf_arr_create(1, &(priv->mem.barr));
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_buf_arr_set_params(priv->mem.barr, priv->mem.mmap, (size_t)mem.len, 0);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_buf_arr_set_target_gpu(priv->mem.barr, gdevs[mem.devId].second);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_buf_arr_start(priv->mem.barr);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_buf_arr_get_gpu_handle(priv->mem.barr, &(priv->mem.barr_gpu));
	if (result != DOCA_SUCCESS)
		goto error;

	doca_rdma_bridge_get_dev_pd(ddev, &pd);
	doca_rdma_bridge_get_mmap_mkey_from_pd(priv->mem.mmap, pd, &mkey);

	// printf("Local MMAP addr %p size %d mkey %x\n", priv->mem.addr, (int)mem.len, mkey);
	out = (nixlBackendMD*) priv; //typecast?

	return NIXL_SUCCESS;

error:
	if (priv->mem.barr)
		doca_buf_arr_destroy(priv->mem.barr);

	if (priv->mem.mmap)
		doca_mmap_destroy(priv->mem.mmap);

	return NIXL_ERR_BACKEND;
}

nixl_status_t nixlDocaEngine::deregisterMem(nixlBackendMD* meta)
{
	doca_error_t result;
	nixlDocaPrivateMetadata *priv = (nixlDocaPrivateMetadata*) meta;

	result = doca_mmap_destroy(priv->mem.mmap);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to call doca_mmap_destroy: %s", doca_error_get_descr(result));

	delete priv;
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::getPublicData (const nixlBackendMD* meta,
											std::string &str) const {
	const nixlDocaPrivateMetadata *priv = (nixlDocaPrivateMetadata*) meta;
	str = priv->remoteMmapStr;
	return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::internalMDHelper(const nixl_blob_t &blob,
								 const std::string &agent,
								 nixlBackendMD* &output)
{
	doca_error_t result;
	nixlDocaConnection conn;
	nixlDocaPublicMetadata *md = new nixlDocaPublicMetadata;
	size_t size = blob.size();
	auto search = remoteConnMap.find(agent);

	if(search == remoteConnMap.end()) {
		//TODO: err: remote connection not found
		DOCA_LOG_ERR("err: remote connection not found");
		return NIXL_ERR_NOT_FOUND;
	}
	conn = (nixlDocaConnection) search->second;

	//directly copy underlying conn struct
	md->conn = conn;

	char *addr = new char[size];
	nixlSerDes::_stringToBytes(addr, blob, size);

	result = doca_mmap_create_from_export(NULL,
		addr,
		size,
		ddev,
		&md->mem.mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_mmap_create_from_export failed: %s", doca_error_get_descr(result));
		return NIXL_ERR_BACKEND;
	}

	/* Remote buffer array */
	result = doca_buf_arr_create(1, &(md->mem.barr));
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_buf_arr_set_params(md->mem.barr, md->mem.mmap, size, 0);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_buf_arr_set_target_gpu(md->mem.barr, gdevs[0].second);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_buf_arr_start(md->mem.barr);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_buf_arr_get_gpu_handle(md->mem.barr, &(md->mem.barr_gpu));
	if (result != DOCA_SUCCESS)
		goto error;

	output = (nixlBackendMD*) md;

	// printf("Remote MMAP created %p raddr %p size %zd\n", (void*)md->mem.mmap, (void*)addr, size);

	delete[] addr;

	return NIXL_SUCCESS;

error:
	if (md->mem.barr)
		doca_buf_arr_destroy(md->mem.barr);

	return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlDocaEngine::loadLocalMD (nixlBackendMD* input,
							nixlBackendMD* &output)
{
	/* supportsLocal == false. Should it be true? */
	// nixlDocaPrivateMetadata* input_md = (nixlDocaPrivateMetadata*) input;
	// return internalMDHelper(input_md->remoteMmapStr, localAgent, output);

	return NIXL_SUCCESS;
}

// To be cleaned up
nixl_status_t nixlDocaEngine::loadRemoteMD (const nixlBlobDesc &input,
										   const nixl_mem_t &nixl_mem,
										   const std::string &remote_agent,
										   nixlBackendMD* &output)
{
	return internalMDHelper(input.metaInfo, remote_agent, output);
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::unloadMD (nixlBackendMD* input) {
	return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/
nixl_status_t nixlDocaEngine::prepXfer (const nixl_xfer_op_t &operation,
									   const nixl_meta_dlist_t &local,
									   const nixl_meta_dlist_t &remote,
									   const std::string &remote_agent,
									   nixlBackendReqH* &handle,
									   const nixl_opt_b_args_t* opt_args)
{
	uint32_t pos;
	nixlDocaBckndReq *treq = new nixlDocaBckndReq;
	nixlDocaPrivateMetadata *lmd;
	nixlDocaPublicMetadata *rmd;
	uint32_t lcnt = (uint32_t)local.descCount();
	uint32_t rcnt = (uint32_t)remote.descCount();

	treq->stream = (cudaStream_t)opt_args->customParam;
	treq->devId = (uint32_t)opt_args->devId;

	auto it = std::find_if(gdevs.begin(), gdevs.end(),
							[&treq](std::pair<uint32_t, struct doca_gpu*> &x)
							{ return x.first == treq->devId; }
						);
	if (it == gdevs.end()) {
		std::cout << "Can't prepare transfer for unknown device " << treq->devId << std::endl;
		return NIXL_ERR_INVALID_PARAM;
	}

	if (lcnt != rcnt)
		return NIXL_ERR_INVALID_PARAM;

	if (lcnt == 0)
		return NIXL_ERR_INVALID_PARAM;

	treq->start_pos = (xferRingPos.fetch_add(1) & (DOCA_XFER_REQ_MAX - 1));
	pos = treq->start_pos;

	do {
		for (uint32_t idx = 0; idx < lcnt && idx < DOCA_XFER_REQ_SIZE; idx++) {
			size_t lsize = local[idx].len;
			size_t rsize = remote[idx].len;
			if (lsize != rsize)
				return NIXL_ERR_INVALID_PARAM;

			lmd = (nixlDocaPrivateMetadata*) local[idx].metadataP;
			rmd = (nixlDocaPublicMetadata*) remote[idx].metadataP;

			xferReqRingCpu[pos].larr[idx] = (uintptr_t)lmd->mem.barr_gpu;
			xferReqRingCpu[pos].rarr[idx] = (uintptr_t)rmd->mem.barr_gpu;
			xferReqRingCpu[pos].size[idx] = lsize;
			xferReqRingCpu[pos].num++;
		}

		if (lcnt > DOCA_XFER_REQ_SIZE) {
			lcnt -= DOCA_XFER_REQ_SIZE;
			pos = (xferRingPos.fetch_add(1) & (DOCA_XFER_REQ_MAX - 1));
		} else {
			lcnt = 0;
		}
	} while(lcnt > 0);

	treq->end_pos = xferRingPos;

	// Need also a stream warmup?
	// doca_kernel_write(treq->stream, rdma_gpu, nullptr, 0);

	handle = treq;

	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::postXfer (const nixl_xfer_op_t &operation,
									   const nixl_meta_dlist_t &local,
									   const nixl_meta_dlist_t &remote,
									   const std::string &remote_agent,
									   nixlBackendReqH* &handle,
									   const nixl_opt_b_args_t* opt_args)
{
	nixlDocaBckndReq *treq = (nixlDocaBckndReq *) handle;

	std::cout << "postXfer start " << treq->start_pos << " end " << treq->end_pos << "\n";

	for (uint32_t idx = treq->start_pos; idx < treq->end_pos; idx++) {
		switch (operation) {
			case NIXL_READ:
				std::cout << "READ KERNEL, pos " << idx << " num " << xferReqRingCpu[idx].num << "\n";
				doca_kernel_read(treq->stream, rdma_gpu, xferReqRingGpu, idx);
				break;
			case NIXL_WRITE:
				std::cout << "WRITE KERNEL, pos " << idx << " num " << xferReqRingCpu[idx].num << "\n";
				doca_kernel_write(treq->stream, rdma_gpu, xferReqRingGpu, idx);
				break;
			default:
				return NIXL_ERR_INVALID_PARAM;
		}
	}

	return NIXL_IN_PROG;
}

nixl_status_t nixlDocaEngine::checkXfer(nixlBackendReqH* handle)
{
	nixlDocaBckndReq *treq = (nixlDocaBckndReq *) handle;

	for (uint32_t idx = treq->start_pos; idx < treq->end_pos; idx++) {
		// printf("Checking position %d\n", idx);
		if (xferReqRingCpu[idx].num > 0 && xferReqRingCpu[idx].num < DOCA_XFER_REQ_SIZE)
			return NIXL_IN_PROG;
		if (xferReqRingCpu[idx].num > DOCA_XFER_REQ_SIZE)
			return NIXL_ERR_BACKEND;
	}
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::releaseReqH(nixlBackendReqH* handle)
{
	firstXferRingPos = xferRingPos & (DOCA_XFER_REQ_MAX - 1);

	return NIXL_SUCCESS;
}

int nixlDocaEngine::progress() {
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::getNotifs(notif_list_t &notif_list)
{
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::genNotif(const std::string &remote_agent, const std::string &msg)
{
	nixl_status_t ret = NIXL_SUCCESS;
	// nixlDocaReq req;

	// ret = notifSendPriv(remote_agent, msg, req);

	switch(ret) {
	case NIXL_IN_PROG:
		/* do not track the request */
		// uw->reqRelease(req);
	case NIXL_SUCCESS:
		break;
	default:
		/* error case */
		return ret;
	}
	return NIXL_SUCCESS;
}