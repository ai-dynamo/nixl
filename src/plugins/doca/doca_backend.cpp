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
	std::vector<std::string> ndevs; /* Empty vector */
	std::vector<std::string> gdevs; /* Empty vector */
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

	if (custom_params->count("network_devices")!=0)
		ndevs = str_split((*custom_params)["network_devices"], " ");
	if (custom_params->count("gpu_devices")!=0)
		gdevs = str_split((*custom_params)["gpu_devices"], " ");

	std::cout << "DOCA network devices:" << std::endl;
	for (const std::string& str : ndevs) {
		std::cout << str << " ";
	}
	std::cout << std::endl;

	if (ndevs.size() > 1)
		throw std::invalid_argument("Only 1 network device is allowed");

	std::cout << "DOCA GPU devices:" << std::endl;
	for (const std::string& str : gdevs) {
		std::cout << str << " ";
	}
	std::cout << std::endl;

	if (gdevs.size() > 1)
		throw std::invalid_argument("Only 1 GPU device is allowed");

	/* Open DOCA device */
	result = open_doca_device_with_ibdev_name((const uint8_t *)(ndevs[0].c_str()),
						  ndevs[0].size(),
						  &(ddev));
	if (result != DOCA_SUCCESS) {
		throw std::invalid_argument("Failed to open DOCA device");
	}

	/* Create DOCA GPU */
	result = doca_gpu_create(gdevs[0].c_str(), &gdev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA GPU device: %s", doca_error_get_descr(result));
	}

	std::cout << "GPU " << gdev << "and NIC " << ddev << " created" << std::endl;

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
	result = doca_ctx_set_datapath_on_gpu(rdma_ctx, gdev);
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

	result = doca_ctx_stop(rdma_ctx);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to stop RDMA context: %s", doca_error_get_descr(result));

	result = doca_rdma_destroy(rdma);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to destroy DOCA RDMA: %s", doca_error_get_descr(result));

	result = doca_dev_close(ddev);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to close DOCA device: %s", doca_error_get_descr(result));

	result = doca_gpu_destroy(gdev);
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
	// str = nixlSerDes::_bytesToString(workerAddr, workerSize);
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::connect(const std::string &remote_agent) {
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::disconnect(const std::string &remote_agent) {
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::loadRemoteConnInfo (const std::string &remote_agent,
												 const std::string &remote_conn_info)
{
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
	doca_error_t result, result2;

	result = doca_mmap_create(&(priv->mem.mmap));
	if (result != DOCA_SUCCESS)
		return NIXL_ERR_BACKEND;

	// RELAXED ORDERING
	result = doca_mmap_set_permissions(priv->mem.mmap, permissions);
	if (result != DOCA_SUCCESS)
		goto error;

	result = doca_mmap_set_memrange(priv->mem.mmap, (void*)mem.addr, mem.len);
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

	out = (nixlBackendMD*) priv; //typecast?

	// TODO: Add nixl_mem check?
    // ret = uw->memReg((void*) mem.addr, mem.len, priv->mem);
    // if (ret) {
    //     return NIXL_ERR_BACKEND;
    // }
    // ret = uw->packRkey(priv->mem, rkey_addr, rkey_size);
    // if (ret) {
    //     return NIXL_ERR_BACKEND;
    // }
    // priv->rkeyStr = nixlSerDes::_bytesToString((void*) rkey_addr, rkey_size);

	return NIXL_SUCCESS;

error:
	result2 = doca_mmap_destroy(priv->mem.mmap);
	if (result2 != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to call doca_mmap_destroy: %s", doca_error_get_descr(result2));

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
	return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::loadLocalMD (nixlBackendMD* input,
							nixlBackendMD* &output)
{
	// nixlDocaPrivateMetadata* input_md = (nixlDocaPrivateMetadata*) input;
    // return internalMDHelper(input_md->rkeyStr, localAgent, output);

	return NIXL_SUCCESS;
}

// To be cleaned up
nixl_status_t nixlDocaEngine::loadRemoteMD (const nixlBlobDesc &input,
										   const nixl_mem_t &nixl_mem,
										   const std::string &remote_agent,
										   nixlBackendMD* &output)
{
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::unloadMD (nixlBackendMD* input) {
	return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

nixl_status_t nixlDocaEngine::retHelper(nixl_status_t ret, nixlDocaBckndReq *head, nixlDocaReq &req)
{
	/* if transfer wasn't immediately completed */
	switch(ret) {
		case NIXL_IN_PROG:
			// head->link((nixlDocaBckndReq*)req);
			break;
		case NIXL_SUCCESS:
			// Nothing to do
			break;
		default:
			// Error. Release all previously initiated ops and exit:
			// if (head->next()) {
			//     releaseReqH(head->next());
			// }
			return NIXL_ERR_BACKEND;
	}
	return NIXL_SUCCESS;
}

//Accept cudaStream_t as input
nixl_status_t nixlDocaEngine::prepXfer (const nixl_xfer_op_t &operation,
									   const nixl_meta_dlist_t &local,
									   const nixl_meta_dlist_t &remote,
									   const std::string &remote_agent,
									   nixlBackendReqH* &handle,
									   const nixl_opt_b_args_t* opt_args)
{
	// No preprations needed
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::postXfer (const nixl_xfer_op_t &operation,
									   const nixl_meta_dlist_t &local,
									   const nixl_meta_dlist_t &remote,
									   const std::string &remote_agent,
									   nixlBackendReqH* &handle,
									   const nixl_opt_b_args_t* opt_args)
{
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::checkXfer (nixlBackendReqH* handle)
{
	return NIXL_SUCCESS;
}

nixl_status_t nixlDocaEngine::releaseReqH(nixlBackendReqH* handle)
{
	return NIXL_SUCCESS;
}

int nixlDocaEngine::progress() {
	return NIXL_SUCCESS;
}

/****************************************
 * Notifications
*****************************************/

void nixlDocaEngine::notifCombineHelper(notif_list_t &src, notif_list_t &tgt)
{
}

void nixlDocaEngine::notifProgressCombineHelper(notif_list_t &src, notif_list_t &tgt)
{
}

void nixlDocaEngine::notifProgress()
{
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
