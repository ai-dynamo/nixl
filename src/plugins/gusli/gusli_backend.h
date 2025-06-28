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

#ifndef __GUSLI_BACKEND_H
#define __GUSLI_BACKEND_H
#include <nixl.h>
#include <nixl_types.h>
#include "backend/backend_engine.h"
#include "gusli_client_api.hpp"

static inline nixl_mem_list_t __getSupportedGusliMems(void) {
	return {BLK_SEG, DRAM_SEG};	// Transfer between RAM and BDEV
}

class nixlGusliEngine : public nixlBackendEngine {
 private:
	gusli::global_clnt_context::init_params gp;				// Library params
	gusli::global_clnt_context* lib;						// Library context
	struct bdev_refcount_t { 								// No support for open/close so use refcount
		gusli::bdev_info bi;
		int ref_count;
	};
	std::unordered_map<uint64_t, struct bdev_refcount_t> bdevs;	// Hash of open block devices
	nixl_status_t _open (uint64_t devId);						// open()/close() called implicitly by *registerMem()
	nixl_status_t _close(uint64_t devId);
	int32_t get_gid_of_bdev(uint64_t devId) const {
		const struct bdev_refcount_t& v = bdevs.find(devId)->second;	// Already open
		return v.bi.bdev_descriptor;
	}
 public:
	nixlGusliEngine(const nixlBackendInitParams* init_params);
	~nixlGusliEngine();
	bool supportsNotif( void) const { return false; }
	bool supportsRemote(void) const { return false; }
	bool supportsLocal( void) const { return true; }		// Nixel client abscures the server so it acts as local, without remote
	bool supportsProgTh(void) const { return false;	}
	nixl_mem_list_t getSupportedMems(void) const { return __getSupportedGusliMems(); }
	nixl_status_t connect(   const std::string &remote_agent) override { return NIXL_SUCCESS; }
	nixl_status_t disconnect(const std::string &remote_agent) override { return NIXL_SUCCESS; }
	nixl_status_t loadLocalMD(nixlBackendMD* in, nixlBackendMD* &out) { out = in; return NIXL_SUCCESS; }
	nixl_status_t unloadMD(nixlBackendMD* input) { return NIXL_SUCCESS; }
	nixl_status_t registerMem(const nixlBlobDesc &mem,
								const nixl_mem_t &nixl_mem,
								nixlBackendMD* &out);
	nixl_status_t deregisterMem(nixlBackendMD*  out);

	nixl_status_t prepXfer( const nixl_xfer_op_t &operation,
							const nixl_meta_dlist_t &local,
							const nixl_meta_dlist_t &remote,
							const std::string &remote_agent,
							nixlBackendReqH* &io_handle,
							const nixl_opt_b_args_t* opt_args=nullptr) const;

	nixl_status_t postXfer( const nixl_xfer_op_t &operation,
							const nixl_meta_dlist_t &local,
							const nixl_meta_dlist_t &remote,
							const std::string &remote_agent,
							nixlBackendReqH* &io_handle,
							const nixl_opt_b_args_t* opt_args=nullptr) const;
	nixl_status_t checkXfer(  nixlBackendReqH* io_handle) const;
	nixl_status_t releaseReqH(nixlBackendReqH* io_handle) const;
};
#endif
