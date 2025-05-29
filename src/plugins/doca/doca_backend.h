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
#ifndef __DOCA_BACKEND_H
#define __DOCA_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <sys/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_gpunetio.h>
#include <doca_rdma.h>
#include <doca_rdma_bridge.h>
#include <doca_mmap.h>
#include <doca_buf_array.h>

#include "nixl.h"
#include "backend/backend_engine.h"
#include "common/str_tools.h"

// Local includes
#include "common/nixl_time.h"
#include "common/list_elem.h"

#define DOCA_MAX_COMPLETION_INFLIGHT 64
#define DOCA_DEVINFO_IBDEV_NAME_SIZE 64
#define RDMA_RECV_QUEUE_SIZE 2048
#define RDMA_SEND_QUEUE_SIZE 2048
#define DOCA_POST_STREAM_NUM 8
#define DOCA_XFER_REQ_SIZE 512
#define DOCA_XFER_REQ_MAX 32
#define DOCA_XFER_REQ_MASK (DOCA_XFER_REQ_MAX - 1)
#define DOCA_ENG_MAX_CONN 20
#define DOCA_RDMA_CM_LOCAL_PORT_CLIENT 6543
#define DOCA_RDMA_CM_LOCAL_PORT_SERVER 6544
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DOCA_RDMA_SERVER_ADDR_LEN (MAX(MAX(DOCA_DEVINFO_IPV4_ADDR_SIZE, DOCA_DEVINFO_IPV6_ADDR_SIZE), DOCA_GID_BYTE_LENGTH))
#define DOCA_RDMA_SERVER_CONN_DELAY 500 //500us
//Pre-fill the whole recv queue with notif once
#define DOCA_MAX_NOTIF_INFLIGHT RDMA_RECV_QUEUE_SIZE
#define DOCA_MAX_NOTIF_MESSAGE_SIZE 4096
#define DOCA_NOTIF_NULL 0xFFFFFFFF
#define DOCA_MSG_TAG 0xFF

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile uint8_t *)&(x))
#endif

struct docaXferReqGpu {
	uint32_t id;
	uintptr_t larr[DOCA_XFER_REQ_SIZE];
	uintptr_t rarr[DOCA_XFER_REQ_SIZE];
	size_t size[DOCA_XFER_REQ_SIZE];
	uint16_t num;
	uint8_t in_use;
	uint32_t conn_idx;
	uint32_t has_notif_msg_idx;
	struct doca_gpu_buf_arr *notif_barr_gpu;
	uint32_t *last_rsvd;
	uint32_t *last_posted;
	nixl_xfer_op_t backendOp;           /* Needed only in case of GPU device transfer */
	struct doca_gpu_dev_rdma *rdma_gpu_data; /* DOCA RDMA instance GPU handler */
	struct doca_gpu_dev_rdma *rdma_gpu_notif; /* DOCA RDMA instance GPU handler */
};

struct nixlDocaMem {
	void *addr;
	uint32_t len;
	struct doca_mmap *mmap;
	void *export_mmap;
	size_t export_len;
	struct doca_buf_arr *barr;
	struct doca_gpu_buf_arr *barr_gpu;
	uint32_t devId;
};

struct nixlDocaNotif {
	uint32_t elems_num;
	uint32_t elems_size;
	uint8_t *send_addr;
	std::atomic<uint32_t> send_pi;
	struct doca_mmap *send_mmap;
	struct doca_buf_arr *send_barr;
	struct doca_gpu_buf_arr *send_barr_gpu;
	uint8_t *recv_addr;
	std::atomic<uint32_t> recv_pi;
	struct doca_mmap *recv_mmap;
	struct doca_buf_arr *recv_barr;
	struct doca_gpu_buf_arr *recv_barr_gpu;
};

struct docaXferCompletion {
	uint8_t completed;
	struct docaXferReqGpu *xferReqRingGpu;
};

struct docaNotifRecv {
	struct doca_gpu_dev_rdma *rdma_qp;
	struct doca_gpu_buf_arr *barr_gpu;
	int num_progress;
};

struct docaNotifSend {
	struct doca_gpu_dev_rdma *rdma_qp;
	struct doca_gpu_buf_arr *barr_gpu;
	int buf_idx;
};

class nixlDocaConnection : public nixlBackendConnMD {
	private:
		std::string remoteAgent;
		// rdma qp
		// nixlDocaEp ep;
		volatile bool connected;

	public:
		// Extra information required for UCX connections

	friend class nixlDocaEngine;
};

// A private metadata has to implement get, and has all the metadata
class nixlDocaPrivateMetadata : public nixlBackendMD {
	private:
		nixlDocaMem mem;
		nixl_blob_t remoteMmapStr;

	public:
		nixlDocaPrivateMetadata() : nixlBackendMD(true) {
		}

		~nixlDocaPrivateMetadata(){
		}

		std::string get() const {
			return remoteMmapStr;
		}

	friend class nixlDocaEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlDocaPublicMetadata : public nixlBackendMD {

	public:
		nixlDocaMem mem;
		nixlDocaConnection conn;

		nixlDocaPublicMetadata() : nixlBackendMD(false) {}

		~nixlDocaPublicMetadata(){
		}
};

struct nixlDocaRdmaQp {
	struct doca_dev *dev;	  /* DOCA device handler associated to queues */
	struct doca_gpu *gpu;	  /* DOCA device handler associated to queues */
	struct doca_rdma *rdma_data;		    /* DOCA RDMA instance */
	struct doca_gpu_dev_rdma *rdma_gpu_data; /* DOCA RDMA instance GPU handler */
	struct doca_ctx *rdma_ctx_data;	    /* DOCA context to be used with DOCA RDMA */
	const void *connection_details_data;	    /* Remote peer connection details */
	size_t conn_det_len_data;		    /* Remote peer connection details data length */
	struct doca_rdma_connection *connection_data; /* The RDMA_CM connection instance */

	struct doca_rdma *rdma_notif;		    /* DOCA RDMA instance */
	struct doca_gpu_dev_rdma *rdma_gpu_notif; /* DOCA RDMA instance GPU handler */
	struct doca_ctx *rdma_ctx_notif;	    /* DOCA context to be used with DOCA RDMA */
	const void *connection_details_notif;	    /* Remote peer connection details */
	size_t conn_det_len_notif;		    /* Remote peer connection details data length */
	struct doca_rdma_connection *connection_notif; /* The RDMA_CM connection instance */

};

class nixlDocaEngine : public nixlBackendEngine {
	private:
		struct doca_log_backend *sdk_log;
		std::string msg_tag = "DOCA";
		std::vector<struct nixlDocaRdmaQp> rdma_qp_v;

		uint32_t local_port;
		int noSyncIters;
		uint8_t ipv4_addr[4];
		std::thread pthr;
		uint32_t *last_flags;
		cudaStream_t post_stream[DOCA_POST_STREAM_NUM];
		cudaStream_t wait_stream;
		std::atomic<uint32_t> xferStream;
		std::atomic<uint32_t> lastPostedReq;

		struct docaXferReqGpu *xferReqRingGpu;
		struct docaXferReqGpu *xferReqRingCpu;
		std::atomic<uint32_t> xferRingPos;
		uint32_t firstXferRingPos;

		struct docaXferCompletion *completion_list_gpu;
		struct docaXferCompletion *completion_list_cpu;
		uint32_t *wait_exit_gpu;
		uint32_t *wait_exit_cpu;
		int oob_sock_client;
		struct docaNotifRecv *notif_fill_gpu;
		struct docaNotifRecv *notif_fill_cpu;
		struct docaNotifRecv *notif_progress_gpu;
		struct docaNotifRecv *notif_progress_cpu;

		struct docaNotifSend *notif_send_gpu;
		struct docaNotifSend *notif_send_cpu;

		// Map of agent name to saved nixlDocaConnection info
		std::unordered_map<std::string, nixlDocaConnection,
						   std::hash<std::string>, strEqual> remoteConnMap;
		
		std::unordered_map<std::string, struct nixlDocaRdmaQp *,
						   std::hash<std::string>, strEqual> qpMap;

		std::unordered_map<std::string, struct nixlDocaNotif *,
						   std::hash<std::string>, strEqual> notifMap;

		pthread_t server_thread_id;

		class nixlDocaBckndReq : public nixlLinkElem<nixlDocaBckndReq>, public nixlBackendReqH {
			private:
			public:
				cudaStream_t stream;
				uint32_t devId;
				uint32_t start_pos;
				uint32_t end_pos;
				uintptr_t backendHandleGpu;

				nixlDocaBckndReq() : nixlLinkElem(), nixlBackendReqH() {
				}

				~nixlDocaBckndReq() {
				}
		};

		// Request management
		static void _requestInit(void *request);
		static void _requestFini(void *request);
		void requestReset(nixlDocaBckndReq *req) {
			_requestInit((void *)req);
		}

		// Memory management helpers
		nixl_status_t internalMDHelper (const nixl_blob_t &blob,
										const std::string &agent,
										nixlBackendMD* &output);

		// Threading infrastructure
		//   TODO: move the thread management one outside of NIXL common infra
		// void progressFunc(void* arg);
		void progressThreadStart();
		void progressThreadStop();
		void progressThreadRestart();

		nixl_status_t connectClientRdmaQp(int oob_sock_client, const std::string &remote_agent);
		nixl_status_t nixlDocaDestroyNotif(struct doca_gpu *gpu, struct nixlDocaNotif *notif);

		std::mutex notifSendLock;
	public:
		CUcontext main_cuda_ctx;
		int oob_sock_server;
		std::mutex notifFillLock;
		std::mutex notifProgressLock;
		std::vector<std::pair<uint32_t, struct doca_gpu *>> gdevs; /* List of DOCA GPUNetIO device handlers */
		struct doca_dev *ddev;	  /* DOCA device handler associated to queues */
		nixl_status_t addRdmaQp(const std::string &remote_agent);
		nixl_status_t connectServerRdmaQp(int oob_sock_client, const std::string &remote_agent);
		nixl_status_t nixlDocaInitNotif(const std::string &remote_agent, struct doca_dev *dev, struct doca_gpu *gpu);
		
		volatile uint8_t pthrStop, pthrActive;
		nixlDocaEngine(const nixlBackendInitParams* init_params);
		~nixlDocaEngine();

		bool supportsRemote () const { return true; }
		bool supportsLocal () const { return false; }
		bool supportsNotif () const { return true; }
		bool supportsProgTh () const { return false; }

		nixl_mem_list_t getSupportedMems () const;

		/* Object management */
		nixl_status_t getPublicData (const nixlBackendMD* meta,
									 std::string &str) const override;
		nixl_status_t getConnInfo(std::string &str) const override;
		nixl_status_t loadRemoteConnInfo (const std::string &remote_agent,
										  const std::string &remote_conn_info) override;

		nixl_status_t connect(const std::string &remote_agent) override;
		nixl_status_t disconnect(const std::string &remote_agent) override;

		nixl_status_t registerMem (const nixlBlobDesc &mem,
								   const nixl_mem_t &nixl_mem,
								   nixlBackendMD* &out) override;
		nixl_status_t deregisterMem (nixlBackendMD* meta) override;

		nixl_status_t loadLocalMD (nixlBackendMD* input,
								   nixlBackendMD* &output) override;

		nixl_status_t loadRemoteMD (const nixlBlobDesc &input,
									const nixl_mem_t &nixl_mem,
									const std::string &remote_agent,
									nixlBackendMD* &output) override;
		nixl_status_t unloadMD (nixlBackendMD* input) override;

		// Data transfer
		nixl_status_t prepXfer (const nixl_xfer_op_t &operation,
								const nixl_meta_dlist_t &local,
								const nixl_meta_dlist_t &remote,
								const std::string &remote_agent,
								nixlBackendReqH* &handle,
								const nixl_opt_b_args_t* opt_args=nullptr) const override;

		nixl_status_t postXfer (const nixl_xfer_op_t &operation,
								const nixl_meta_dlist_t &local,
								const nixl_meta_dlist_t &remote,
								const std::string &remote_agent,
								nixlBackendReqH* &handle,
								const nixl_opt_b_args_t* opt_args=nullptr) const override;

		nixl_status_t checkXfer (nixlBackendReqH* handle) const override;
		nixl_status_t releaseReqH(nixlBackendReqH* handle) const override;
		int progress();

		nixl_status_t getNotifs(notif_list_t &notif_list);
		nixl_status_t genNotif(const std::string &remote_agent, const std::string &msg) const override;

		void addConnection(struct doca_rdma_connection *connection);
		uint32_t getConnectionLast();
		void removeConnection(uint32_t connection_idx);
		uint32_t getGpuCudaId();

		nixl_status_t sendLocalAgent(int oob_sock_client);
		nixl_status_t recvLocalAgent(int oob_sock_client, std::string &remote_agent);

};

doca_error_t doca_util_map_and_export(struct doca_dev *dev, uint32_t permissions, void *addr, uint32_t size, nixlDocaMem *mem);

#if __cplusplus
extern "C" {
#endif

// prepXferGpu postXferGpuGet();
doca_error_t doca_kernel_write(cudaStream_t stream, struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos);
doca_error_t doca_kernel_read(cudaStream_t stream, struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos);
doca_error_t doca_kernel_progress(cudaStream_t stream, struct docaXferCompletion *completion_list,
								struct docaNotifRecv *notif_fill,
								struct docaNotifRecv *notif_progress,
								struct docaNotifSend *notif_send_gpu,
								uint32_t *exit_flag);

#if __cplusplus
}
#endif

#endif
