# SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree WQS secondary tier for vLLM 0.27.1 KV offloading."""

import ctypes
from contextlib import ExitStack

from vllm.logger import init_logger
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.obj.manager import (
    ObjectStoreSecondaryTierManager,
    TransferEntry,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

logger = init_logger("vllm.wqs_offload")


class WqsAsyncLookupManager(AsyncLookupManager):
    """Resolve secondary-tier block existence through NIXL's WQS backend."""

    def __init__(self, tier):
        super().__init__(tier_type=tier.tier_type)
        self._tier = tier

    def batch_lookup(self, keys, req_context):
        """Return one existence result per requested block.

        Args:
            keys: vLLM block keys to resolve through the tier's file mapper.
            req_context: Request context supplied by vLLM's lookup scheduler.

        Returns:
            Boolean results in the same order as the input keys.
        """
        descriptors = [
            (0, 1, 0, self._tier._file_mapper.get_file_name(key)) for key in keys
        ]
        results = self._tier._agent.query_memory(
            descriptors, backend="WQS", mem_type="OBJ"
        )
        hits = [result is not None for result in results]
        logger.info("WQS_NIXL_LOOKUP keys=%d hits=%d", len(keys), sum(hits))
        return hits


class WqsSecondaryTierManager(ObjectStoreSecondaryTierManager):
    """Store complete CPU cache blocks using vLLM's object-tier transfer flow."""

    def __init__(
        self,
        offloading_spec,
        primary_kv_view,
        tier_type,
        backend_params=None,
        prefix="",
    ):
        from nixl._api import nixl_agent, nixl_agent_config

        SecondaryTierManager.__init__(self, offloading_spec, primary_kv_view, tier_type)
        self.locality = None
        self.events = None
        self._store_job_keys = {}
        self._job_info = {}
        self._transfers = {}
        self._pending_results = []
        self._primary_reg = None
        self._dram_prepped_handle = None
        self._block_size_bytes = 0
        self._next_obj_dev_id = 1
        self._file_mapper = FileMapper.from_offloading_spec(
            f"{prefix}/" if prefix else "",
            offloading_spec,
            parallel_agnostic=True,
        )
        params = {key: str(value) for key, value in (backend_params or {}).items()}
        config = nixl_agent_config(backends=[], enable_prog_thread=False)
        # Inherited transfer preparation addresses the local agent by this name.
        self._agent = nixl_agent("ObjAgent", config)
        self._agent.create_backend("WQS", params)

        base_addr = ctypes.addressof(ctypes.c_char.from_buffer(primary_kv_view))
        stride = primary_kv_view.strides[0]
        self._primary_reg = self._agent.register_memory(
            [(base_addr, primary_kv_view.nbytes, 0, "")], "DRAM"
        )
        self._block_size_bytes = stride
        self._dram_prepped_handle = self._agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            [
                (base_addr + index * stride, stride, 0)
                for index in range(len(primary_kv_view))
            ],
            "DRAM",
        )
        self._lookup_manager = WqsAsyncLookupManager(self)
        logger.info(
            "WQS NIXL secondary tier ready: blocks=%d block_bytes=%d",
            len(primary_kv_view),
            stride,
        )

    def _exists(self, obj_key):
        results = self._agent.query_memory(
            [(0, 1, 0, obj_key)], backend="WQS", mem_type="OBJ"
        )
        return results[0] is not None

    def _submit_transfer(self, job_id, block_ids, obj_keys, op):
        block_ids = list(block_ids)
        obj_keys = list(obj_keys)
        self._job_info[job_id] = (op, len(obj_keys))
        logger.info(
            "WQS_NIXL_SUBMIT job=%d op=%s keys=%d bytes=%d",
            job_id,
            op,
            len(obj_keys),
            len(obj_keys) * self._block_size_bytes,
        )
        try:
            with ExitStack() as cleanup:
                descriptors = [
                    (0, self._block_size_bytes, dev_id, key)
                    for dev_id, key in enumerate(obj_keys, self._next_obj_dev_id)
                ]
                self._next_obj_dev_id += len(descriptors)
                files = self._agent.register_memory(descriptors, "OBJ")
                if files is None:
                    raise RuntimeError("WQS object registration failed")
                cleanup.callback(self._agent.deregister_memory, files)
                objects = self._agent.prep_xfer_dlist("ObjAgent", files.trim())
                if not objects:
                    raise RuntimeError("WQS object transfer-list preparation failed")
                cleanup.callback(self._agent.release_dlist_handle, objects)
                transfer = self._agent.make_prepped_xfer(
                    op,
                    self._dram_prepped_handle,
                    block_ids,
                    objects,
                    list(range(len(descriptors))),
                )
                if not transfer:
                    raise RuntimeError("WQS transfer preparation failed")
                cleanup.callback(self._agent.release_xfer_handle, transfer)
                if self._agent.transfer(transfer) == "ERR":
                    raise RuntimeError("WQS transfer submission failed")
                self._transfers[job_id] = TransferEntry(transfer, files, objects)
                cleanup.pop_all()
        except Exception as error:
            logger.warning("WQS transfer job %d failed: %s", job_id, error)
            self._pending_results.append(JobResult(job_id=job_id, success=False))

    def get_finished_jobs(self):
        """Return completed vLLM transfer jobs and log their WQS outcomes."""
        results = super().get_finished_jobs()
        for result in results:
            op, count = self._job_info.pop(result.job_id)
            logger.info(
                "WQS_NIXL_DONE job=%d op=%s keys=%d bytes=%d success=%s",
                result.job_id,
                op,
                count,
                count * self._block_size_bytes,
                result.success,
            )
        return results


SecondaryTierFactory.register_tier("wqs", __name__, "WqsSecondaryTierManager")


class WqsOffloadingSpec(TieringOffloadingSpec):
    """Select vLLM's tiered offloading implementation with the WQS tier registered."""
