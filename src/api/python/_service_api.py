# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-level Pythonic API for :class:`nixlServiceAgent`.

This module mirrors :mod:`._api` but backs every agent with the service-layer
``nixlServiceAgent`` instead of the base ``nixlAgent``.  The public interface
is intentionally identical.

Service-specific features (marshal modes, service memory pool, per-transfer
marshal opt-args) are available through additional config options and methods.
"""

from typing import Optional, Union

import numpy as np

from . import _bindings as nixlBind  # type: ignore
from . import _service_bindings as svcBind  # type: ignore
from ._api import (
    nixl_agent,
    nixl_agent_config,
    nixl_backend_handle,
    nixl_prepped_dlist_handle,
    nixl_xfer_handle,
)
from .logging import get_logger

logger = get_logger(__name__)

# Re-export marshal config types so callers never need _service_bindings.
nixlMarshalDirectConfig = svcBind.nixlMarshalDirectConfig
nixlMarshalStagingConfig = svcBind.nixlMarshalStagingConfig
nixlMarshalCompressConfig = svcBind.nixlMarshalCompressConfig

nixlMarshalDirectOptArgs = svcBind.nixlMarshalDirectOptArgs
nixlMarshalStagingOptArgs = svcBind.nixlMarshalStagingOptArgs
nixlMarshalCompressOptArgs = svcBind.nixlMarshalCompressOptArgs
nixl_marshal_compress_algo_t = svcBind.nixl_marshal_compress_algo_t

recommendServiceMemSize = svcBind.recommendServiceMemSize


class nixl_service_agent_config(nixl_agent_config):
    """Configuration for a service agent.

    Extends :class:`nixl_agent_config` with a ``mode`` field that selects the
    marshal pipeline (direct, staging, or compress).  All base-config
    parameters are forwarded unchanged.

    :param mode: Marshal-mode configuration object.  Defaults to
        :class:`nixlMarshalDirectConfig` (passthrough, no staging memory).
    """

    def __init__(
        self,
        mode=None,
        enable_prog_thread: bool = True,
        enable_listen_thread: bool = False,
        listen_port: int = 0,
        capture_telemetry: bool = False,
        num_threads: int = 0,
        backends: list[str] = ["UCX"],
    ):
        super().__init__(
            enable_prog_thread=enable_prog_thread,
            enable_listen_thread=enable_listen_thread,
            listen_port=listen_port,
            capture_telemetry=capture_telemetry,
            num_threads=num_threads,
            backends=backends,
        )
        self.mode = mode if mode is not None else svcBind.nixlMarshalDirectConfig()


class nixl_service_agent(nixl_agent):
    """Drop-in replacement for :class:`nixl_agent` backed by ``nixlServiceAgent``.

    The public interface is identical to :class:`nixl_agent`.
    Service-specific features are exposed through extra config options and a
    handful of additional methods.

    Because ``nixlServiceAgent`` IS-A ``nixlAgent`` in the C++ hierarchy, all
    inherited Python methods (``register_memory``, ``get_xfer_descs``,
    ``prep_xfer_dlist``, ``make_prepped_xfer``, ``transfer``,
    ``check_xfer_state``, etc.) dispatch to the correct service overloads
    automatically.
    """

    def __init__(
        self,
        agent_name: str,
        nixl_conf: Optional[nixl_service_agent_config] = None,
        instantiate_all: bool = False,
    ):
        if nixl_conf is not None and instantiate_all:
            instantiate_all = False
            logger.warning(
                "Ignoring instantiate_all based on the provided config in agent creation."
            )

        if nixl_conf is None:
            nixl_conf = nixl_service_agent_config()

        thread_config = (
            nixlBind.NIXL_THREAD_SYNC_STRICT
            if nixl_conf.enable_listen
            else nixlBind.NIXL_THREAD_SYNC_NONE
        )

        svc_config = svcBind.nixlServiceAgentConfig()
        svc_config.useProgThread = nixl_conf.enable_pthread
        svc_config.useListenThread = nixl_conf.enable_listen
        svc_config.listenPort = nixl_conf.port
        svc_config.syncMode = thread_config
        svc_config.pthrDelay = 0
        svc_config.lthrDelay = 100000
        svc_config.captureTelemetry = nixl_conf.capture_telemetry
        svc_config.mode = nixl_conf.mode

        # Create the service agent (IS-A nixlAgent).
        self.agent = svcBind.nixlServiceAgent(agent_name, svc_config)

        self.name = agent_name
        self._leaked_xfer_handles: list[int] = []
        self.notifs: dict[str, list[bytes]] = {}
        self.backends: dict[str, nixl_backend_handle] = {}
        self.backend_mems: dict[str, list[str]] = {}
        self.backend_options: dict[str, dict[str, str]] = {}

        self.plugin_list = self.agent.getAvailPlugins()
        if len(self.plugin_list) == 0:
            logger.error("No plugins available, cannot start transfers!")
            raise RuntimeError("No plugins available for NIXL, cannot start transfers!")

        self.plugin_b_options: dict[str, dict[str, str]] = {}
        self.plugin_mem_types: dict[str, list[str]] = {}
        for plugin in self.plugin_list:
            (backend_options, mem_types) = self.agent.getPluginParams(plugin)
            self.plugin_b_options[plugin] = backend_options
            self.plugin_mem_types[plugin] = mem_types

        if instantiate_all:
            nixl_conf.backends = self.plugin_list

        for bknd in nixl_conf.backends:
            if bknd not in self.plugin_list:
                logger.warning(
                    "Skipping backend registration %s due to the missing plugin.",
                    bknd,
                )
            else:
                init: dict[str, str] = {}
                if nixl_conf.num_threads > 0:
                    if bknd == "UCX" or bknd == "OBJ":
                        init["num_threads"] = str(nixl_conf.num_threads)
                    elif bknd == "GDS_MT":
                        init["thread_count"] = str(nixl_conf.num_threads)
                    elif bknd == "UCCL":
                        init["num_cpus"] = str(nixl_conf.num_threads)
                self.create_backend(bknd, init)

        self.nixl_mems = {
            "DRAM": nixlBind.DRAM_SEG,
            "VRAM": nixlBind.VRAM_SEG,
            "FILE": nixlBind.FILE_SEG,
            "BLOCK": nixlBind.BLK_SEG,
            "OBJ": nixlBind.OBJ_SEG,
            "cpu": nixlBind.DRAM_SEG,
            "cuda": nixlBind.VRAM_SEG,
        }
        self.nixl_ops = {
            "WRITE": nixlBind.NIXL_WRITE,
            "READ": nixlBind.NIXL_READ,
        }

        logger.info("Initialized NIXL service agent: %s", agent_name)

    # ------------------------------------------------------------------
    # Service-specific methods
    # ------------------------------------------------------------------

    def register_service_memory(
        self,
        reg_list,
        mem_type: Optional[str] = None,
        backends: list[str] = [],
    ) -> nixlBind.nixlRegDList:
        """Register a staging / compression memory pool with the service layer."""
        reg_descs = self.get_reg_descs(reg_list, mem_type)
        backends_list = [self.backends[b] for b in backends]
        self.agent.registerServiceMem(reg_descs, backends_list)
        return reg_descs

    def deregister_service_memory(
        self,
        dereg_list: nixlBind.nixlRegDList,
        backends: list[str] = [],
    ):
        """Deregister previously-registered service memory."""
        backends_list = [self.backends[b] for b in backends]
        self.agent.deregisterServiceMem(dereg_list, backends_list)

    # ------------------------------------------------------------------
    # Transfer overrides that forward marshal_opt_args
    # ------------------------------------------------------------------

    def initialize_xfer(
        self,
        operation: str,
        local_descs: nixlBind.nixlXferDList,
        remote_descs: nixlBind.nixlXferDList,
        remote_agent: str,
        notif_msg: bytes = b"",
        backends: list[str] = [],
        marshal_opt_args=None,
    ) -> nixl_xfer_handle:
        op = self.nixl_ops[operation]
        handle_list = [self.backends[b] for b in backends]
        handle = self.agent.createXferReq(
            op, local_descs, remote_descs, remote_agent, notif_msg,
            handle_list, marshal_opt_args,
        )
        return nixl_xfer_handle(self.agent, handle)

    def make_prepped_xfer(
        self,
        operation: str,
        local_xfer_side: nixl_prepped_dlist_handle,
        local_indices: Union[list[int], np.ndarray],
        remote_xfer_side: nixl_prepped_dlist_handle,
        remote_indices: Union[list[int], np.ndarray],
        notif_msg: bytes = b"",
        backends: list[str] = [],
        skip_desc_merge: bool = False,
        marshal_opt_args=None,
    ) -> nixl_xfer_handle:
        op = self.nixl_ops[operation]
        handle_list = [self.backends[b] for b in backends]
        handle = self.agent.makeXferReq(
            op,
            local_xfer_side._handle,
            local_indices,
            remote_xfer_side._handle,
            remote_indices,
            notif_msg,
            handle_list,
            skip_desc_merge,
            marshal_opt_args,
        )
        return nixl_xfer_handle(self.agent, handle)
