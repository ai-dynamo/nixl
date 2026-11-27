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

"""Backend-name handling of the Python wrapper, against mock bindings.

Names stay exact; a near miss in case or padding only gets a spelling hint.
Needs neither a built NIXL nor any plugin.
"""

import logging
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest


class BindingNotFoundError(RuntimeError):
    """Stand-in for nixlNotFoundError raised by the native createBackend."""


class BindingInvalidParamError(RuntimeError):
    """Stand-in for nixlInvalidParamError."""


@pytest.fixture(scope="module")
def api():
    """Import this checkout's wrapper without importing or replacing an installed nixl."""
    package = ModuleType("_nixl_backend_resolution_test")
    # Attributes the wrapper reads at import or call time are preset: a bare
    # Mock would hand back a Mock instead of an exception class.
    package._bindings = Mock(
        DEFAULT_COMM_PORT=0,
        nixlNotFoundError=BindingNotFoundError,
        nixlInvalidParamError=BindingInvalidParamError,
        nixlAgent=None,
    )
    path = Path(__file__).resolve().parents[2] / "src/api/python/_api.py"
    spec = spec_from_file_location(f"{package.__name__}._api", path)
    module = module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, package.__name__, package)
        patch.setitem(
            sys.modules,
            f"{package.__name__}.logging",
            SimpleNamespace(get_logger=logging.getLogger),
        )
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_agent(api, monkeypatch):
    def make(plugins, configured=(), num_threads=0):
        binding = Mock()
        binding.getAvailPlugins.return_value = list(plugins)

        def create_backend(name, params):
            if name not in plugins:
                raise BindingNotFoundError(name)
            return plugins[name]

        binding.createBackend.side_effect = create_backend
        binding.getBackendParams.return_value = ({"option": "1"}, ["DRAM"])
        binding.getPluginParams.return_value = ({"default": "0"}, ["DRAM"])
        monkeypatch.setattr(api.nixlBind, "nixlAgent", lambda name, config: binding)
        agent = api.nixl_agent(
            "local",
            api.nixl_agent_config(backends=list(configured), num_threads=num_threads),
        )
        monkeypatch.setattr(agent, "get_reg_descs", lambda descs, mem_type: descs)
        return agent

    return make


def warnings_of(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_configured_backend_keeps_exact_names_and_suggests_the_spelling(
    make_agent, caplog
):
    agent = make_agent(
        {"LIBFABRIC": 11, "UCX": 12}, ["libfabric", "UCX", "missing"], num_threads=2
    )
    agent.agent.createBackend.assert_called_once_with("UCX", {"num_threads": "2"})
    assert agent.backends == {"UCX": 12}
    assert warnings_of(caplog) == [
        "Skipping backend registration libfabric due to the missing plugin."
        " Did you mean 'LIBFABRIC'?",
        "Skipping backend registration missing due to the missing plugin.",
    ]


def test_create_backend_not_found_suggests_the_plugin_spelling(make_agent, caplog):
    agent = make_agent({"UCX": 11})
    with pytest.raises(BindingNotFoundError):
        agent.create_backend(" ucx ")
    agent.agent.createBackend.assert_called_once_with(" ucx ", {})
    assert agent.backends == {}
    assert warnings_of(caplog) == [
        "Backend plugin  ucx  not found. Did you mean 'UCX'?"
    ]

    caplog.clear()
    with pytest.raises(BindingNotFoundError):
        agent.create_backend("missing")
    assert warnings_of(caplog) == []


def test_plugin_and_backend_queries_suggest_the_spelling(make_agent, caplog):
    agent = make_agent({"UCX": 11, "ucx": 12, "POSIX": 13}, ["UCX"])
    assert agent.get_plugin_params("Ucx") == {}
    assert agent.get_plugin_mem_types("Ucx") == []
    assert agent.get_backend_params("ucx") == {}
    assert agent.get_backend_mem_types("ucx") == []
    agent.agent.getPluginParams.assert_not_called()
    assert warnings_of(caplog) == [
        "Plugin Ucx is not available to get its parameters."
        " Did you mean 'UCX' or 'ucx'?",
        "Plugin Ucx is not available to get its supported mem types."
        " Did you mean 'UCX' or 'ucx'?",
        "Backend ucx not instantiated to get its parameters. Did you mean 'UCX'?",
        "Backend ucx not instantiated to get its supported mem types."
        " Did you mean 'UCX'?",
    ]


@pytest.mark.parametrize(
    "use",
    [
        lambda agent, names: agent.register_memory("descs", backends=names),
        lambda agent, names: agent.deregister_memory("descs", backends=names),
        lambda agent, names: agent.make_connection("peer", backends=names),
        lambda agent, names: agent.get_new_notifs(backends=names),
    ],
)
def test_unknown_backend_name_is_a_readable_key_error(make_agent, use):
    agent = make_agent({"UCX": 11, "POSIX": 22}, ["UCX", "POSIX"])
    with pytest.raises(KeyError) as excinfo:
        use(agent, ["POSIX", "ucx"])
    assert str(excinfo.value) == (
        "Backend 'ucx' not found. Available backends: ['UCX', 'POSIX']."
        " Did you mean 'UCX'?"
    )
    with pytest.raises(KeyError) as excinfo:
        use(agent, ["missing"])
    assert str(excinfo.value).endswith("Available backends: ['UCX', 'POSIX'].")


def test_query_memory_suggests_the_spelling(make_agent):
    agent = make_agent({"UCX": 11}, ["UCX"])
    with pytest.raises(ValueError, match=r"not found.*Did you mean 'UCX'\?"):
        agent.query_memory("descs", "ucx")
    agent.agent.queryMem.assert_not_called()
    agent.query_memory("descs", "UCX")
    agent.agent.queryMem.assert_called_once_with("descs", 11)


def test_backend_lists_pass_each_handle_once_in_request_order(make_agent):
    agent = make_agent(
        {"UCX": 11, "POSIX": 22, "SECOND_NAME": 11}, ["UCX", "POSIX", "SECOND_NAME"]
    )
    agent.register_memory("descs", backends=["POSIX", "UCX", "POSIX", "SECOND_NAME"])
    agent.agent.registerMem.assert_called_once_with("descs", [22, 11])
    agent.deregister_memory("descs", backends=["UCX", "POSIX", "UCX"])
    agent.agent.deregisterMem.assert_called_once_with("descs", [11, 22])
    agent.make_connection("peer", backends=["UCX", "UCX"])
    agent.agent.makeConnection.assert_called_once_with("peer", [11])


def test_notification_passes_a_handle_list(make_agent):
    agent = make_agent({"UCX": 11}, ["UCX"])
    agent.send_notif("peer", b"hello")
    agent.send_notif("peer", b"hello", backend="UCX")
    assert agent.agent.genNotif.call_args_list == [
        call("peer", b"hello"),
        call("peer", b"hello", [11]),
    ]
