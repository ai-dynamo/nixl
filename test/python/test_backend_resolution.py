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
    package._bindings = Mock(
        DEFAULT_COMM_PORT=0, nixlInvalidParamError=BindingInvalidParamError
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
    def make(plugins, configured=()):
        binding = Mock()
        binding.getAvailPlugins.return_value = list(plugins)
        created = set()

        def create_backend(name, params):
            if name not in plugins:
                raise BindingNotFoundError(name)
            if name in created:
                raise BindingInvalidParamError(name)
            created.add(name)
            return plugins[name]

        binding.createBackend.side_effect = create_backend
        binding.getBackendParams.return_value = ({"option": "1"}, ["DRAM"])
        binding.getPluginParams.return_value = ({"default": "0"}, ["DRAM"])
        monkeypatch.setattr(api.nixlBind, "nixlAgent", lambda name, config: binding)
        agent = api.nixl_agent(
            "local", api.nixl_agent_config(backends=list(configured))
        )
        monkeypatch.setattr(agent, "get_reg_descs", lambda descs, mem_type: descs)
        return agent

    return make


@pytest.mark.parametrize("name", ["ucx", "UCX", " \tUcX\r\n"])
def test_create_and_query_use_the_canonical_backend(make_agent, name):
    agent = make_agent({"UCX": 11})
    agent.create_backend(name)
    agent.agent.createBackend.assert_called_once_with("UCX", {})
    with pytest.raises(BindingInvalidParamError):
        agent.create_backend("UCX")
    assert agent.backends == {"UCX": 11}
    assert agent.get_backend_params(name) == {"option": "1"}
    assert agent.get_backend_mem_types(name) == ["DRAM"]
    assert agent.get_plugin_params(name) == {"default": "0"}
    assert agent.get_plugin_mem_types(name) == ["DRAM"]
    agent.agent.getPluginParams.assert_called_once_with("UCX")
    agent.query_memory("descs", name)
    agent.agent.queryMem.assert_called_once_with("descs", 11)


def test_backend_lists_register_each_handle_once_in_request_order(make_agent):
    agent = make_agent(
        {"UCX": 11, "POSIX": 22, "SECOND_NAME": 11},
        ["ucx", " UCX ", "POSIX", "SECOND_NAME"],
    )
    assert agent.agent.createBackend.call_args_list == [
        call("UCX", {}),
        call("POSIX", {}),
        call("SECOND_NAME", {}),
    ]
    assert (
        agent.register_memory(
            "descs", backends=[" posix ", "ucx", " UCX ", "SECOND_NAME"]
        )
        == "descs"
    )
    agent.agent.registerMem.assert_called_once_with("descs", [22, 11])
    agent.deregister_memory("descs", backends=["UCX", "posix", " ucx "])
    agent.agent.deregisterMem.assert_called_once_with("descs", [11, 22])


@pytest.mark.parametrize("first, second", [("UCX", "ucx"), ("ucx", "UCX")])
def test_ambiguous_even_if_only_one_spelling_is_created(make_agent, first, second):
    agent = make_agent({first: 11, second: 22}, [first])
    error = "Ambiguous backend name 'Ucx': UCX, ucx"
    with pytest.raises(ValueError, match=error):
        agent.create_backend(" Ucx ")
    with pytest.raises(ValueError, match=error):
        agent.register_memory("descs", backends=[first, "Ucx"])
    agent.agent.registerMem.assert_not_called()
    agent.agent.createBackend.assert_called_once_with(first, {})
    with pytest.raises(KeyError):
        agent.register_memory("descs", backends=[second])

    agent.create_backend(" " + second)
    agent.register_memory("descs", backends=[second, first])
    agent.agent.registerMem.assert_called_once_with("descs", [22, 11])
    with pytest.raises(ValueError, match=error):
        agent.query_memory("descs", "Ucx")
    agent.agent.queryMem.assert_not_called()


def test_newly_available_spelling_is_considered_before_an_alias_is_used(make_agent):
    plugins = {"UCX": 11}
    agent = make_agent(plugins, ["UCX"])
    plugins["ucx"] = 22
    agent.agent.getAvailPlugins.return_value = list(plugins)
    with pytest.raises(ValueError, match="UCX, ucx"):
        agent.query_memory("descs", "Ucx")
    agent.create_backend(" ucx ")
    assert agent.backends == {"UCX": 11, "ucx": 22}


@pytest.mark.parametrize("unknown", ["missing", "ß", "\u00a0UCX\u00a0"])
def test_not_found_and_ascii_only_matching(make_agent, unknown):
    agent = make_agent({"UCX": 11, "SS": 22}, ["UCX", "SS"])
    with pytest.raises(BindingNotFoundError):
        agent.create_backend(" " + unknown + " ")
    agent.agent.createBackend.assert_called_with(unknown, {})
    assert agent.get_plugin_params(unknown) == {}
    assert agent.get_backend_mem_types(unknown) == []
    with pytest.raises(KeyError):
        agent.register_memory("descs", backends=["UCX", unknown])
    with pytest.raises(ValueError, match="not found"):
        agent.query_memory("descs", unknown)
    agent.agent.registerMem.assert_not_called()
    agent.agent.queryMem.assert_not_called()


@pytest.mark.parametrize("name", ["", " \t\r\n\f\v"])
def test_empty_name_is_invalid(make_agent, name):
    agent = make_agent({"UCX": 11})
    with pytest.raises(BindingInvalidParamError, match="Empty backend name"):
        agent.create_backend(name)
    agent.agent.createBackend.assert_not_called()
