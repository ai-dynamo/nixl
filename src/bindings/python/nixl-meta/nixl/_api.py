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

# This file is a type stub for static analysis tools (pyright, mypy, IDEs).
# At runtime it is shadowed by the actual nixl_cu12._api or nixl_cu13._api
# module, which __init__.py injects into sys.modules["nixl._api"].
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from nixl_cu13._api import nixl_agent, nixl_agent_config  # type: ignore[import]  # noqa: F401
    except ImportError:
        from nixl_cu12._api import nixl_agent, nixl_agent_config  # type: ignore[import]  # noqa: F401
