# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Centralized logging configuration for NIXL.

This module provides a single point of configuration for logging across the entire NIXL project.
It supports:
1. Loading configuration from logging.ini file
2. Override log levels using NIXL_LOG_LEVEL environment variable
3. Fallback to sensible defaults if configuration files are missing

Usage:
    from nixl_logging import get_logger

    logger = get_logger(__name__)
    logger.info("This is a log message")

Or for backward compatibility:
    import nixl_logging
    import logging

    logger = logging.getLogger(__name__)
    logger.info("This is a log message")
"""

import logging
import logging.config
import os

_logging_configured = False


def set_log_level_by_env(nixl_logger: logging.Logger) -> None:
    # Override log level from environment variable if set
    env_log_level = os.getenv("NIXL_LOG_LEVEL")
    if env_log_level:
        try:
            # Convert string to logging level
            numeric_level = getattr(logging, env_log_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {env_log_level}")

            # Set the level for the nixl logger
            nixl_logger.setLevel(numeric_level)

            nixl_logger.info(
                "Log level set to %s from NIXL_LOG_LEVEL environment variable",
                env_log_level.upper(),
            )
        except (ValueError, AttributeError) as e:
            nixl_logger.warning(
                "Invalid NIXL_LOG_LEVEL value '%s': %s. Using configuration default.",
                env_log_level,
                e,
            )


def setup_logging() -> None:
    global _logging_configured

    if _logging_configured:
        return

    # Find python_logging.ini in the same directory as this file (project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, "python_logging.ini")
    config_file = os.path.abspath(config_file_path)

    logging.config.fileConfig(config_file, disable_existing_loggers=False)

    nixl_logger = logging.getLogger("nixl")
    set_log_level_by_env(nixl_logger)
    logging.raiseExceptions = True

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    # Convert module name to nixl hierarchy
    # e.g., '_api' -> 'nixl.api', 'test_nixl_bindings' -> 'nixl.test_nixl_bindings'
    clean_name = name.lstrip("_")  # Remove leading underscores
    return logging.getLogger(f"nixl.{clean_name}")
