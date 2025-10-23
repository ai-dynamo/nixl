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

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simpleFormatter": {
            "format": "%(asctime)s NIXL %(levelname)-7s %(filename)s:%(lineno)d %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simpleFormatter",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "nixl": {"level": "INFO", "handlers": ["consoleHandler"], "propagate": False}
    },
}


def set_log_level_by_env(nixl_logger: logging.Logger) -> None:
    # Override log level from environment variable if set
    env_log_level = os.getenv("NIXL_LOG_LEVEL")
    if env_log_level:
        try:
            # Convert string to logging level
            numeric_level = {
                "TRACE": logging.DEBUG,
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARN": logging.WARNING,
                "ERROR": logging.ERROR,
                "FATAL": logging.CRITICAL,
            }.get(env_log_level.upper())
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

    # Use dictionary-based configuration
    logging.config.dictConfig(LOGGING_CONFIG)

    nixl_logger = logging.getLogger("nixl")
    set_log_level_by_env(nixl_logger)

    logging.raiseExceptions = os.getenv("NIXL_DEBUG_LOGGING", "").lower() in (
        "true",
        "1",
        "yes",
    )

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    # Convert module name to nixl hierarchy
    # e.g., '_api' -> 'nixl.api', 'test_nixl_bindings' -> 'nixl.test_nixl_bindings'
    clean_name = name.lstrip("_")  # Remove leading underscores
    return logging.getLogger(f"nixl.{clean_name}")
