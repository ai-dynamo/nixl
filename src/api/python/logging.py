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
import socket
from typing import Optional

_logging_configured = False


class HostnameFilter(logging.Filter):
    """Filter that adds hostname to log records."""

    def __init__(self):
        super().__init__()
        self.hostname = socket.gethostname()

    def filter(self, record):
        record.hostname = self.hostname
        return True


def setup_logging(
    config_file: Optional[str] = None, force_reconfigure: bool = False
) -> None:
    """
    Setup logging configuration from INI file with environment variable support.

    Args:
        config_file: Path to logging configuration file. If None, uses default location.
        force_reconfigure: If True, reconfigures logging even if already configured.
    """
    global _logging_configured

    if _logging_configured and not force_reconfigure:
        return

    # Determine config file path
    if config_file is None:
        # Find python_logging.ini in the same directory as this file (project root)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "python_logging.ini")
        config_file = os.path.abspath(config_file)

    # Load the INI configuration file
    logging.config.fileConfig(config_file, disable_existing_loggers=False)

    # Add hostname filter to the nixl logger and all its handlers
    nixl_logger = logging.getLogger("nixl")
    hostname_filter = HostnameFilter()
    nixl_logger.addFilter(hostname_filter)

    # Also add the filter to all existing handlers
    for handler in nixl_logger.handlers:
        handler.addFilter(hostname_filter)

    # Override log level from environment variable if set
    env_log_level = os.getenv("NIXL_LOG_LEVEL")
    if env_log_level:
        try:
            # Convert string to logging level
            numeric_level = getattr(logging, env_log_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {env_log_level}")

            # Set the level for the nixl logger
            nixl_logger = logging.getLogger("nixl")
            nixl_logger.setLevel(numeric_level)

            # Also update handler levels if they exist
            for handler in nixl_logger.handlers:
                if hasattr(handler, "setLevel"):
                    handler.setLevel(numeric_level)

            nixl_logger.info(
                "Log level set to %s from NIXL_LOG_LEVEL environment variable",
                env_log_level.upper(),
            )
        except (ValueError, AttributeError) as e:
            nixl_logger = logging.getLogger("nixl")
            nixl_logger.warning(
                "Invalid NIXL_LOG_LEVEL value '%s': %s. Using configuration default.",
                env_log_level,
                e,
            )

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a NIXL logger with the given name, ensuring logging is configured.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance under the nixl hierarchy
    """
    setup_logging()
    # Convert module name to nixl hierarchy
    # e.g., '_api' -> 'nixl.api', 'test_nixl_bindings' -> 'nixl.test_nixl_bindings'
    clean_name = name.lstrip("_")  # Remove leading underscores
    return logging.getLogger(f"nixl.{clean_name}")


# Automatically configure logging when this module is imported
setup_logging()
