#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

extract_trap_cmd() {
  printf '%s\n' "${3:-}"
}

trap_add() {
  local trap_add_cmd="$1"

  shift || {
    echo "${FUNCNAME[0]} usage error" >&2
    return 1
  }
  if [ "$#" -eq 0 ]; then
    echo "${FUNCNAME[0]} usage error: at least one signal required" >&2
    return 1
  fi

  local trap_name existing_cmd
  for trap_name in "$@"; do
    if [ -z "${trap_name}" ]; then
      echo "${FUNCNAME[0]} usage error: empty signal name" >&2
      return 1
    fi

    existing_cmd=$(eval "extract_trap_cmd $(trap -p "${trap_name}")")
    if [ -n "${existing_cmd}" ]; then
      trap -- "${existing_cmd}; ${trap_add_cmd}" "${trap_name}"
    else
      trap -- "${trap_add_cmd}" "${trap_name}"
    fi
  done
}
