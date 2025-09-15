#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

current_year=$(date +%Y)
failures=()

# Expected headers (you can extend this array if you want multiple copyright holders)
expected_copyright="SPDX-FileCopyrightText: Copyright (c) ${current_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved."
expected_license="SPDX-License-Identifier: Apache-2.0"

for f in $(git ls-files); do
  case "$f" in
    *.png|*.jpg|*.jpeg|*.gif|*.ico|*.zip|*.rst|*.pyc|*.lock|LICENSE|*.md|*.json)
      continue
      ;;
  esac

  header=$(head -n 20 "$f" | sed 's|^# *||; s|^// *||; s|^ \* *||; s|^<!-- *||; s|-->$||')

  # Verify at least one full copyright line matches
  if ! echo "$header" | grep -Fxq "$expected_copyright"; then
    failures+=("$f (copyright line missing or wrong)")
    continue
  fi

  # Verify license line matches exactly
  if ! echo "$header" | grep -Fxq "$expected_license"; then
    failures+=("$f (license line missing or wrong)")
    continue
  fi
done

if ((${#failures[@]} > 0)); then
  echo "❌ SPDX header check failed in:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
else
  echo "✅ All SPDX headers valid"
fi
