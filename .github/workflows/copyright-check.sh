#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

failures=()

for f in $(git ls-files); do
  # Skip non-source files
  case "$f" in
    *.png|*.jpg|*.jpeg|*.gif|*.ico|*.zip|*.rst|*.pyc|*.lock|LICENSE)
      continue
      ;;
  esac

  header=$(head -n 20 "$f")

  # Extract last modification year from git
  last_modified=$(git log -1 --pretty="%cs" -- "$f" | cut -d- -f1)

  # Extract copyright years (handles YYYY or YYYY-YYYY)
  copyright_years=$(echo "$header" | \
    grep -Eo 'Copyright \(c\) [0-9]{4}(-[0-9]{4})?' | \
    sed -E 's/.* ([0-9]{4})(-[0-9]{4})?/\1\2/' || true)

  if [[ -z "$copyright_years" ]]; then
    failures+=("$f (missing copyright)")
    continue
  fi

  # Get last year (handles range)
  end_year=$(echo "$copyright_years" | sed -E 's/.*-//' || true)

  # Validate date
  if (( end_year < last_modified )); then
    failures+=("$f (copyright year $end_year < last modified $last_modified)")
    continue
  fi

  # License line must exist
  if ! echo "$header" | grep -Eq '^[[:space:]]*(#|//|/\*|\*|<!--)?[[:space:]]*"?SPDX-License-Identifier:[[:space:]]*Apache-2\.0'; then
    failures+=("$f (missing license)")
    continue
  fi
done

if ((${#failures[@]} > 0)); then
  echo "❌ SPDX header check failed:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
else
  echo "✅ All SPDX headers valid"
fi
