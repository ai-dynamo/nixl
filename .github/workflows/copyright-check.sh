#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

failures=()

# Accepted copyright holders. Add new orgs here — each must appear in a header line as:
#   SPDX-FileCopyrightText: Copyright (c) YYYY(-YYYY) <AUTHOR>. All rights reserved.
AUTHORS=(
  "NVIDIA CORPORATION & AFFILIATES"
  "Advanced Micro Devices, Inc"
)

for f in $(git ls-files); do
  # Normalize path
  f=${f#./}

  # Skip ignored folders anywhere in path
  case "$f" in
    .github/*|.ci/*)
      continue
      ;;
  esac

  # Skip ignored top-level paths
  case "$f" in
    *.png|*.jpg|*.jpeg|*.gif|*.ico|*.zip|*.rst|*.pyc|*.lock|*.md|*.svg|*.wrap|*.in|*.json|*.template|*.patch|*.gitignore|*.python-version|*py.typed)
      continue
      ;;
    CODEOWNERS|*LICENSE*|Doxyfile|.clang-format|.clang-tidy|.codespellrc|.coderabbit.yaml)
      continue
      ;;
  esac

  header=$(head -n 20 "$f")

  # Match SPDX-FileCopyrightText with year(s)
  copyright_lines=$(echo "$header" | grep -E 'SPDX-FileCopyrightText:\s*Copyright \(c\) [0-9]{4}(-[0-9]{4})? .+\. All rights reserved\.' || true)

  # Keep only lines whose author field exactly matches an entry in AUTHORS.
  # The author field is the text between the year and ". All rights reserved.";
  # exact comparison rejects any extra text before/after an accepted author.
  matched_lines=""
  while IFS= read -r line; do
    line_author=$(echo "$line" | sed -E 's/.*Copyright \(c\) [0-9]{4}(-[0-9]{4})? (.+)\. All rights reserved\..*/\2/')
    for author in "${AUTHORS[@]}"; do
      if [[ "$line_author" == "$author" ]]; then
        matched_lines+="$line"$'\n'
        break
      fi
    done
  done <<< "$copyright_lines"

  if [[ -z "$matched_lines" ]]; then
    failures+=("$f (missing or incorrect copyright line)")
    continue
  fi

  # Extract last modification year from git
  last_modified=$(git log -1 --pretty="%cs" -- "$f" | cut -d- -f1)

  # Extract copyright years (handles YYYY or YYYY-YYYY) from every matched line;
  # a file may carry more than one accepted author (e.g. a dual-attributed derivative).
  copyright_years=$(echo "$matched_lines" | \
    grep -Eo 'Copyright \(c\) [0-9]{4}(-[0-9]{4})?' | \
    sed -E 's/.* ([0-9]{4})(-[0-9]{4})?/\1\2/')

  # Get the latest end year across all matched authors (handles ranges)
  end_year=$(echo "$copyright_years" | sed -E 's/.*-//' | sort -n | tail -1)

  # Validate date
  if (( end_year < last_modified )); then
    failures+=("$f (copyright year $end_year < last modified $last_modified)")
    continue
  fi

  # License line must exist
  if ! echo "$header" | grep -Eq '^[[:space:]]*(#|//|\*|/\*|<!--)[[:space:]]*SPDX-License-Identifier:.*Apache-2\.0'; then
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
