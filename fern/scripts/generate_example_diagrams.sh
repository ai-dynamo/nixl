#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
shopt -s globstar nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$(cd "$SCRIPT_DIR/../docs/assets" && pwd)"
DIAGRAMS_DIR="$ASSETS_DIR/diagrams/examples"
THEMES_DIR="$ASSETS_DIR/diagrams/theme"
FIGURES_DIR="$ASSETS_DIR/figures/generated/examples"

for source in "$DIAGRAMS_DIR"/**/*.d2; do
  relative="${source#"$DIAGRAMS_DIR"/}"

  for variant in light dark; do
    theme=0
    [[ "$variant" == dark ]] && theme=200
    output="$FIGURES_DIR/${relative%.d2}_$variant.svg"

    mkdir -p "$(dirname "$output")"
    echo "Rendering: ${relative%.d2}_$variant.svg"
    printf '...@%s\n...@%s\n' "$THEMES_DIR/$variant.d2" "$source" |
      d2 --theme "$theme" --pad 10 - "$output"
  done
done
