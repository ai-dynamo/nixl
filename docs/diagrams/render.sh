#!/usr/bin/env bash
set -euo pipefail

# Check d2 availability (per D-16)
if ! command -v d2 &>/dev/null; then
  echo "Error: d2 CLI not found."
  echo "Install: https://d2lang.com/tour/install or brew install d2"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIGURES_DIR="$SCRIPT_DIR/../figures"

# Topic → list of diagram base names (without _light/_dark suffix)
declare -A TOPICS
TOPICS[data-flow]="nixl_flow_01_init nixl_flow_02_metadata nixl_flow_03_transfer nixl_flow_04_teardown nixl_flow_05_scaling nixl_desc_hierarchy"
TOPICS[basic-transfer]="nixl_basic_transfer_01_init nixl_basic_transfer_02_metadata nixl_basic_transfer_03_transfer nixl_basic_transfer_04_teardown"
TOPICS[gds-direct]="nixl_gds_direct_01_init nixl_gds_direct_02_write nixl_gds_direct_03_read nixl_gds_direct_04_verify"
TOPICS[etcd-metadata]="nixl_etcd_metadata_01_init nixl_etcd_metadata_02_publish nixl_etcd_metadata_03_fetch nixl_etcd_metadata_04_transfer nixl_etcd_metadata_05_invalidation"
TOPICS[remote-storage]="nixl_remote_storage_01_init nixl_remote_storage_02_metadata nixl_remote_storage_03_write nixl_remote_storage_04_read nixl_remote_storage_pipeline_read nixl_remote_storage_pipeline_write"

render_diagram() {
  local topic="$1"
  local name="$2"
  local src_dir="$SCRIPT_DIR/$topic"
  local out_dir="$FIGURES_DIR/$topic"
  mkdir -p "$out_dir"

  for variant in light dark; do
    local src="$src_dir/${name}_${variant}.d2"
    local out="$out_dir/${name}_${variant}.svg"
    if [[ ! -f "$src" ]]; then
      echo "Warning: $src not found, skipping."
      continue
    fi
    local theme=0
    [[ "$variant" == "dark" ]] && theme=200
    echo "Rendering: $src -> $out (theme $theme)"
    d2 --theme "$theme" --pad 20 "$src" "$out"
  done
}

render_topic() {
  local topic="$1"
  if [[ -z "${TOPICS[$topic]+x}" ]]; then
    echo "Error: unknown topic '$topic'"
    echo "Available topics: ${!TOPICS[*]}"
    exit 1
  fi
  for diagram in ${TOPICS[$topic]}; do
    render_diagram "$topic" "$diagram"
  done
}

if [[ $# -eq 0 ]]; then
  # Render all topics
  for topic in "${!TOPICS[@]}"; do
    render_topic "$topic"
  done
elif [[ $# -eq 1 ]]; then
  # Render one topic
  render_topic "$1"
elif [[ $# -eq 2 ]]; then
  # Render one diagram within a topic
  render_diagram "$1" "$2"
else
  echo "Usage: $0 [topic] [diagram_name]"
  echo "Topics: ${!TOPICS[*]}"
  exit 1
fi

echo "Done."
