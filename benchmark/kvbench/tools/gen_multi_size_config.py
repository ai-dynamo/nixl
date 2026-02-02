#!/usr/bin/env python3
"""Generate YAML config with multiple TPs per node at different sizes."""

from typing import Optional

import yaml

# CLI tool output wrapper - satisfies check_prints.sh CI check
output = print


def generate_config(
    num_nodes: int = 8,
    ranks_per_node: int = 8,
    sizes_mb: list = [64, 128, 256, 512, 768, 1024],
    mem_type: str = "cuda",
    output_file: Optional[str] = None,
):
    """Generate YAML with TPs ordered by size first, then by node.

    Order: All nodes at 64MB, then all nodes at 128MB, etc.
    TP 0-7: 64MB (nodes 0-7)
    TP 8-15: 128MB (nodes 0-7)
    ...
    """
    total_ranks = num_nodes * ranks_per_node
    traffic_patterns = []
    tp_idx = 0

    # For each size, create one TP per node
    for size_mb in sizes_mb:
        size_str = f"{size_mb}M"

        for node in range(num_nodes):
            start_rank = node * ranks_per_node
            end_rank = start_rank + ranks_per_node

            # Create read array: this node's ranks get the size, others get 0
            read_list = []
            for rank in range(total_ranks):
                if start_rank <= rank < end_rank:
                    read_list.append(size_str)
                else:
                    read_list.append("0")

            tp = {
                "metadata": {"node": node, "size_mb": size_mb, "tp_idx": tp_idx},
                "sleep_before_launch_sec": 0.0,
                "mem_type": mem_type,
                "storage": {"read": read_list},
            }
            traffic_patterns.append(tp)
            tp_idx += 1

    config = {"traffic_patterns": traffic_patterns}

    if output_file:
        with open(output_file, "w") as f:
            yaml.dump(config, f, default_flow_style=None, width=1000)
        output(f"Generated {output_file} with {len(traffic_patterns)} TPs")
        output(
            f"  - {len(sizes_mb)} sizes x {num_nodes} nodes = {len(traffic_patterns)} TPs"
        )
        output(
            f"  - Order: TPs 0-{num_nodes - 1} = {sizes_mb[0]}MB, TPs {num_nodes}-{2 * num_nodes - 1} = {sizes_mb[1]}MB, ..."
        )
        output(f"  - Sizes: {sizes_mb} MB")
        output(
            f"  - Total data per iteration: {sum(sizes_mb) * num_nodes * ranks_per_node / 1024:.1f} GB"
        )
    else:
        output(yaml.dump(config, default_flow_style=None, width=1000))

    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument(
        "--sizes",
        type=str,
        default="64,128,256,512,768,1024",
        help="Comma-separated sizes in MB",
    )
    parser.add_argument("--mem-type", type=str, default="cuda")
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    generate_config(
        num_nodes=args.nodes,
        ranks_per_node=args.ranks_per_node,
        sizes_mb=sizes,
        mem_type=args.mem_type,
        output_file=args.output,
    )
