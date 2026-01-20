#!/usr/bin/env python3
"""Plot timeline of TP execution phases from kvbench JSON results."""

import argparse
import json
import yaml
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_data(json_path: str, yaml_path: str = None):
    """Load JSON results and optionally YAML config for TP metadata."""
    with open(json_path) as f:
        data = json.load(f)
    
    config = None
    if yaml_path:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
    
    return data, config


def get_tp_node(tp_config, gpus_per_node=8):
    """Get node ID from TP config based on which ranks have storage ops."""
    read_sizes = tp_config.get('storage', {}).get('read', [])
    for i, size in enumerate(read_sizes):
        if size != 0 and size != '0':
            return i // gpus_per_node
    return -1


def plot_timeline(json_path: str, yaml_path: str = None, output_path: str = None, 
                  iteration: int = 0, show_isolated: bool = True):
    """Plot timeline showing when each TP runs."""
    data, config = load_data(json_path, yaml_path)
    
    # Get iteration results
    if iteration >= len(data['iterations_results']):
        print(f"Error: iteration {iteration} not found (max: {len(data['iterations_results'])-1})")
        return
    
    iter_results = data['iterations_results'][iteration]
    num_tps = len(iter_results)
    
    # Extract timing data
    tp_timings = []
    base_ts = None
    
    for i, tp in enumerate(iter_results):
        start = tp.get('storage_read_start_ts') or tp.get('min_start_ts')
        end = tp.get('storage_read_end_ts') or tp.get('max_end_ts')
        
        if start and end:
            if base_ts is None:
                base_ts = start
            tp_timings.append({
                'tp': i,
                'start': start - base_ts,
                'end': end - base_ts,
                'duration_ms': (end - start) * 1000,
                'size_gb': tp.get('storage_read_size_gb', tp.get('size', 0)),
                'iso_p50_ms': tp.get('isolated_read_p50_ms', tp.get('isolated_rdma_p50_ms', 0)),
            })
    
    if not tp_timings:
        print("No timing data found in results")
        return
    
    # Get node mapping from YAML if available
    tp_nodes = {}
    if config:
        for i, tp_cfg in enumerate(config.get('traffic_patterns', [])):
            tp_nodes[i] = get_tp_node(tp_cfg)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    
    # Color palette by node - use tab20 for more distinct colors
    node_colors = plt.cm.tab20.colors[:12]  # 12 distinct colors
    
    # Plot 1: Gantt chart of TP execution
    ax1.set_title(f'TP Execution Timeline (Iteration {iteration})', fontsize=14, fontweight='bold')
    
    for timing in tp_timings:
        tp_idx = timing['tp']
        node = tp_nodes.get(tp_idx, tp_idx // 2 % 12)
        color = node_colors[node % len(node_colors)]
        
        # Main bar
        bar = ax1.barh(tp_idx, timing['end'] - timing['start'], 
                       left=timing['start'], height=0.7,
                       color=color, edgecolor='black', linewidth=0.5,
                       label=f'Node {node}' if tp_idx == 0 else '')
        
        # Add duration text
        if timing['duration_ms'] > 50:  # Only show for visible bars
            ax1.text(timing['start'] + (timing['end'] - timing['start'])/2, tp_idx,
                    f'{timing["duration_ms"]:.0f}ms', 
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Traffic Pattern Index', fontsize=12)
    ax1.set_yticks(range(0, num_tps, max(1, num_tps // 20)))
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()  # TP 0 at top
    
    # Add legend for nodes
    legend_handles = [mpatches.Patch(color=node_colors[i], label=f'Node {i}') 
                      for i in range(min(12, max(tp_nodes.values()) + 1 if tp_nodes else 4))]
    ax1.legend(handles=legend_handles, loc='upper right', ncol=4, fontsize=8)
    
    # Plot 2: Timeline density / overlap analysis
    ax2.set_title('Concurrent TPs Over Time', fontsize=12)
    
    # Create time bins
    max_time = max(t['end'] for t in tp_timings)
    bins = np.linspace(0, max_time, 200)
    concurrent = np.zeros(len(bins) - 1)
    
    for timing in tp_timings:
        for j in range(len(bins) - 1):
            if timing['start'] < bins[j+1] and timing['end'] > bins[j]:
                concurrent[j] += 1
    
    ax2.fill_between(bins[:-1], concurrent, alpha=0.7, color='steelblue')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Sequential (no overlap)')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('# Concurrent TPs', fontsize=12)
    ax2.set_ylim(0, max(concurrent) * 1.1 + 0.5)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved timeline to {output_path}")
    else:
        plt.show()
    
    # Print summary
    print("\n=== TIMELINE SUMMARY ===")
    total_time = max(t['end'] for t in tp_timings)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Number of TPs: {len(tp_timings)}")
    print(f"Max concurrent TPs: {int(max(concurrent))}")
    
    if max(concurrent) <= 1:
        print("\n⚠️  TPs run SEQUENTIALLY (no overlap)")
        print("   This means earlier TPs (Node 0) get exclusive Lustre access!")
    else:
        print(f"\n✓ TPs overlap (up to {int(max(concurrent))} concurrent)")
    
    # Per-node timing
    if tp_nodes:
        print("\n=== PER-NODE TIMING ===")
        node_times = {}
        for timing in tp_timings:
            node = tp_nodes.get(timing['tp'], -1)
            if node not in node_times:
                node_times[node] = {'first_start': timing['start'], 'last_end': timing['end'], 'tps': 0}
            node_times[node]['first_start'] = min(node_times[node]['first_start'], timing['start'])
            node_times[node]['last_end'] = max(node_times[node]['last_end'], timing['end'])
            node_times[node]['tps'] += 1
        
        print(f"{'Node':<6} {'First Start':<12} {'Last End':<12} {'# TPs':<8}")
        print("-" * 40)
        for node in sorted(node_times.keys()):
            nt = node_times[node]
            print(f"{node:<6} +{nt['first_start']:>8.2f}s   +{nt['last_end']:>8.2f}s   {nt['tps']:<8}")


def plot_isolated_timeline(json_path: str, yaml_path: str = None, output_path: str = None):
    """Plot isolated benchmark execution order (from logs if available)."""
    # This would require parsing logs - for now, just show the expected order
    data, config = load_data(json_path, yaml_path)
    
    if not config:
        print("YAML config required for isolated timeline")
        return
    
    # Build TP to node mapping
    tp_nodes = []
    for tp_cfg in config.get('traffic_patterns', []):
        tp_nodes.append(get_tp_node(tp_cfg))
    
    # Show original vs interleaved order
    print("=== TP EXECUTION ORDER COMPARISON ===")
    print("\nOriginal Order (by TP index):")
    print("  " + " → ".join(f"TP{i}(N{tp_nodes[i]})" for i in range(min(8, len(tp_nodes)))) + " ...")
    
    # Simulate interleaved order
    from collections import defaultdict
    node_to_tps = defaultdict(list)
    for tp_ix, node in enumerate(tp_nodes):
        node_to_tps[node].append(tp_ix)
    
    interleaved = []
    max_tps = max(len(tps) for tps in node_to_tps.values()) if node_to_tps else 0
    for i in range(max_tps):
        for node in sorted(node_to_tps.keys()):
            if i < len(node_to_tps[node]):
                interleaved.append(node_to_tps[node][i])
    
    print("\nInterleaved Order (round-robin by node):")
    print("  " + " → ".join(f"TP{i}(N{tp_nodes[i]})" for i in interleaved[:8]) + " ...")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    
    node_colors = plt.cm.tab20.colors[:12]  # 12 distinct colors
    
    # Original order
    ax1.set_title('Original TP Order (Sequential by Node)', fontsize=12)
    for i, tp_idx in enumerate(range(min(30, len(tp_nodes)))):
        node = tp_nodes[tp_idx]
        ax1.barh(0, 1, left=i, height=0.8, color=node_colors[node], edgecolor='black', linewidth=0.3)
        if i < 16:
            ax1.text(i + 0.5, 0, f'{tp_idx}', ha='center', va='center', fontsize=7)
    ax1.set_xlim(0, min(30, len(tp_nodes)))
    ax1.set_yticks([])
    ax1.set_xlabel('Execution Order')
    
    # Interleaved order
    ax2.set_title('Interleaved TP Order (Round-Robin by Node)', fontsize=12)
    for i, tp_idx in enumerate(interleaved[:30]):
        node = tp_nodes[tp_idx]
        ax2.barh(0, 1, left=i, height=0.8, color=node_colors[node], edgecolor='black', linewidth=0.3)
        if i < 16:
            ax2.text(i + 0.5, 0, f'{tp_idx}', ha='center', va='center', fontsize=7)
    ax2.set_xlim(0, min(30, len(interleaved)))
    ax2.set_yticks([])
    ax2.set_xlabel('Execution Order')
    
    # Legend
    legend_handles = [mpatches.Patch(color=node_colors[i], label=f'Node {i}') 
                      for i in range(min(12, max(tp_nodes) + 1))]
    fig.legend(handles=legend_handles, loc='upper right', ncol=6, fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        out = output_path.replace('.png', '_order.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\nSaved order comparison to {out}")
    else:
        plt.show()


def print_text_gantt(json_path: str, iteration: int = 0):
    """Print ASCII Gantt chart of TP execution."""
    with open(json_path) as f:
        data = json.load(f)
    
    iter_results = data['iterations_results'][iteration]
    
    base_ts = None
    timings = []
    for i, tp in enumerate(iter_results):
        start = tp.get('storage_read_start_ts') or tp.get('min_start_ts')
        end = tp.get('storage_read_end_ts') or tp.get('max_end_ts')
        
        if start and end:
            if base_ts is None:
                base_ts = start
            timings.append((i, start - base_ts, end - base_ts))
    
    if not timings:
        print("No timing data found")
        return
    
    # Find overlapping pairs
    print("=" * 60)
    print("OVERLAPPING TP PAIRS")
    print("=" * 60)
    overlaps = []
    for i in range(len(timings)):
        for j in range(i + 1, len(timings)):
            tp1, s1, e1 = timings[i]
            tp2, s2, e2 = timings[j]
            if s1 < e2 and s2 < e1:
                overlap_dur = (min(e1, e2) - max(s1, s2)) * 1000
                overlaps.append((tp1, tp2, overlap_dur))
    
    overlaps.sort(key=lambda x: -x[2])
    for tp1, tp2, dur in overlaps[:10]:
        print(f"  TP {tp1:>2} & TP {tp2:>2}: {dur:>6.1f} ms overlap")
    print(f"\nTotal: {len(overlaps)} overlapping pairs")
    
    # Gantt chart
    print("\n" + "=" * 80)
    print("GANTT CHART (each character = 100ms)")
    print("=" * 80)
    
    max_time = max(t[2] for t in timings)
    for second in range(min(25, int(max_time) + 1)):
        line = f"{second:>2}s "
        for tenth in range(10):
            t = second + tenth * 0.1
            running = sum(1 for tp, s, e in timings if s <= t < e)
            if running == 0:
                line += "·"
            elif running == 1:
                line += "▪"
            elif running == 2:
                line += "▬"
            elif running == 3:
                line += "█"
            else:
                line += str(min(running, 9))
        print(line)
    
    print("\nLegend: · = 0 TPs, ▪ = 1 TP, ▬ = 2 TPs, █ = 3 TPs, 4+ = number")
    
    max_concurrent = 0
    for t_check in np.linspace(0, max_time, 1000):
        concurrent = sum(1 for tp, s, e in timings if s <= t_check < e)
        max_concurrent = max(max_concurrent, concurrent)
    
    print(f"\n{'✅ PARALLEL' if max_concurrent > 1 else '❌ SEQUENTIAL'}: Max {max_concurrent} TPs concurrent")


def main():
    parser = argparse.ArgumentParser(description='Plot TP execution timeline')
    parser.add_argument('json_file', help='Path to kvbench JSON results')
    parser.add_argument('--yaml', '-y', help='Path to YAML config for TP metadata')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--iteration', '-i', type=int, default=0, help='Iteration to plot')
    parser.add_argument('--order', action='store_true', help='Plot TP execution order comparison')
    parser.add_argument('--text', '-t', action='store_true', help='Print ASCII Gantt chart (no matplotlib)')
    
    args = parser.parse_args()
    
    if args.text:
        print_text_gantt(args.json_file, args.iteration)
    elif args.order:
        plot_isolated_timeline(args.json_file, args.yaml, args.output)
    else:
        plot_timeline(args.json_file, args.yaml, args.output, args.iteration)


if __name__ == '__main__':
    main()

