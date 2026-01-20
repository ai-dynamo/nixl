#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Analyze kvbench benchmark results - detect performance patterns and anomalies.

Usage:
    python analyze_results.py <results.json> [--yaml <traffic_pattern.yaml>] [--csv <output.csv>]

Example:
    python analyze_results.py /path/to/results.json --yaml /path/to/metadata_storage_only.yaml
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def expand_nodelist(nodelist: str) -> List[str]:
    """Expand SLURM nodelist like 'hgx-isr1-[098,100,102-103]' to list of hostnames."""
    import subprocess
    try:
        result = subprocess.run(
            ['scontrol', 'show', 'hostnames', nodelist],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return [h.strip() for h in result.stdout.strip().split('\n') if h.strip()]
    except Exception:
        pass
    # Fallback: return empty list
    return []


def parse_size(size_str) -> int:
    """Parse size string like '144M' or '2396M' to bytes."""
    if isinstance(size_str, (int, float)):
        return int(size_str)
    if size_str == 0 or size_str == '0':
        return 0
    size_str = str(size_str).strip().upper()
    multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    for suffix, mult in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-1]) * mult)
    return int(size_str)


def get_tp_ranks(tp_config: dict) -> List[int]:
    """Get list of ranks that have storage operations in this TP."""
    ranks = []
    if 'storage' in tp_config:
        read_sizes = tp_config['storage'].get('read', [])
        write_sizes = tp_config['storage'].get('write', [])
        for i, (r, w) in enumerate(zip(
            read_sizes + [0] * max(0, len(write_sizes) - len(read_sizes)),
            write_sizes + [0] * max(0, len(read_sizes) - len(write_sizes))
        )):
            if parse_size(r) > 0 or parse_size(w) > 0:
                ranks.append(i)
        # Handle case where only read or write exists
        if not ranks:
            for i, size in enumerate(read_sizes):
                if parse_size(size) > 0:
                    ranks.append(i)
            for i, size in enumerate(write_sizes):
                if parse_size(size) > 0 and i not in ranks:
                    ranks.append(i)
    return ranks


def ranks_to_nodes(ranks: List[int], gpus_per_node: int = 8) -> List[int]:
    """Convert rank list to node list."""
    return sorted(set(r // gpus_per_node for r in ranks))


def ranks_to_numas(ranks: List[int], gpus_per_numa: int = 4) -> List[int]:
    """Convert rank list to NUMA domains (within node)."""
    return sorted(set((r % 8) // gpus_per_numa for r in ranks))


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def analyze_single_iteration(
    data: Dict,
    iteration: int,
    tp_configs: Optional[List] = None,
    gpus_per_node: int = 8,
) -> List[Dict]:
    """Analyze a single iteration and return list of TP results."""
    tps = data['iterations_results'][iteration]
    
    results = []
    for i, tp in enumerate(tps):
        # Get sizes and latencies
        read_size = tp.get('storage_read_size_gb', 0)
        write_size = tp.get('storage_write_size_gb', 0)
        rdma_size = tp.get('size', 0)
        
        read_p50 = tp.get('isolated_read_p50_ms', 0)
        read_p90 = tp.get('isolated_read_p90_ms', 0)
        write_p50 = tp.get('isolated_write_p50_ms', 0)
        rdma_p50 = tp.get('isolated_rdma_p50_ms', 0)
        
        workload_read = tp.get('storage_read_avg_ms', 0)
        workload_write = tp.get('storage_write_avg_ms', 0)
        workload_rdma = tp.get('latency', 0) or 0
        
        # Calculate BWs
        read_bw = (read_size / (read_p50 / 1000)) if read_p50 > 0 else 0
        write_bw = (write_size / (write_p50 / 1000)) if write_p50 > 0 else 0
        rdma_bw = (rdma_size / (rdma_p50 / 1000)) if rdma_p50 > 0 else 0
        
        # Get rank/node mapping if available
        ranks = []
        nodes = []
        numas = []
        if tp_configs and i < len(tp_configs):
            ranks = get_tp_ranks(tp_configs[i])
            nodes = ranks_to_nodes(ranks, gpus_per_node)
            numas = ranks_to_numas(ranks)
        
        results.append({
            'tp': i,
            'read_size_gb': read_size,
            'write_size_gb': write_size,
            'rdma_size_gb': rdma_size,
            'read_p50_ms': read_p50,
            'read_p90_ms': read_p90,
            'write_p50_ms': write_p50,
            'rdma_p50_ms': rdma_p50,
            'workload_read_ms': workload_read,
            'workload_write_ms': workload_write,
            'workload_rdma_ms': workload_rdma,
            'read_bw_gbs': read_bw,
            'write_bw_gbs': write_bw,
            'rdma_bw_gbs': rdma_bw,
            'ranks': ranks,
            'nodes': nodes,
            'numas': numas,
        })
    
    return results


def analyze_results(
    json_path: str,
    yaml_path: Optional[str] = None,
    gpus_per_node: int = 8,
    iteration: int = 0,
    nodelist: Optional[str] = None,
    all_iterations: bool = False,
) -> Dict:
    """
    Analyze benchmark results and return comprehensive analysis.
    
    Args:
        json_path: Path to results JSON file
        yaml_path: Optional path to traffic pattern YAML for rank mapping
        gpus_per_node: Number of GPUs per node (default 8)
        iteration: Which iteration to analyze (default 0 = first)
        all_iterations: If True, analyze all iterations and aggregate
    
    Returns:
        Dictionary with analysis results
    """
    with open(json_path) as f:
        data = json.load(f)
    
    # Load YAML if provided
    tp_configs = None
    if yaml_path and HAS_YAML:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
            tp_configs = yaml_data.get('traffic_patterns', [])
    
    num_iterations = len(data['iterations_results'])
    
    if all_iterations:
        # Analyze all iterations
        all_iter_results = []
        for iter_idx in range(num_iterations):
            iter_results = analyze_single_iteration(data, iter_idx, tp_configs, gpus_per_node)
            all_iter_results.append(iter_results)
        
        # Aggregate: for each TP, compute mean/std across iterations
        num_tps = len(all_iter_results[0])
        aggregated = []
        
        for tp_idx in range(num_tps):
            tp_data = [all_iter_results[i][tp_idx] for i in range(num_iterations)]
            
            # Aggregate numeric fields
            agg = {
                'tp': tp_idx,
                'read_size_gb': tp_data[0]['read_size_gb'],  # Same across iters
                'write_size_gb': tp_data[0]['write_size_gb'],
                'rdma_size_gb': tp_data[0]['rdma_size_gb'],
                'ranks': tp_data[0]['ranks'],
                'nodes': tp_data[0]['nodes'],
                'numas': tp_data[0]['numas'],
            }
            
            # Fields to average
            for field in ['read_p50_ms', 'read_p90_ms', 'write_p50_ms', 'rdma_p50_ms',
                          'workload_read_ms', 'workload_write_ms', 'workload_rdma_ms',
                          'read_bw_gbs', 'write_bw_gbs', 'rdma_bw_gbs']:
                values = [t[field] for t in tp_data if t[field] > 0]
                if values:
                    agg[field] = statistics.mean(values)
                    agg[f'{field}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                    agg[f'{field}_min'] = min(values)
                    agg[f'{field}_max'] = max(values)
                else:
                    agg[field] = 0.0
                    agg[f'{field}_std'] = 0.0
                    agg[f'{field}_min'] = 0.0
                    agg[f'{field}_max'] = 0.0
            
            aggregated.append(agg)
        
        tps = aggregated
    else:
        tps = analyze_single_iteration(data, iteration, tp_configs, gpus_per_node)
    
    # Build node ID to hostname mapping
    node_names = {}
    if nodelist:
        hostnames = expand_nodelist(nodelist)
        for i, hostname in enumerate(hostnames):
            node_names[i] = hostname
    
    return {
        'tps': tps,
        'metadata': data.get('metadata', {}),
        'json_path': json_path,
        'yaml_path': yaml_path,
        'node_names': node_names,
        'all_iterations': all_iterations,
        'num_iterations': num_iterations,
    }


def print_full_table(analysis: Dict, show_ranks: bool = False):
    """Print comprehensive analysis table."""
    tps = analysis['tps']
    
    print("=" * 120)
    print("FULL RESULTS TABLE")
    print("=" * 120)
    
    header = f"{'TP':<4} {'Size(GB)':<10} {'p50(ms)':<10} {'BW(GB/s)':<10} {'Wkld(ms)':<10} {'p90/p50':<8}"
    if show_ranks and tps and tps[0]['nodes']:
        header += f" {'Nodes':<12} {'Ranks':<6}"
    print(header)
    print("-" * 120)
    
    for t in sorted(tps, key=lambda x: (x['read_size_gb'], x['tp'])):
        if t['read_p50_ms'] <= 0:
            continue
        
        p90_ratio = t['read_p90_ms'] / t['read_p50_ms'] if t['read_p50_ms'] > 0 else 0
        
        row = f"{t['tp']:<4} {t['read_size_gb']:<10.3f} {t['read_p50_ms']:<10.1f} {t['read_bw_gbs']:<10.2f} {t['workload_read_ms']:<10.1f} {p90_ratio:<8.2f}"
        if show_ranks and t['nodes']:
            row += f" {str(t['nodes']):<12} {len(t['ranks']):<6}"
        print(row)
    
    print("-" * 120)


def print_node_analysis(analysis: Dict):
    """Print per-node performance breakdown."""
    tps = analysis['tps']
    node_names = analysis.get('node_names', {})
    
    # Group by node
    node_perf: Dict[int, List[float]] = {}
    for t in tps:
        if t['read_bw_gbs'] > 0:
            for node in t['nodes']:
                if node not in node_perf:
                    node_perf[node] = []
                node_perf[node].append(t['read_bw_gbs'])
    
    if not node_perf:
        return
    
    print("\n" + "=" * 90)
    print("PERFORMANCE BY NODE")
    print("=" * 90)
    
    if node_names:
        print(f"\n{'Node':<6} {'Hostname':<20} {'Avg BW':<10} {'Min':<10} {'Max':<10} {'Std':<10} {'TPs':<6}")
        print("-" * 90)
    else:
        print(f"\n{'Node':<6} {'Avg BW':<10} {'Min':<10} {'Max':<10} {'Std':<10} {'TPs':<6}")
        print("-" * 60)
    
    for node in sorted(node_perf.keys()):
        bws = node_perf[node]
        avg_bw = statistics.mean(bws)
        std_bw = statistics.stdev(bws) if len(bws) > 1 else 0
        
        hostname = node_names.get(node, "")
        if node_names:
            print(f"{node:<6} {hostname:<20} {avg_bw:<10.2f} {min(bws):<10.2f} {max(bws):<10.2f} {std_bw:<10.2f} {len(bws):<6}")
        else:
            print(f"{node:<6} {avg_bw:<10.2f} {min(bws):<10.2f} {max(bws):<10.2f} {std_bw:<10.2f} {len(bws):<6}")


def print_numa_analysis(analysis: Dict):
    """Print per-Node-NUMA performance breakdown."""
    tps = analysis['tps']
    
    # Group by (node, numa) tuple
    node_numa_perf: Dict[Tuple[int, int], List[float]] = {}
    for t in tps:
        if t['read_bw_gbs'] > 0 and t['nodes'] and t['numas']:
            for node in t['nodes']:
                for numa in t['numas']:
                    key = (node, numa)
                    if key not in node_numa_perf:
                        node_numa_perf[key] = []
                    node_numa_perf[key].append(t['read_bw_gbs'])
    
    if not node_numa_perf:
        return
    
    # Check if there's actually NUMA variation (skip if all TPs use both NUMAs)
    nodes = sorted(set(k[0] for k in node_numa_perf.keys()))
    has_numa_variation = False
    for node in nodes:
        numa0 = node_numa_perf.get((node, 0), [])
        numa1 = node_numa_perf.get((node, 1), [])
        if len(numa0) != len(numa1):  # Different number of TPs per NUMA = variation exists
            has_numa_variation = True
            break
    
    if not has_numa_variation:
        return  # Skip NUMA analysis if all TPs span both NUMAs
    
    print("\n" + "=" * 80)
    print("PERFORMANCE BY NODE + NUMA")
    print("=" * 80)
    
    print(f"\n{'Node':<6} {'NUMA':<6} {'Avg BW':<10} {'Min':<10} {'Max':<10} {'TPs':<6}")
    print("-" * 60)
    
    for (node, numa) in sorted(node_numa_perf.keys()):
        bws = node_numa_perf[(node, numa)]
        avg_bw = statistics.mean(bws)
        print(f"{node:<6} {numa:<6} {avg_bw:<10.2f} {min(bws):<10.2f} {max(bws):<10.2f} {len(bws):<6}")
    
    # Within-node comparison
    print("\n" + "-" * 60)
    print("NUMA0 vs NUMA1 per node:")
    print("-" * 60)
    
    for node in nodes:
        numa0_bws = node_numa_perf.get((node, 0), [])
        numa1_bws = node_numa_perf.get((node, 1), [])
        if numa0_bws and numa1_bws:
            avg0 = statistics.mean(numa0_bws)
            avg1 = statistics.mean(numa1_bws)
            diff = (avg1 - avg0) / avg0 * 100 if avg0 > 0 else 0
            print(f"  Node {node}: NUMA0={avg0:.1f} GB/s, NUMA1={avg1:.1f} GB/s, diff={diff:+.1f}%")
        elif numa0_bws:
            print(f"  Node {node}: NUMA0={statistics.mean(numa0_bws):.1f} GB/s, NUMA1=N/A")
        elif numa1_bws:
            print(f"  Node {node}: NUMA0=N/A, NUMA1={statistics.mean(numa1_bws):.1f} GB/s")




def print_outliers(analysis: Dict):
    """Print outlier detection - TPs with >20% deviation from median BW."""
    tps = analysis['tps']
    
    bws = [t['read_bw_gbs'] for t in tps if t['read_bw_gbs'] > 0]
    if not bws:
        return
    
    median_bw = statistics.median(bws)
    
    outliers = []
    for t in tps:
        if t['read_bw_gbs'] > 0:
            expected = (t['read_size_gb'] / median_bw) * 1000
            deviation = (t['read_p50_ms'] - expected) / expected * 100 if expected > 0 else 0
            if abs(deviation) > 20:
                outliers.append({
                    **t,
                    'expected_ms': expected,
                    'deviation_pct': deviation,
                })
    
    if not outliers:
        return  # Skip section if no outliers
    
    print("\n" + "=" * 90)
    print(f"OUTLIERS (>20% deviation from median BW={median_bw:.2f} GB/s)")
    print("=" * 90)
    
    print(f"\n{'TP':<4} {'Size(GB)':<10} {'Actual(ms)':<12} {'Expected(ms)':<12} {'Deviation':<12} {'BW(GB/s)':<10} {'Nodes'}")
    print("-" * 90)
    
    for o in sorted(outliers, key=lambda x: x['deviation_pct']):
        print(f"{o['tp']:<4} {o['read_size_gb']:<10.3f} {o['read_p50_ms']:<12.1f} {o['expected_ms']:<12.1f} {o['deviation_pct']:+.1f}%{'':<6} {o['read_bw_gbs']:<10.2f} {o['nodes']}")


def print_isolated_vs_workload(analysis: Dict):
    """Compare isolated vs workload performance - summary only."""
    tps = analysis['tps']
    
    # Collect data
    data = []
    for t in tps:
        if t['read_p50_ms'] > 0 and t['workload_read_ms'] > 0:
            iso_bw = t['read_bw_gbs']
            wkld_bw = (t['read_size_gb'] / (t['workload_read_ms'] / 1000)) if t['workload_read_ms'] > 0 else 0
            overhead = ((t['workload_read_ms'] / t['read_p50_ms']) - 1) * 100 if t['read_p50_ms'] > 0 else 0
            data.append({
                'tp': t['tp'],
                'size': t['read_size_gb'],
                'iso_ms': t['read_p50_ms'],
                'wkld_ms': t['workload_read_ms'],
                'iso_bw': iso_bw,
                'wkld_bw': wkld_bw,
                'overhead': overhead,
                'nodes': t['nodes'],
            })
    
    if not data:
        return
    
    print("\n" + "=" * 80)
    print("ISOLATED vs WORKLOAD COMPARISON")
    print("=" * 80)
    
    # Summary statistics only
    overheads = [d['overhead'] for d in data]
    iso_bws = [d['iso_bw'] for d in data]
    wkld_bws = [d['wkld_bw'] for d in data]
    
    print(f"\n  Overhead (workload latency / isolated latency - 1):")
    print(f"    Mean:   {statistics.mean(overheads):+.1f}%")
    print(f"    Median: {statistics.median(overheads):+.1f}%")
    print(f"    Min:    {min(overheads):+.1f}%")
    print(f"    Max:    {max(overheads):+.1f}%")
    print(f"\n  Bandwidth:")
    print(f"    Isolated mean:  {statistics.mean(iso_bws):.2f} GB/s")
    print(f"    Workload mean:  {statistics.mean(wkld_bws):.2f} GB/s")
    print(f"    Ratio:          {statistics.mean(wkld_bws)/statistics.mean(iso_bws)*100:.1f}%")
    
    # Per-node breakdown
    node_to_data = {}
    for d in data:
        for node in d['nodes']:
            if node not in node_to_data:
                node_to_data[node] = {'overheads': [], 'iso_bws': [], 'wkld_bws': []}
            node_to_data[node]['overheads'].append(d['overhead'])
            node_to_data[node]['iso_bws'].append(d['iso_bw'])
            node_to_data[node]['wkld_bws'].append(d['wkld_bw'])
    
    if node_to_data and len(node_to_data) > 1:
        print(f"\n  Per-Node:")
        print(f"  {'Node':<6} {'Overhead':<12} {'Iso BW':<10} {'Wkld BW':<10} {'Ratio':<8}")
        print(f"  {'-'*50}")
        for node in sorted(node_to_data.keys()):
            nd = node_to_data[node]
            ratio = statistics.mean(nd['wkld_bws']) / statistics.mean(nd['iso_bws']) * 100 if nd['iso_bws'] else 0
            print(f"  {node:<6} {statistics.mean(nd['overheads']):+.1f}%{'':<6} {statistics.mean(nd['iso_bws']):<10.2f} {statistics.mean(nd['wkld_bws']):<10.2f} {ratio:<.1f}%")


def print_iteration_variance(analysis: Dict):
    """Print cross-iteration variance analysis when --all-iterations is used."""
    if not analysis.get('all_iterations'):
        return
    
    tps = analysis['tps']
    num_iters = analysis.get('num_iterations', 0)
    
    # Check if we have std fields
    if not tps or 'read_bw_gbs_std' not in tps[0]:
        return
    
    print("\n" + "=" * 90)
    print(f"CROSS-ITERATION VARIANCE ({num_iters} iterations)")
    print("=" * 90)
    
    print(f"\n{'TP':<4} {'Size(GB)':<10} {'BW Mean':<10} {'BW Std':<10} {'CV%':<8} {'Min':<10} {'Max':<10} {'Nodes':<8}")
    print("-" * 90)
    
    for t in sorted(tps, key=lambda x: x['read_size_gb']):
        if t['read_bw_gbs'] <= 0:
            continue
        
        mean_bw = t['read_bw_gbs']
        std_bw = t.get('read_bw_gbs_std', 0)
        min_bw = t.get('read_bw_gbs_min', mean_bw)
        max_bw = t.get('read_bw_gbs_max', mean_bw)
        cv = (std_bw / mean_bw * 100) if mean_bw > 0 else 0
        
        node_str = str(t['nodes']) if t['nodes'] else ""
        print(f"{t['tp']:<4} {t['read_size_gb']:<10.3f} {mean_bw:<10.2f} {std_bw:<10.3f} {cv:<8.1f} {min_bw:<10.2f} {max_bw:<10.2f} {node_str:<8}")
    
    print("-" * 90)
    
    # Summary stats
    all_cvs = [(t.get('read_bw_gbs_std', 0) / t['read_bw_gbs'] * 100) if t['read_bw_gbs'] > 0 else 0 for t in tps]
    valid_cvs = [cv for cv in all_cvs if cv > 0]
    
    if valid_cvs:
        print(f"\n  Mean CV: {statistics.mean(valid_cvs):.1f}%,  Max CV: {max(valid_cvs):.1f}%")


def print_bandwidth_histogram(analysis: Dict):
    """Print ASCII bandwidth histogram."""
    tps = analysis['tps']
    
    data = [(t['tp'], t['read_bw_gbs'], t['nodes'], t['read_size_gb']) for t in tps if t['read_bw_gbs'] > 0]
    if not data:
        return
    
    bws = [d[1] for d in data]
    median_bw = statistics.median(bws)
    
    print("\n" + "=" * 100)
    print("BANDWIDTH DISTRIBUTION (sorted by BW)")
    print("=" * 100)
    print()
    
    for tp, bw, nodes, size_gb in sorted(data, key=lambda x: x[1]):
        bar_len = int(bw * 2)
        bar = "█" * bar_len
        fast = "🚀" if bw > median_bw * 1.2 else ("🐢" if bw < median_bw * 0.8 else "  ")
        node_str = f"N{nodes[0]}" if nodes else ""
        size_str = f"{size_gb*1024:.0f}MB" if size_gb < 1 else f"{size_gb:.1f}GB"
        print(f"  TP{tp:02d} {fast} {size_str:<7} {node_str:<4} |{bar} {bw:.1f} GB/s")


def print_summary(analysis: Dict):
    """Print executive summary."""
    tps = analysis['tps']
    
    bws = [t['read_bw_gbs'] for t in tps if t['read_bw_gbs'] > 0]
    sizes = [t['read_size_gb'] for t in tps if t['read_size_gb'] > 0]
    
    if not bws:
        print("No valid data to summarize")
        return
    
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"""
  Total TPs:           {len(tps)}
  Total data:          {sum(sizes):.1f} GB
  
  Bandwidth:
    Min:               {min(bws):.2f} GB/s
    Max:               {max(bws):.2f} GB/s
    Mean:              {statistics.mean(bws):.2f} GB/s
    Median:            {statistics.median(bws):.2f} GB/s
    Std Dev:           {statistics.stdev(bws):.2f} GB/s
    CV (std/mean):     {statistics.stdev(bws)/statistics.mean(bws)*100:.1f}%
    Spread (max-min):  {(max(bws)-min(bws))/statistics.mean(bws)*100:.0f}%
""")


def export_csv(analysis: Dict, output_path: str):
    """Export results to CSV."""
    tps = analysis['tps']
    
    with open(output_path, 'w') as f:
        # Header
        headers = [
            'tp', 'read_size_gb', 'read_p50_ms', 'read_p90_ms', 'read_bw_gbs',
            'workload_read_ms', 'write_size_gb', 'write_p50_ms', 'write_bw_gbs',
            'rdma_size_gb', 'rdma_p50_ms', 'rdma_bw_gbs', 'nodes', 'numas'
        ]
        f.write(','.join(headers) + '\n')
        
        # Data
        for t in tps:
            row = [
                str(t['tp']),
                f"{t['read_size_gb']:.6f}",
                f"{t['read_p50_ms']:.3f}",
                f"{t['read_p90_ms']:.3f}",
                f"{t['read_bw_gbs']:.3f}",
                f"{t['workload_read_ms']:.3f}",
                f"{t['write_size_gb']:.6f}",
                f"{t['write_p50_ms']:.3f}",
                f"{t['write_bw_gbs']:.3f}",
                f"{t['rdma_size_gb']:.6f}",
                f"{t['rdma_p50_ms']:.3f}",
                f"{t['rdma_bw_gbs']:.3f}",
                '"' + str(t['nodes']) + '"',
                '"' + str(t['numas']) + '"',
            ]
            f.write(','.join(row) + '\n')
    
    print(f"\n✓ CSV exported to: {output_path}")



def print_per_size_analysis(analysis: Dict):
    """Print performance breakdown by transfer size."""
    tps = analysis['tps']
    
    # Group by size
    size_groups = {}
    for t in tps:
        size_gb = round(t['read_size_gb'], 2)
        if size_gb not in size_groups:
            size_groups[size_gb] = []
        if t['read_bw_gbs'] > 0:
            size_groups[size_gb].append(t)
    
    if not size_groups:
        return
    
    print("\n" + "=" * 110)
    print("PERFORMANCE BY TRANSFER SIZE")
    print("=" * 110)
    
    # Header
    print(f"\n{'Size':<10} {'TPs':<5} {'Avg BW':<10} {'Min BW':<10} {'Max BW':<10} {'Spread':<10} {'Avg Wkld':<12} {'Efficiency':<12}")
    print("-" * 110)
    
    for size_gb in sorted(size_groups.keys()):
        group = size_groups[size_gb]
        if not group:
            continue
        
        bws = [t['read_bw_gbs'] for t in group]
        wklds = [t['workload_read_ms'] for t in group if t['workload_read_ms'] > 0]
        
        avg_bw = statistics.mean(bws)
        min_bw = min(bws)
        max_bw = max(bws)
        spread = (max_bw - min_bw) / avg_bw * 100 if avg_bw > 0 else 0
        
        avg_wkld = statistics.mean(wklds) if wklds else 0
        
        # Calculate efficiency (workload BW / isolated BW)
        iso_bws = [t['read_bw_gbs'] for t in group]
        wkld_bws = [(t['read_size_gb'] / (t['workload_read_ms'] / 1000)) if t['workload_read_ms'] > 0 else 0 for t in group]
        avg_iso = statistics.mean(iso_bws)
        avg_wkld_bw = statistics.mean([w for w in wkld_bws if w > 0]) if any(w > 0 for w in wkld_bws) else 0
        efficiency = (avg_wkld_bw / avg_iso * 100) if avg_iso > 0 else 0
        
        size_str = f"{size_gb*1024:.0f}MB" if size_gb < 1 else f"{size_gb:.1f}GB"
        print(f"{size_str:<10} {len(group):<5} {avg_bw:<10.2f} {min_bw:<10.2f} {max_bw:<10.2f} {spread:<9.1f}% {avg_wkld:<12.1f} {efficiency:<11.1f}%")
    
    print("-" * 110)


def print_node_size_matrix(analysis: Dict):
    """Print Node × Size matrix showing bandwidth for each combination."""
    tps = analysis['tps']
    node_names = analysis.get('node_names', {})
    
    # Group by size
    size_groups = {}
    for t in tps:
        size_gb = round(t['read_size_gb'], 2)
        if size_gb not in size_groups:
            size_groups[size_gb] = []
        if t['read_bw_gbs'] > 0:
            size_groups[size_gb].append(t)
    
    if not size_groups:
        return
    
    sizes = sorted(size_groups.keys())
    nodes = sorted(set(t['nodes'][0] for t in tps if t['nodes'] and t['read_bw_gbs'] > 0))
    
    if not nodes:
        return
    
    print("\n" + "=" * 110)
    print("NODE × SIZE MATRIX (Isolated BW in GB/s)")
    print("=" * 110)
    
    # Header
    header = f"{'Node':<6}"
    for size_gb in sizes:
        size_str = f"{size_gb*1024:.0f}M" if size_gb < 1 else f"{size_gb:.1f}G"
        header += f"{size_str:<10}"
    print(header)
    print("-" * (6 + 10 * len(sizes)))
    
    # Per-node rows
    for node in nodes:
        row = f"N{node:<5}"
        for size_gb in sizes:
            # Find TP for this node and size
            matching = [t for t in tps if t['nodes'] and t['nodes'][0] == node and round(t['read_size_gb'], 2) == size_gb]
            if matching:
                bw = matching[0]['read_bw_gbs']
                # Flag outliers
                all_bws_for_size = [t['read_bw_gbs'] for t in size_groups[size_gb]]
                median_bw = statistics.median(all_bws_for_size)
                if bw < median_bw * 0.8:
                    row += f"{bw:<8.1f}🐢"
                elif bw > median_bw * 1.2:
                    row += f"{bw:<8.1f}🚀"
                else:
                    row += f"{bw:<10.1f}"
            else:
                row += f"{'--':<10}"
        hostname = node_names.get(node, "")
        if hostname:
            row += f"  ({hostname})"
        print(row)


def detect_bottlenecks(analysis: Dict) -> List[Dict]:
    """
    Detect potential bottlenecks and anomalies in the benchmark results.
    Returns list of detected issues with severity and recommendations.
    """
    tps = analysis['tps']
    issues = []
    
    # 1. Cold Start Penalty - Check if TP0 is significantly slower
    tp0 = next((t for t in tps if t['tp'] == 0), None)
    if tp0 and tp0['read_bw_gbs'] > 0:
        size_gb = round(tp0['read_size_gb'], 2)
        same_size_tps = [t for t in tps if round(t['read_size_gb'], 2) == size_gb and t['tp'] != 0 and t['read_bw_gbs'] > 0]
        if same_size_tps:
            avg_others = statistics.mean([t['read_bw_gbs'] for t in same_size_tps])
            penalty = (1 - tp0['read_bw_gbs'] / avg_others) * 100
            if penalty > 20:
                issues.append({
                    'type': 'COLD_START',
                    'severity': 'MEDIUM' if penalty < 40 else 'HIGH',
                    'description': f'TP0 is {penalty:.0f}% slower ({tp0["read_bw_gbs"]:.1f} vs {avg_others:.1f} GB/s avg)',
                    'recommendation': 'First TP incurs initialization overhead (GDS/cuFile warmup, NFS connection setup). Consider adding dedicated warmup phase.',
                    'tp': 0,
                })
    
    # 2. Node Imbalance - Check for consistently slow nodes
    node_bws: Dict[int, List[float]] = {}
    for t in tps:
        if t['read_bw_gbs'] > 0 and t['nodes']:
            for node in t['nodes']:
                if node not in node_bws:
                    node_bws[node] = []
                node_bws[node].append(t['read_bw_gbs'])
    
    if len(node_bws) > 1:
        node_avgs = {n: statistics.mean(bws) for n, bws in node_bws.items()}
        overall_avg = statistics.mean([bw for bws in node_bws.values() for bw in bws])
        
        for node, avg in node_avgs.items():
            deviation = (avg / overall_avg - 1) * 100
            if deviation < -15:
                issues.append({
                    'type': 'SLOW_NODE',
                    'severity': 'HIGH' if deviation < -25 else 'MEDIUM',
                    'description': f'Node {node} is {-deviation:.0f}% slower than average ({avg:.1f} vs {overall_avg:.1f} GB/s)',
                    'recommendation': 'Check: 1) NIC health (ibstat), 2) Storage mount options, 3) Background processes (nvidia-smi), 4) PCIe errors (dmesg)',
                    'node': node,
                })
    
    # 3. Large Transfer Degradation - Check if BW drops for large transfers
    size_bws: Dict[float, List[float]] = {}
    for t in tps:
        if t['read_bw_gbs'] > 0:
            size_gb = round(t['read_size_gb'], 2)
            if size_gb not in size_bws:
                size_bws[size_gb] = []
            size_bws[size_gb].append(t['read_bw_gbs'])
    
    if len(size_bws) > 2:
        sizes_sorted = sorted(size_bws.keys())
        small_sizes = sizes_sorted[:len(sizes_sorted)//3]
        large_sizes = sizes_sorted[-len(sizes_sorted)//3:]
        
        small_avg = statistics.mean([bw for s in small_sizes for bw in size_bws[s]])
        large_avg = statistics.mean([bw for s in large_sizes for bw in size_bws[s]])
        
        if large_avg < small_avg * 0.8:
            drop = (1 - large_avg / small_avg) * 100
            issues.append({
                'type': 'LARGE_TRANSFER_DEGRADATION',
                'severity': 'MEDIUM',
                'description': f'Large transfers ({large_sizes[0]:.2f}+ GB) are {drop:.0f}% slower ({large_avg:.1f} vs {small_avg:.1f} GB/s)',
                'recommendation': 'Possible causes: 1) Storage saturation, 2) Memory pressure, 3) NFS buffer limits, 4) GDS chunk size limits',
            })
    
    # 4. Small Transfer Latency Overhead
    if len(size_bws) > 2:
        smallest_size = min(size_bws.keys())
        largest_size = max(size_bws.keys())
        
        small_bw_avg = statistics.mean(size_bws[smallest_size])
        large_bw_avg = statistics.mean(size_bws[largest_size])
        
        # Expected: BW should be similar or larger transfers slightly faster
        if small_bw_avg < large_bw_avg * 0.7:
            overhead = (1 - small_bw_avg / large_bw_avg) * 100
            issues.append({
                'type': 'SMALL_TRANSFER_OVERHEAD',
                'severity': 'LOW',
                'description': f'Small transfers ({smallest_size*1024:.0f}MB) have {overhead:.0f}% lower BW ({small_bw_avg:.1f} vs {large_bw_avg:.1f} GB/s)',
                'recommendation': 'Small transfers are latency-bound. Consider batching small requests or using larger transfer sizes.',
            })
    
    # 5. High Workload Contention - Check if workload BW << isolated
    workload_comparisons = []
    for t in tps:
        if t['read_p50_ms'] > 0 and t['workload_read_ms'] > 0:
            iso_bw = t['read_bw_gbs']
            wkld_bw = t['read_size_gb'] / (t['workload_read_ms'] / 1000) if t['workload_read_ms'] > 0 else 0
            if wkld_bw > 0:
                efficiency = wkld_bw / iso_bw
                workload_comparisons.append(efficiency)
    
    if workload_comparisons:
        avg_efficiency = statistics.mean(workload_comparisons)
        if avg_efficiency < 0.3:
            issues.append({
                'type': 'SEVERE_CONTENTION',
                'severity': 'HIGH',
                'description': f'Workload efficiency is only {avg_efficiency*100:.0f}% of isolated performance',
                'recommendation': 'Severe storage contention. Consider: 1) Fewer concurrent TPs, 2) Staggered scheduling, 3) More storage bandwidth, 4) Check storage I/O parallelism limits',
            })
        elif avg_efficiency < 0.5:
            issues.append({
                'type': 'HIGH_CONTENTION',
                'severity': 'MEDIUM',
                'description': f'Workload efficiency is {avg_efficiency*100:.0f}% of isolated performance',
                'recommendation': 'Moderate storage contention. Storage bandwidth is shared inefficiently among concurrent operations.',
            })
    
    # 6. NUMA Imbalance - Check if one NUMA domain is consistently slower
    numa_bws: Dict[Tuple[int, int], List[float]] = {}
    for t in tps:
        if t['read_bw_gbs'] > 0 and t['nodes'] and t['numas']:
            for node in t['nodes']:
                for numa in t['numas']:
                    key = (node, numa)
                    if key not in numa_bws:
                        numa_bws[key] = []
                    numa_bws[key].append(t['read_bw_gbs'])
    
    if numa_bws:
        nodes_checked = set(k[0] for k in numa_bws.keys())
        for node in nodes_checked:
            numa0 = numa_bws.get((node, 0), [])
            numa1 = numa_bws.get((node, 1), [])
            if numa0 and numa1:
                avg0 = statistics.mean(numa0)
                avg1 = statistics.mean(numa1)
                diff = abs(avg1 - avg0) / max(avg0, avg1) * 100
                if diff > 20:
                    slower = "NUMA0" if avg0 < avg1 else "NUMA1"
                    issues.append({
                        'type': 'NUMA_IMBALANCE',
                        'severity': 'MEDIUM',
                        'description': f'Node {node}: {slower} is {diff:.0f}% slower ({avg0:.1f} vs {avg1:.1f} GB/s)',
                        'recommendation': 'NUMA imbalance may indicate PCIe/NIC affinity issues. Check numactl/GPU-NIC topology.',
                        'node': node,
                    })
    
    # 7. High Variance Within Size Group
    for size_gb, bws in size_bws.items():
        if len(bws) > 2:
            cv = statistics.stdev(bws) / statistics.mean(bws) * 100
            if cv > 20:
                issues.append({
                    'type': 'HIGH_VARIANCE',
                    'severity': 'MEDIUM' if cv < 35 else 'HIGH',
                    'description': f'Size {size_gb*1024:.0f}MB has {cv:.0f}% variance (min={min(bws):.1f}, max={max(bws):.1f} GB/s)',
                    'recommendation': 'High variance suggests inconsistent storage/network performance. Check for interference or node-specific issues.',
                    'size_gb': size_gb,
                })
    
    return issues


def print_bottleneck_report(analysis: Dict):
    """Print comprehensive bottleneck detection report."""
    issues = detect_bottlenecks(analysis)
    
    print("\n" + "=" * 110)
    print("🔍 BOTTLENECK DETECTION REPORT")
    print("=" * 110)
    
    if not issues:
        print("\n  ✅ No significant bottlenecks detected!")
        return
    
    # Sort by severity
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    issues.sort(key=lambda x: severity_order.get(x['severity'], 3))
    
    # Count by severity
    high = sum(1 for i in issues if i['severity'] == 'HIGH')
    medium = sum(1 for i in issues if i['severity'] == 'MEDIUM')
    low = sum(1 for i in issues if i['severity'] == 'LOW')
    
    print(f"\n  Found {len(issues)} potential issues: {high} HIGH, {medium} MEDIUM, {low} LOW")
    
    for i, issue in enumerate(issues, 1):
        severity_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[issue['severity']]
        print(f"\n  {i}. [{severity_emoji} {issue['severity']}] {issue['type']}")
        print(f"     {issue['description']}")
        print(f"     → {issue['recommendation']}")
    
    print("\n" + "-" * 110)


def print_aggregate_throughput(analysis: Dict):
    """Print aggregate throughput for the entire workload."""
    tps = analysis['tps']
    
    print("\n" + "=" * 100)
    print("AGGREGATE THROUGHPUT ANALYSIS")
    print("=" * 100)
    
    # Calculate total data
    total_read_gb = sum(t['read_size_gb'] for t in tps)
    total_write_gb = sum(t['write_size_gb'] for t in tps)
    total_rdma_gb = sum(t['rdma_size_gb'] for t in tps)
    total_data_gb = total_read_gb + total_write_gb + total_rdma_gb
    
    # Per-TP bandwidth (isolated)
    read_bws = [t['read_bw_gbs'] for t in tps if t['read_bw_gbs'] > 0]
    sum_isolated_bw = sum(read_bws) if read_bws else 0
    
    # Workload wall-clock time
    workload_latencies = [t['workload_read_ms'] for t in tps if t['workload_read_ms'] > 0]
    
    if workload_latencies:
        workload_wall_time_sec = max(workload_latencies) / 1000
        workload_aggregate_bw = total_read_gb / workload_wall_time_sec if workload_wall_time_sec > 0 else 0
    else:
        workload_wall_time_sec = 0
        workload_aggregate_bw = 0
    
    # Isolated aggregate
    isolated_latencies = [t['read_p50_ms'] for t in tps if t['read_p50_ms'] > 0]
    if isolated_latencies:
        isolated_wall_time_sec = max(isolated_latencies) / 1000
        isolated_aggregate_bw = total_read_gb / isolated_wall_time_sec if isolated_wall_time_sec > 0 else 0
    else:
        isolated_wall_time_sec = 0
        isolated_aggregate_bw = 0
    
    efficiency = (workload_aggregate_bw/isolated_aggregate_bw*100) if isolated_aggregate_bw > 0 else 0
    
    print(f"""
  TOTAL DATA VOLUME:
    Read:              {total_read_gb:.2f} GB
    Write:             {total_write_gb:.2f} GB  
    RDMA:              {total_rdma_gb:.2f} GB
    Total:             {total_data_gb:.2f} GB

  ISOLATED (Speed of Light):
    Sum of all TP BWs: {sum_isolated_bw:.2f} GB/s (if all ran in isolation)
    Max TP latency:    {max(isolated_latencies)/1000 if isolated_latencies else 0:.3f} sec
    Aggregate BW:      {isolated_aggregate_bw:.2f} GB/s (total_data / max_latency)

  WORKLOAD (All TPs concurrent):
    Wall-clock time:   {workload_wall_time_sec:.3f} sec
    Aggregate BW:      {workload_aggregate_bw:.2f} GB/s (total_data / wall_time)
    
  EFFICIENCY:
    Workload/Isolated: {efficiency:.1f}%
""")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze kvbench benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.json
  %(prog)s results.json --yaml traffic_pattern.yaml
  %(prog)s results.json --yaml traffic_pattern.yaml --csv output.csv
  %(prog)s results.json --gpus-per-node 4
  %(prog)s results.json --nodelist "hgx-isr1-[098,100,102-103]"
        """
    )
    parser.add_argument('json_path', help='Path to results JSON file')
    parser.add_argument('--yaml', '-y', dest='yaml_path', help='Path to traffic pattern YAML (for rank/node mapping)')
    parser.add_argument('--csv', '-c', dest='csv_path', help='Export results to CSV file')
    parser.add_argument('--gpus-per-node', '-g', type=int, default=8, help='GPUs per node (default: 8)')
    parser.add_argument('--iteration', '-i', type=int, default=0, help='Which iteration to analyze (default: 0)')
    parser.add_argument('--all-iterations', '-a', action='store_true', help='Analyze all iterations and show variance')
    parser.add_argument('--no-histogram', action='store_true', help='Skip bandwidth histogram')
    parser.add_argument('--no-conclusions', action='store_true', help='Skip bottleneck detection (show raw data only)')
    parser.add_argument('--nodelist', '-n', dest='nodelist', help='SLURM nodelist (e.g. "hgx-isr1-[098,100]") for physical hostname mapping')
    
    args = parser.parse_args()
    
    if not Path(args.json_path).exists():
        print(f"Error: File not found: {args.json_path}")
        sys.exit(1)
    
    if args.yaml_path and not HAS_YAML:
        print("Warning: PyYAML not installed, cannot parse YAML file")
        args.yaml_path = None
    
    # Run analysis
    analysis = analyze_results(
        args.json_path,
        args.yaml_path,
        args.gpus_per_node,
        args.iteration,
        args.nodelist,
        args.all_iterations,
    )
    
    # Print reports
    has_nodes = analysis['tps'] and analysis['tps'][0]['nodes']
    
    # Header with iteration info
    if args.all_iterations:
        print(f"\n{'='*80}")
        print(f"ANALYZING ALL {analysis['num_iterations']} ITERATIONS (aggregated)")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"ANALYZING ITERATION {args.iteration}")
        print(f"{'='*80}")
    
    print_summary(analysis)
    print_per_size_analysis(analysis)
    print_full_table(analysis, show_ranks=has_nodes)
    print_outliers(analysis)
    print_isolated_vs_workload(analysis)
    print_aggregate_throughput(analysis)
    
    if not args.no_conclusions:
        print_bottleneck_report(analysis)
    
    if has_nodes:
        print_node_analysis(analysis)
        print_numa_analysis(analysis)
        print_node_size_matrix(analysis)
    
    if args.all_iterations:
        print_iteration_variance(analysis)
    
    if not args.no_histogram:
        print_bandwidth_histogram(analysis)
    
    # Export CSV if requested
    if args.csv_path:
        export_csv(analysis, args.csv_path)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
