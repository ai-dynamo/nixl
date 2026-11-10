# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import os
from typing import Callable, Optional, Union

import numpy as np
import torch


def flush_l2_cache(enabled: bool = True):
    """
    Flush the GPU L2 cache by writing a large zero-initialized tensor.

    Arguments:
        enabled: if `False`, does nothing.
    """
    l2_flush_cache_size = 256e6
    if enabled:
        torch.empty(
            int(l2_flush_cache_size // 4), dtype=torch.int, device="cuda"
        ).zero_()


def _prepare_benchmark(flush_l2: bool, barrier: Optional[Callable]):
    flush_l2_cache(flush_l2)
    if barrier is not None and not int(
        os.environ.get("NIXL_EP_DISABLE_BARRIER_PROFILING", 0)
    ):
        torch.cuda._sleep(
            int(2e7)
        )  # ~10ms: hide CPU launch skew before the rank barrier
        barrier()


def bench(
    fn,
    num_warmups: int = 50,
    num_tests: int = 50,
    post_fn: Optional[Callable] = None,
    flush_l2: bool = True,
    *,
    barrier: Optional[Callable] = None,
):
    """
    Benchmark a function using CUDA events.

    Arguments:
        fn: the function to benchmark.
        num_warmups: the number of warmup iterations.
        num_tests: the number of measurement iterations.
        post_fn: an optional function to call after each test iteration.
        flush_l2: whether to flush the L2 cache before each iteration.
        barrier: optional rank barrier after cache flushing and before the timed interval.

    Returns:
        avg: the average execution time in seconds.
        min: the minimum execution time in seconds.
        max: the maximum execution time in seconds.
    """
    torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        _prepare_benchmark(flush_l2, barrier)
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return np.average(times), np.min(times), np.max(times)


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    trace_path: Optional[str] = None,
    flush_l2: bool = True,
    barrier: Optional[Callable] = None,
):
    """
    Benchmark a function using the PyTorch profiler (kineto) to get per-kernel timing.

    Arguments:
        fn: the function to benchmark.
        kernel_names: the CUDA kernel name(s) to profile.
        num_tests: the number of test iterations.
        trace_path: the path to save the Chrome trace (`None` to skip).
        flush_l2: whether to flush the L2 cache before each iteration.
        barrier: optional zero-argument barrier called before each iteration, e.g.
            `buffer.barrier` or `torch.distributed.barrier`; `None` disables it.

    Returns:
        durations: the average kernel duration(s) in seconds.

    Raises:
        RuntimeError: if external NVIDIA tools are enabled, which conflict with Kineto.
    """
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)

    if int(os.environ.get("NIXL_EP_USE_NVIDIA_TOOLS", 0)):
        raise RuntimeError(
            "Kineto timing conflicts with NIXL_EP_USE_NVIDIA_TOOLS: disable Kineto "
            "timing in the caller, or unset NIXL_EP_USE_NVIDIA_TOOLS when external tools are not active"
        )

    # For some auto-tuning kernels with prints
    fn()
    torch.cuda.synchronize()

    # Profile
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        acc_events=True,
    )
    with profiler:
        for i in range(2):
            for _ in range(num_tests):
                _prepare_benchmark(flush_l2, barrier)
                fn()
            torch.cuda.synchronize()
            profiler.step()

    # Parse the profiling table
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert (
            sum([name in line for line in prof_lines]) <= 1
        ), f"Errors of the kernel {name} in the profiling table: {prof_lines}"

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations: list[float] = []
    for name in kernel_names:
        total_time = 0.0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += (
                            float(time_str.replace(unit, "")) / scale * int(num_str)
                        )
                        total_num += int(num_str)
                        break
        kernel_durations.append(total_time / total_num if total_num > 0 else 0)

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]
