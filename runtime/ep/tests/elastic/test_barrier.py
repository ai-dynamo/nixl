# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
import torch.distributed as dist

from nixl.runtime import ep
from nixl.runtime.ep.utils.envs import init_dist, dist_print
from nixl.runtime.ep.utils.testing import bench_kineto


def test_barrier(buffer: ep.ElasticBuffer, args: argparse.Namespace):
    dist_print('Profiling barrier:', once_in_node=True)
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()
    dist_print(f'Config:\n'
               f' > Ranks: {num_scaleout_ranks} x {num_scaleup_ranks}\n',
               once_in_node=True)

    # Test barrier time
    def loop_barrier(num_tests=1000):
        for i in range(num_tests):
            buffer.barrier()

    t = bench_kineto(lambda: loop_barrier(), 'barrier', barrier_comm_profiling=True, barrier=buffer.barrier)
    dist_print(f' > EP: {buffer.rank_idx:3}/{buffer.num_ranks:3}, '
               f'barrier time: {t * 1e6:.3f} us')
    dist_print(once_in_node=True)


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    do_pressure_test = args.do_pressure_test
    for i in range(int(1e9) if do_pressure_test else 1):
        buffer = ep.ElasticBuffer(
            group, num_bytes=2 ** 30,
            explicitly_destroy=True
        )

        # Test barrier
        test_barrier(buffer, args)

        # Destroy the runtime and communication group
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test elastic EP barrier performance')

    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')
    args = parser.parse_args()

    # Launch test processes
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
