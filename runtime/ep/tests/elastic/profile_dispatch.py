"""
Phase timing of the dispatch kernel (debug). Run with NIXL_EP_ADAPTIVE_PROFILE=1 so the JIT compiles the
`EP_PROFILE_STAMP` timestamps in; prints, on rank 0, the time from kernel entry to each phase boundary (all stamps are SM 0's).

    NIXL_EP_ADAPTIVE_PROFILE=1 python tests/elastic/profile_dispatch.py --num-processes 4 --num-sms 26
"""
import argparse
import os
import torch
import torch.distributed as dist

import nixl_ep_v2
from nixl_ep_v2.utils.envs import init_dist
from nixl_ep_v2.utils.gate import get_unbalanced_scores

PHASES = ['entry', 'start barrier done', 'notify: grid reduction done', 'notify: peer writes issued',
          'notify: peer counts arrived', 'notify: prefix sums done', 'end barrier reached', 'end barrier done',
          'notify: token counting done', 'end barrier: local grid sync done', 'first data warp / last sender warp done',
          'last data warp / last forward warp done']


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=0)
    assert args.num_experts % num_ranks == 0
    buffer = nixl_ep_v2.ElasticBuffer(group, num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                                   num_experts_per_rank=args.num_experts // num_ranks,
                                   allow_multiple_reduction=True, prefer_overlap_with_compute=True,
                                   explicitly_destroy=True)
    num_tokens, hidden, num_topk, num_experts = args.num_tokens_used, args.hidden, args.num_topk, args.num_experts
    num_sms = args.num_sms or buffer.get_theoretical_num_sms(num_topk)
    scores = get_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, 1.0, False)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(nixl_ep_v2.topk_idx_t)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    rows = []
    for it in range(args.iters):
        torch.cuda.synchronize(); dist.barrier()
        buffer.dispatch(x=x, topk_idx=topk_idx, topk_weights=topk_weights, num_sms=num_sms,
                        num_max_tokens_per_rank=args.num_tokens, do_cpu_sync=True)
        torch.cuda.synchronize()
        prof = buffer.runtime.read_profile()
        if it >= 2:  # skip JIT / warm-up iterations
            rows.append(prof)
    if rank_idx == 0:
        mode = buffer.get_topology()[0]
        print(f'== mode={mode} num_sms={num_sms} tokens={num_tokens}: '
              f'median over {len(rows)} iterations, us from kernel entry ==', flush=True)
        for slot, name in enumerate(PHASES):
            deltas = sorted((r[slot] - r[0]) / 1000.0 for r in rows if r[slot] != 0)
            if deltas:
                print(f'  {slot:>2} {name:<32} {deltas[len(deltas) // 2]:8.1f}', flush=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-sms', type=int, default=0)
    parser.add_argument('--num-tokens', type=int, default=4096, help='buffer max tokens per rank')
    parser.add_argument('--num-tokens-used', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--iters', type=int, default=12)
    args = parser.parse_args()
    assert os.environ.get('NIXL_EP_ADAPTIVE_PROFILE') == '1', 'set NIXL_EP_ADAPTIVE_PROFILE=1'
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
