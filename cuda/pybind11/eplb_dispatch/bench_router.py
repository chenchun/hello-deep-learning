import torch
from eplb_dispatch import dispatch_tokens_to_phy_id, dispatch_tokens_to_phy_id_2
import torch.utils.benchmark as benchmark

# @torch.compile(dynamic=True)
def torch_dispatch_tokens_to_phy_id_raw(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_count: torch.Tensor,
        log2phy: torch.Tensor,
) -> torch.tensor:
    expert_counts = expert_count[topk_ids]
    selected_experts = torch.remainder((topk_weights * 10000).int(),
                                        expert_counts)
    # [num_tokens, topk, max_num_redundant_expert]
    topk_phy_ids = log2phy[topk_ids]
    # convert from [num_tokens, topk(0-x)] to [num_tokens, topk(phy_id)]
    topk_phy_ids = topk_phy_ids.gather(
        2, selected_experts.unsqueeze(-1).to(dtype=torch.int64)).squeeze(-1).type(topk_ids.dtype)
    return topk_phy_ids

torch_dispatch_tokens_to_phy_id = torch.compile(torch_dispatch_tokens_to_phy_id_raw)

tokens=50000

log2phy = torch.tensor(
    [[12, -1], [13, 15], [11, -1], [6, -1], [5, 7], [2, 0], [1, -1],
     [3, -1], [4, -1], [9, -1], [10, 8], [14, -1]], device='cuda', dtype=torch.int32)
expert_count = torch.tensor([1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1], device='cuda', dtype=torch.int32)
topk_weights = torch.rand(tokens, 8, device='cuda')
topk_ids = torch.rand(tokens, 8, device='cuda')*expert_count.shape[-1]
topk_ids = topk_ids.to(dtype=torch.int32)
assert dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy).allclose(
    torch_dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy))
assert torch_dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy).allclose(
    dispatch_tokens_to_phy_id_2(topk_weights, topk_ids, expert_count, log2phy))

t0 = benchmark.Timer(
    stmt='dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy)',
    setup='from eplb_dispatch import dispatch_tokens_to_phy_id',
    globals={
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
        'expert_count': expert_count,
        'log2phy': log2phy,
    })

t1 = benchmark.Timer(
    stmt='dispatch_tokens_to_phy_id_2(topk_weights, topk_ids, expert_count, log2phy)',
    setup='from eplb_dispatch import dispatch_tokens_to_phy_id_2',
    globals={
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
        'expert_count': expert_count,
        'log2phy': log2phy,
    })

t2 = benchmark.Timer(
    stmt='torch_dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy)',
    setup='from __main__ import torch_dispatch_tokens_to_phy_id',
    globals={
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
        'expert_count': expert_count,
        'log2phy': log2phy,
    })

print(t0.timeit(100))
print(t1.timeit(100))
print(t2.timeit(100))


results = []

for tokens in (1, 4, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840, 327680, 655360):
    # label and sub_label are the rows
    # description is the column
    label = 'Batched dot'
    sub_label = f'tokens={tokens}'
    topk_weights = torch.rand(tokens, 8, device='cuda')
    topk_ids = torch.rand(tokens, 8, device='cuda')*expert_count.shape[-1]
    topk_ids = topk_ids.to(dtype=torch.int32)
    for num_threads in [1, 4]:
        results.append(benchmark.Timer(
            stmt='torch_dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy)',
            setup='from __main__ import torch_dispatch_tokens_to_phy_id',
            globals={
                'topk_weights': topk_weights,
                'topk_ids': topk_ids,
                'expert_count': expert_count,
                'log2phy': log2phy,
            },
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='torch_dispatch_tokens_to_phy_id',
        ).blocked_autorange(min_run_time=0.5))
        results.append(benchmark.Timer(
            stmt='dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy)',
            setup='from eplb_dispatch import dispatch_tokens_to_phy_id',
            globals={
                'topk_weights': topk_weights,
                'topk_ids': topk_ids,
                'expert_count': expert_count,
                'log2phy': log2phy,
            },
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='dispatch_tokens_to_phy_id',
        ).blocked_autorange(min_run_time=0.5))
        results.append(benchmark.Timer(
            stmt='dispatch_tokens_to_phy_id_2(topk_weights, topk_ids, expert_count, log2phy)',
            setup='from eplb_dispatch import dispatch_tokens_to_phy_id_2',
            globals={
                'topk_weights': topk_weights,
                'topk_ids': topk_ids,
                'expert_count': expert_count,
                'log2phy': log2phy,
            },
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='dispatch_tokens_to_phy_id_2',
        ).blocked_autorange(min_run_time=0.5))

compare = benchmark.Compare(results)
compare.colorize()
compare.print()