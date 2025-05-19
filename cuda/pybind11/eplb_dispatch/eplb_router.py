import torch
from eplb_dispatch import dispatch_tokens_to_phy_id

# 输入数据（与问题中的例子相同）
topk_weights = torch.tensor(
    [[0.0286, 0.0386, 0.0297, 0.0614, 0.0288, 0.0284],
     [0.0438, 0.0864, 0.0343, 0.0577, 0.0627, 0.0274]], device='cuda')
topk_ids = torch.tensor([[0, 2, 5, 4, 11, 7],
                         [5, 7, 4, 0, 9, 10]], device='cuda', dtype=torch.int32)
log2phy = torch.tensor(
    [[12, -1], [13, 15], [11, -1], [6, -1], [5, 7], [2, 0], [1, -1],
     [3, -1], [4, -1], [9, -1], [10, 8], [14, -1]], device='cuda', dtype=torch.int32)
expert_count = torch.tensor([1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1], device='cuda', dtype=torch.int32)

@torch.compile(dynamic=True)
def torch_dispatch_tokens_to_phy_id(
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

# 调用CUDA函数
output1 = dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy)
print(output1)

output2 = torch_dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy)
print(output2)

