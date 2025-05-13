import torch
from dispatch_cuda import dispatch_tokens_to_phy_id

# 输入数据（与问题中的例子相同）
topk_weights = torch.tensor(
    [[0.0286, 0.0386, 0.0297, 0.0614, 0.0288, 0.0284],
     [0.0438, 0.0864, 0.0343, 0.0577, 0.0627, 0.0274]], device='cuda')
topk_ids = torch.tensor([[0, 2, 5, 4, 11, 7],
                         [5, 7, 4, 0, 9, 10]], device='cuda')
log2phy = torch.tensor(
    [[12, -1], [13, 15], [11, -1], [6, -1], [5, 7], [2, 0], [1, -1],
     [3, -1], [4, -1], [9, -1], [10, 8], [14, -1]], device='cuda')
expert_count = torch.tensor([1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1], device='cuda')

# 调用CUDA函数
output = dispatch_tokens_to_phy_id(topk_weights, topk_ids, expert_count, log2phy)
print(output)