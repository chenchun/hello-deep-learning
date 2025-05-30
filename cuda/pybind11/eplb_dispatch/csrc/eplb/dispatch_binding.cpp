#include <cstdio>
#include <torch/extension.h>

void dispatch_tokens_to_phy_id_cuda(
    const float* topk_weights,
    const int* topk_ids,
    const int* expert_count,
    const int* log2phy,
    int* output,
    int B,
    int T,
    int E,
    int K);

// 使用pybind11封装为Python函数（示例）
torch::Tensor dispatch_tokens_to_phy_id_cuda_wrapper(
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor expert_count,
    torch::Tensor log2phy) {

    int B = topk_weights.size(0);
    int T = topk_weights.size(1);
    int E = log2phy.size(0);
    int K = log2phy.size(1);

    auto output = torch::empty_like(topk_ids);

    dispatch_tokens_to_phy_id_cuda(
        topk_weights.data_ptr<float>(),
        topk_ids.data_ptr<int>(),
        expert_count.data_ptr<int>(),
        log2phy.data_ptr<int>(),
        output.data_ptr<int>(),
        B, T, E, K);

    return output;
}

torch::Tensor dispatch_tokens_to_phy_id_cuda_2(
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor expert_count,
    torch::Tensor log2phy);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_tokens_to_phy_id", &dispatch_tokens_to_phy_id_cuda_wrapper, "Dispatch tokens to physical IDs (CUDA)");
    m.def("dispatch_tokens_to_phy_id_2", &dispatch_tokens_to_phy_id_cuda_2, "CUDA implementation of dispatch tokens to physical IDs");

}

// TORCH_LIBRARY(extension_cpp, m) {
//     m.def("dispatch_tokens_to_phy_id(Tensor topk_weights, Tensor topk_ids, Tensor expert_count, Tensor log2phy) -> (Tensor)");
//     m.def("dispatch_tokens_to_phy_id_2(Tensor topk_weights, Tensor topk_ids, Tensor expert_count, Tensor log2phy) -> (Tensor)");
//  }

//  TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
//     m.def("dispatch_tokens_to_phy_id", &dispatch_tokens_to_phy_id_cuda_wrapper, "Dispatch tokens to physical IDs (CUDA)");
//     m.def("dispatch_tokens_to_phy_id_2", &dispatch_tokens_to_phy_id_cuda_2, "Dispatch tokens to physical IDs (CUDA)");
// }



