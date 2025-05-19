#include <cuda_runtime.h>
#include <cstdio>

__global__ void dispatch_tokens_kernel(
    const float* topk_weights,
    const int* topk_ids,
    const int* expert_count,
    const int* log2phy,
    int* output,
    int B,
    int T,
    int K) {

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || t >= T) return;

    int idx = b * T + t;
    int k = topk_ids[idx];
    int ec = expert_count[k];

    float weight = topk_weights[idx];
    int scaled_weight = static_cast<int>(weight * 10000.0f);
    int selected_expert = scaled_weight % ec;

    output[idx] = log2phy[k * K + selected_expert];
}

// 封装CUDA操作的函数
void dispatch_tokens_to_phy_id_cuda(
    const float* topk_weights,
    const int* topk_ids,
    const int* expert_count,
    const int* log2phy,
    int* output,
    int B,
    int T,
    int E,
    int K) {

    dim3 block(16, 16);
    dim3 grid((B + block.x - 1) / block.x, (T + block.y - 1) / block.y);

    dispatch_tokens_kernel<<<grid, block>>>(
        topk_weights, topk_ids, expert_count, log2phy, output, B, T, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}