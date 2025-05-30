#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <torch/extension.h>

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

__global__ void dispatch_single_col_kernel(
    const int32_t* topk_ids,
    const int32_t* log2phy,
    int32_t* output,
    int batch_size,
    int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * k) return;
    int i = idx / k;
    int j = idx % k;
    int32_t log_id = topk_ids[i * k + j];
    output[i * k + j] = log2phy[log_id];
}

template <typename scalar_t>
__global__ void dispatch_multi_col_kernel(
    const scalar_t* topk_weights,
    const int32_t* topk_ids,
    const int32_t* expert_count,
    const int32_t* log2phy,
    int32_t* output,
    int log2phy_cols,
    int batch_size,
    int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * k) return;
    int i = idx / k;
    int j = idx % k;

    int32_t log_id = topk_ids[i * k + j];
    int32_t count = expert_count[log_id];
    scalar_t weight = topk_weights[i * k + j];
    int32_t scaled = static_cast<int32_t>(weight * 10000.0f);
    int32_t selected = scaled % count;

    output[i * k + j] = log2phy[log_id * log2phy_cols + selected];
}

torch::Tensor dispatch_tokens_to_phy_id_cuda_2(
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor expert_count,
    torch::Tensor log2phy) {
    TORCH_CHECK(topk_weights.is_cuda(), "topk_weights must be a CUDA tensor");
    TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be a CUDA tensor");
    TORCH_CHECK(expert_count.is_cuda(), "expert_count must be a CUDA tensor");
    TORCH_CHECK(log2phy.is_cuda(), "log2phy must be a CUDA tensor");

    int batch_size = topk_ids.size(0);
    int k = topk_ids.size(1);
    int log2phy_cols = log2phy.size(1);
    auto output = torch::empty_like(topk_ids);

    int threads = 256;
    int blocks = (batch_size * k + threads - 1) / threads;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (log2phy_cols == 1) {
        dispatch_single_col_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<int32_t*>(topk_ids.data_ptr()),
            reinterpret_cast<int32_t*>(log2phy.data_ptr()),
            reinterpret_cast<int32_t*>(output.data_ptr()),
            batch_size,
            k);
    } else {
        // reference https://docs.pytorch.org/tutorials/advanced/cpp_extension.html
        AT_DISPATCH_FLOATING_TYPES(topk_weights.scalar_type(), "dispatch_multi_col_kernel_", ([&]{
            dispatch_multi_col_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                topk_weights.data_ptr<scalar_t>(),
                reinterpret_cast<int32_t*>(topk_ids.data_ptr()),
                reinterpret_cast<int32_t*>(expert_count.data_ptr()),
                reinterpret_cast<int32_t*>(log2phy.data_ptr()),
                reinterpret_cast<int32_t*>(output.data_ptr()),
                log2phy_cols,
                batch_size,
                k);
        }));
    }

    return output;
}