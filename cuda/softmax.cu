#include <stdio.h>
#include <cuda_runtime.h>

// 为了简单起见，我们假设输入数组的大小是线程块中线程数的倍数，以便每个线程负责计算输入数组中多个元素的 Softmax。

__global__ void softmax_kernel(float *input, float *output, int size) {
    
	// Number of threads in the Thread Block.
	// Assumes that the Thread Block is one-dimensional
	int num_threads = blockDim.x;

	// Each thread will compute the softmax of 
	// num_elements_per_thread elements
    int num_elements_per_thread = size / num_threads;
    
    int thread_index = threadIdx.x;

	// This thread will compute the softmax of elements from 
	// start_idx to end_idx in the input array
    int start_idx = thread_index * num_elements_per_thread;
    int end_idx = min(size, start_idx + num_elements_per_thread);

    // Loop over the input array to find the maximum value
    float max_val = 0.0;
    for (int i = 0; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Loop over the input array to compute sum of the exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_exp += expf(input[i] - max_val);
    }

    // Store the softmax result in the output array
    for (int i = start_idx; end_idx; i++) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}

//上述实现导致每个线程执行大量重复工作。即使每个线程负责计算特定num_elements_per_thread元素子集的 Softmax，它们仍然会循环遍历整个输入数组两次，以找到最大值并计算指数之和。




__global__ void softmax_kernel_smem(float *input, float *output, int size) {
    
    // Number of threads in the Thread Block.
	// Assumes that the Thread Block is one-dimensional
    int num_threads = blockDim.x;

    // Each thread will compute the softmax of num_elements_per_thread elements
    int num_elements_per_thread = size / num_threads;
    
    int thread_index = threadIdx.x;

    // This thread will compute the softmax of elements from start_idx to end_idx
	// in the input array
    int start_idx = thread_index * num_elements_per_thread;
    int end_idx = min(size, start_idx + num_elements_per_thread);

    // Array in shared memory to store the local max values
    __shared__ float shared_max_val[NUM_THREADS];

    float max_val = 0.0;
    for (int i = start_idx; i < end_idx; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    shared_max_val[thread_index] = max_val;

    // Wait for all threads to finish writing their
    // local max values to shared memory
    __syncthreads();

    for (int i = 0; i < num_threads; i++) {
        if (shared_max_val[i] > max_val) {
            max_val = shared_max_val[i];
        }
    }

    // Array in shared memory to store the local sums of 
    // the exponentials
    __shared__ float shared_sum_exp[NUM_THREADS];

    float sum_exp = 0.0f;
    for (int i = start_idx; i < end_idx; i++) {
        sum_exp += expf(input[i] - max_val);
    }
    shared_sum_exp[thread_index] = sum_exp;

    // Wait for all threads to finish writing their
    // local sums
    __syncthreads();

    for (int i = 0; i < num_threads; i++) {
        sum_exp += shared_sum_exp[i];
    }

    // Compute softmax
    for (int i = start_idx; i < end_idx; i++) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}
// 当输入数组大小为 6,144 且线程数为 1,024 时，共享内存实现的执行时间为 1.5 毫秒，而原始实现的执行时间为 11.23 毫秒。