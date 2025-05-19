#include <stdio.h>
#include <cuda_runtime.h>


__global__ void roll_call_kernel() {
	const int threadIndex = threadIdx.x;
	printf("Thread %d here!\n", threadIndex);
}

void roll_call_launcher() {
    roll_call_kernel<<<1, 5>>>();
    cudaDeviceSynchronize();
}