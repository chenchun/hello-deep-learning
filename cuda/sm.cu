#include <stdio.h>
#include <cuda_runtime.h>

// Nvidia GPU 由一个或多个流多处理器(SM，multiProcessorCount)组成，每个处理器都是一个单独的处理单元，能够并行执行多个线程（maxThreadsPerMultiProcessor）
// T4 SM=40 maxBlocksPerMultiProcessor=16 maxThreadsPerMultiProcessor=1024
// P100 (2016) SM=60
// V100 (2017) SM=84
// A100 (2020) SM=128
// H100 (2022) SM=144
// H20 SM=78 maxBlocksPerMultiProcessor=32 maxThreadsPerMultiProcessor=2048



__global__ void sm_roll_call() {
	const int threadIndex = threadIdx.x;
	
	uint streamingMultiprocessorId;
	asm("mov.u32 %0, %smid;" : "=r"(streamingMultiprocessorId) );
	
	printf("Thread %d running on SM %d!\n", threadIndex, streamingMultiprocessorId);
}

int main() {
	sm_roll_call<<<1, 5>>>();
    // Thread 0 running on SM 0!
    // Thread 1 running on SM 0!
    // Thread 2 running on SM 0!
    // Thread 3 running on SM 0!
    // Thread 4 running on SM 0!
	cudaDeviceSynchronize();

    // Launch 4 thread blocks with 2 threads per block
	sm_roll_call<<<4, 2>>>();
    // 我们看到总共有 8 个线程在 4 个不同的流多处理器上执行（即每个 SM 2 个线程）：
    // Thread 0 running on SM 6!
    // Thread 1 running on SM 6!
    // Thread 0 running on SM 4!
    // Thread 1 running on SM 4!
    // Thread 0 running on SM 0!
    // Thread 1 running on SM 0!
    // Thread 0 running on SM 2!
    // Thread 1 running on SM 2!
    cudaDeviceSynchronize();
	return 0;
}