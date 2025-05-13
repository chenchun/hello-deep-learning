#include <stdio.h>
#include <cuda_runtime.h>

// https://tinkerd.net/blog/machine-learning/cuda-basics/#warp-schedulers
// 当 CUDA 内核启动时，它的线程块首先分布在 GPU 的流多处理器中（如上一个示例所示）；接下来，这些线程块中的线程被调度并以最多 32 个（称为Warp）的批次执行。
// 每个流多处理器 (SM) 都有一个或多个Warp 调度器 (Scheduler)，负责选择在 SM 的计算核心上执行哪些 Warp。
// 例如，A100 GPU 的流多处理器（如下图所示，取自NVIDIA A100 Tensor Core GPU 架构白皮书）显示，此类 SM 有 4 个 Warp 调度器
// Warp 调度器的任务是在每个周期决定哪些可用的 Warp 应该继续运行。在某些情况下，可能没有任何 Warp 能够继续运行。
// Warp 调度器的周期被想象成一条装满空桶的传送带。在每个周期，Warp 调度器必须从可用的 Warp 中选择下一条指令，并将其放入桶中（即“发出槽”）。
// 如果所有 Warp 中都没有准备好执行的指令，则桶为空。
__global__ void warp_roll_call() {

	const int threadIndex = threadIdx.x;
	
	uint streamingMultiprocessorId;
	asm("mov.u32 %0, %smid;" : "=r"(streamingMultiprocessorId));
	
	uint warpId;
	asm volatile ("mov.u32 %0, %warpid;" : "=r"(warpId));
	
    // Lane指的是特定 warp 内的线程的索引，即从 0 到 31 的值
	uint laneId;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneId));
	
	printf("SM: %d | Warp: %d | Lane: %d | Thread %d - Here!\n", streamingMultiprocessorId, warpId, laneId, threadIndex);
}

int main() {

    // Launch 4 thread blocks with 2 threads per block
    printf("warp_roll_call<<<4, 2>>>()\n");
	warp_roll_call<<<4, 2>>>();
    // 由于线程块的线程数均少于 32 个，因此所有线程都在每个 SM 上的单个 Warp（索引为 0）中执行
    // SM: 6 | Warp: 0 | Lane: 0 | Thread 0 - Here!
    // SM: 6 | Warp: 0 | Lane: 1 | Thread 1 - Here!
    // SM: 4 | Warp: 0 | Lane: 0 | Thread 0 - Here!
    // SM: 4 | Warp: 0 | Lane: 1 | Thread 1 - Here!
    // SM: 2 | Warp: 0 | Lane: 0 | Thread 0 - Here!
    // SM: 2 | Warp: 0 | Lane: 1 | Thread 1 - Here!
    // SM: 0 | Warp: 0 | Lane: 0 | Thread 0 - Here!
    // SM: 0 | Warp: 0 | Lane: 1 | Thread 1 - Here!
    cudaDeviceSynchronize();
    printf("warp_roll_call<<<1, 40>>>()\n");
    warp_roll_call<<<1, 40>>>();
    // 我们运行一个更大的线程块（例如，一个包含 40 个线程的线程块），则执行会被拆分成两个 Warp：
    // SM: 0 | Warp: 1 | Lane: 0 | Thread 32 - Here!
    // SM: 0 | Warp: 1 | Lane: 1 | Thread 33 - Here!
    // SM: 0 | Warp: 1 | Lane: 2 | Thread 34 - Here!
    // SM: 0 | Warp: 1 | Lane: 3 | Thread 35 - Here!
    // SM: 0 | Warp: 1 | Lane: 4 | Thread 36 - Here!
    // SM: 0 | Warp: 1 | Lane: 5 | Thread 37 - Here!
    // SM: 0 | Warp: 1 | Lane: 6 | Thread 38 - Here!
    // SM: 0 | Warp: 1 | Lane: 7 | Thread 39 - Here!
    // SM: 0 | Warp: 0 | Lane: 0 | Thread 0 - Here!
    // SM: 0 | Warp: 0 | Lane: 1 | Thread 1 - Here!
    // SM: 0 | Warp: 0 | Lane: 2 | Thread 2 - Here!
    // SM: 0 | Warp: 0 | Lane: 3 | Thread 3 - Here!
    // SM: 0 | Warp: 0 | Lane: 4 | Thread 4 - Here!
    // SM: 0 | Warp: 0 | Lane: 5 | Thread 5 - Here!
    // SM: 0 | Warp: 0 | Lane: 6 | Thread 6 - Here!
    // SM: 0 | Warp: 0 | Lane: 7 | Thread 7 - Here!
    // SM: 0 | Warp: 0 | Lane: 8 | Thread 8 - Here!
    // SM: 0 | Warp: 0 | Lane: 9 | Thread 9 - Here!
    // SM: 0 | Warp: 0 | Lane: 10 | Thread 10 - Here!
    // SM: 0 | Warp: 0 | Lane: 11 | Thread 11 - Here!
    // SM: 0 | Warp: 0 | Lane: 12 | Thread 12 - Here!
    // SM: 0 | Warp: 0 | Lane: 13 | Thread 13 - Here!
    // SM: 0 | Warp: 0 | Lane: 14 | Thread 14 - Here!
    // SM: 0 | Warp: 0 | Lane: 15 | Thread 15 - Here!
    // SM: 0 | Warp: 0 | Lane: 16 | Thread 16 - Here!
    // SM: 0 | Warp: 0 | Lane: 17 | Thread 17 - Here!
    // SM: 0 | Warp: 0 | Lane: 18 | Thread 18 - Here!
    // SM: 0 | Warp: 0 | Lane: 19 | Thread 19 - Here!
    // SM: 0 | Warp: 0 | Lane: 20 | Thread 20 - Here!
    // SM: 0 | Warp: 0 | Lane: 21 | Thread 21 - Here!
    // SM: 0 | Warp: 0 | Lane: 22 | Thread 22 - Here!
    // SM: 0 | Warp: 0 | Lane: 23 | Thread 23 - Here!
    // SM: 0 | Warp: 0 | Lane: 24 | Thread 24 - Here!
    // SM: 0 | Warp: 0 | Lane: 25 | Thread 25 - Here!
    // SM: 0 | Warp: 0 | Lane: 26 | Thread 26 - Here!
    // SM: 0 | Warp: 0 | Lane: 27 | Thread 27 - Here!
    // SM: 0 | Warp: 0 | Lane: 28 | Thread 28 - Here!
    // SM: 0 | Warp: 0 | Lane: 29 | Thread 29 - Here!
    // SM: 0 | Warp: 0 | Lane: 30 | Thread 30 - Here!
    // SM: 0 | Warp: 0 | Lane: 31 | Thread 31 - Here!
    cudaDeviceSynchronize();

	return 0;
}