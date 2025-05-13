// nvcc hello.cu -o hello.o

// https://tinkerd.net/blog/machine-learning/cuda-basics/#writing-a-cuda-kernel

#include <stdio.h>
#include <cuda_runtime.h>

// 函数签名中的关键字__global__告诉编译器该函数是一个 CUDA 内核，它将从 CPU 调用并在 GPU 上执行
// cuda函数必须在.cu文件中定义，如果用.cpp文件定义，nvcc编译的时候会报错找不到threadIdx
__global__ void roll_call() {
	const int threadIndex = threadIdx.x;
	printf("Thread %d here!\n", threadIndex);
}

// main()函数本身像任何其他 C++ 程序一样在 CPU 上执行
int main() {
    // 三尖括号语法（即<<<1, 10>>>）是执行 CUDA 内核时所需的另一个 CUDA 特定的 C++ 扩展。
    // 第一个参数指定要启动的线程块的数量，第二个参数指定每个块将并行运行的线程数。
    // 每个线程块的线程数量上限为 1024 个(maxThreadsPerBlock)，如果我们尝试在单个线程块中
    // 启动包含超过 1024 个线程的内核，CUDA 将会抛出错误invalid configuration argument
	roll_call<<<1, 10>>>();
    // 我们在启动内核后调用cudaDeviceSynchronize()。这会强制暂停执行，main()直到内核执行完毕。
    // 默认情况下，CUDA 内核是异步执行的，这意味着即使 GPU 仍在执行内核，CPU 上的指令仍将继续执行。
	cudaDeviceSynchronize();
	return 0;
}