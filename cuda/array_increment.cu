//https://tinkerd.net/blog/machine-learning/cuda-basics/#reading-and-writing-data-in-cuda-kernels

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA 内核无法显式返回值，因此必须始终将其声明为void
// 为了持久化计算结果，我们必须就地修改输入数据（如上例所示），或者将结果写入用于存储内核输出的单独数组
__global__ void array_increment(int* in) {
	const int threadIndex = threadIdx.x;
	in[threadIndex] = in[threadIndex] + 1;
}

void printArray(int* array, int arraySize) {
	printf("[");
	for (int i = 0; i < arraySize; i++) {
		printf("%d", array[i]);
		if (i < arraySize - 1) {
			printf(", ");
		}
	}
	printf("]\n");
}

// $ ./array_increment
// Before: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
// After:  [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]

int main() {
	const int arraySize = 10;
	
	// Allocate host memory for the input array
	int* array = (int*)malloc(arraySize * sizeof(int));
	
	// Initialize the input array
	for (int i = 0; i < arraySize; i++) {
		array[i] = i*10;
	}
	
	printf("Before: ");
	printArray(array, arraySize);
	
	// Allocate GPU memory for the input array
    // 使用函数在 GPU 上分配等量的内存cudaMalloc，因为 CUDA 内核只能访问驻留在 GPU 内存中的数据
	int* d_array;
	cudaMalloc((void**)&d_array, arraySize * sizeof(int));
	
	// Copy the input array from host memory to GPU memory
	cudaMemcpy(d_array, array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	
	array_increment<<<1, arraySize>>>(d_array);
	
	// Copy the result array from GPU memory back to host memory
    // 在此示例中我们不需要cudaDeviceSynchronize()在启动内核后调用，因为将结果复制回主机内存的最终调用cudaMemcpy将强制 CPU 等待内核执行完毕。
	cudaMemcpy(array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("After: ");
	printArray(array, arraySize);
	
	// Free the host and GPU memory
	free(array);
	cudaFree(d_array);
	return 0;
}