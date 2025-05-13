// https://tinkerd.net/blog/machine-learning/cuda-basics/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void transpose_matrix(int* in, int* out) {
	//线程块的维度，它们存储在dim3类型的变量blockDim中

	// Select an element from the input matrix by using the
	// threadIdx x and y values. blockDim.x contains the number
	// of rows in the 2D Thread Block.
	const int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;

	// Select the corresponding position in the output matrix.
	// blockDim.y contains the number of columns in the 2D Thread Block.
	const int outIdx = threadIdx.y + threadIdx.x * blockDim.y;
	out[outIdx] = in[threadIndex];
}


int main() {

	// Allocate host & GPU memory; Copy the input array to GPU memory
	// ...

	int rows = 3;
	int cols = 4;
	// threadIdx变量的类型是一个包含3个元素的整数元组，其元素可以通过xyz访问
	// 每个线程块的线程数设置为二维或三维元组而不是单个整数完全是可选的，在编写处理多维数据的内核时会很有帮助
	// 将线程块维度声明为一种dim3类型，然后将其作为每个线程块的线程数传入三角括号
	dim3 numThreadsPerBlock(rows, cols);
	transpose_matrix<<<1, numThreadsPerBlock>>>(d_in_matrix, d_out_matrix);

	// Release host and GPU memory
	// ...
}
