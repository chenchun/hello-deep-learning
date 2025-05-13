// https://tinkerd.net/blog/machine-learning/cuda-basics/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void transpose_matrix(int* in, int* out) {
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

	dim3 numThreadsPerBlock(rows, cols);
	transpose_matrix<<<1, numThreadsPerBlock>>>(d_in_matrix, d_out_matrix);

	// Release host and GPU memory
	// ...
}
