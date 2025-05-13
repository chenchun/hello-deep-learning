#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error) << ": " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, device);

        if (error != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties returned " << static_cast<int>(error) << ": " << cudaGetErrorString(error) << std::endl;
            return 1;
        }

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  asyncEngineCount: " << deviceProp.asyncEngineCount << std::endl;
        std::cout << "  warpSize: " << deviceProp.warpSize << std::endl;
        // maxGridSize: Maximum size of each dimension of a grid
        std::cout << "  maxGridSize: [" << deviceProp.maxGridSize[0] << "," << deviceProp.maxGridSize[1] << "," << deviceProp.maxGridSize[2] << "]" << std::endl;
        std::cout << "  maxBlocksPerMultiProcessor: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
        // maxThreadsDim: Maximum size of each dimension of a block
        std::cout << "  maxThreadsDim: [" << deviceProp.maxThreadsDim[0] << "," << deviceProp.maxThreadsDim[1] << "," << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        // maxThreadsPerBlock: Maximum number of threads per block
        std::cout << "  maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << std::endl;
        // maxThreadsPerMultiProcessor: Maximum resident threads per multiprocessor
        std::cout << "  maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        // Global memory bus width in bits
        std::cout << "  memoryBusWidth: " << deviceProp.memoryBusWidth << std::endl;
        std::cout << "  multiProcessorCount: " << deviceProp.multiProcessorCount << std::endl;
        // 32-bit registers available per block
        std::cout << "  regsPerBlock: " << deviceProp.regsPerBlock << std::endl;
        // 32-bit registers available per multiprocessor
        std::cout << "  regsPerMultiprocessor: " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "  sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "  sharedMemPerMultiprocessor: " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    }

/*
Device 0: NVIDIA H20
  asyncEngineCount: 3
  warpSize: 32
  maxGridSize: [2147483647,65535,65535]
  maxBlocksPerMultiProcessor: 32
  maxThreadsDim: [1024,1024,64]
  maxThreadsPerBlock: 1024
  maxThreadsPerMultiProcessor: 2048
  memoryBusWidth: 6144
  multiProcessorCount: 78
  regsPerBlock: 65536
  regsPerMultiprocessor: 65536
  sharedMemPerBlock: 49152
  sharedMemPerMultiprocessor: 233472
Device 1: NVIDIA H20
  asyncEngineCount: 3
  warpSize: 32
  maxGridSize: [2147483647,65535,65535]
  maxBlocksPerMultiProcessor: 32
  maxThreadsDim: [1024,1024,64]
  maxThreadsPerBlock: 1024
  maxThreadsPerMultiProcessor: 2048
  memoryBusWidth: 6144
  multiProcessorCount: 78
  regsPerBlock: 65536
  regsPerMultiprocessor: 65536
  sharedMemPerBlock: 49152
  sharedMemPerMultiprocessor: 233472
*/

/*
Device 0: Tesla T4
  asyncEngineCount: 3
  warpSize: 32
  maxGridSize: [2147483647,65535,65535]
  maxBlocksPerMultiProcessor: 16
  maxThreadsDim: [1024,1024,64]
  maxThreadsPerBlock: 1024
  maxThreadsPerMultiProcessor: 1024
  memoryBusWidth: 256
  multiProcessorCount: 40
  regsPerBlock: 65536
  regsPerMultiprocessor: 65536
  sharedMemPerBlock: 49152
  sharedMemPerMultiprocessor: 65536
*/

    return 0;
}