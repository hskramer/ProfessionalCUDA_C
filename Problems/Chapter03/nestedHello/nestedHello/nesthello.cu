#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


inline
void checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		printf("Error: %s : %d", __FILE__, __LINE__);
		printf("CUDA Runtime Error: %d: %s\n", result, cudaGetErrorString(result));
		exit(1);
	}
#endif
}

__global__ void nestedHello(int iSize, int iMin, int iDepth)
{
	int  tid = blockIdx.x * blockDim.x + threadIdx.x;

	printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid);

	// stop condition
	if (iSize == iMin)  return;

	// half threads
	int  nthreads = iSize >> 1;

	// use thread 0 to launch child grid using recursion
	if (tid == 0 && nthreads > 0)
	{
		int  blocks = (nthreads + blockDim.x - 1) / blockDim.x;
		nestedHello<<<blocks, blockDim.x>>> (nthreads, iMin, ++iDepth);
		printf("----------> nested execution level: %d\n", iDepth);
	}
}


int main(int argc, char** argv)
{
	int  blocksize = 8; // initial block size
	int  igrid = 1;

	if (argc > 1)
	{
		igrid = atoi(argv[1]);		
	}

	if (argc > 2)
	{
		blocksize = atoi(argv[2]);
	}

	int  size = igrid * blocksize;

	dim3  block  (blocksize, 1);
	dim3  grid((size + block.x - 1) / block.x, 1);

	printf_s("size = %d\n", size);
	printf_s("igrid = %d\n", igrid);
	printf_s("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x, block.x);
	nestedHello<<<grid,  block>>> (size, grid.x, 0);

	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());

	return EXIT_SUCCESS;
}