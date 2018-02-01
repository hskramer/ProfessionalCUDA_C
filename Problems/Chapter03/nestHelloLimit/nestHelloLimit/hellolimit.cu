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

__global__ void nestHelloLimit(int iSize, int maxDepth, int iDepth)
{
	int  tid = threadIdx.x;

	printf("Recursion = %d: Hello from thread %d\n", iDepth, tid);

	// condition to stop recursive execution
	if (iSize == 1 || iDepth >= maxDepth)  return;

	// reduce nthreads by half
	int nthreads = iSize >> 1;

	// thread 0 launches child grid recursively
	if (tid == 0 && nthreads > 0)
	{
		nestHelloLimit <<<1,  nthreads>>> (nthreads, maxDepth, ++iDepth);
		printf("--------> nested execution depth: %d\n", iDepth);
	}
	
}

int main(int argc, char** argv)
{
	int  numBlocks = 1;
	int  threads = 8;
	int  depth = 2;

	if (argc > 1)
	{
		depth = atoi(argv[1]);
	}

	int  size = numBlocks * threads;

	dim3  threadsPerBlock (threads, 1);
	dim3  blocksPerGrid ((size + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

	printf_s("To use program  enter max depth(1-4) or enter for default\n");
	printf_s("size: %d\n", size);
	printf_s("grid: %d\n", numBlocks);
	printf_s("%s Execution Configuration: grid %d block %d\n", argv[0], threadsPerBlock.x, blocksPerGrid.x);

	nestHelloLimit <<<blocksPerGrid, threadsPerBlock >>> (threadsPerBlock.x, depth, 0);

	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());

	return EXIT_SUCCESS;

}
