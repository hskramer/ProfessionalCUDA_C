#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

/*
This version is slower than the simple unroll. Memory throughput was almost 10% slower, stalled warps where 13% higher. Achieved occupancy was exactly the  
same making this a good example of why achieving the highest occupancy does not translate into the best performance.
*/

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


int cpuSumRecursive(int *data, int const size)
{
	// terminate check
	if (size == 1) return data[0];

	// renew the stride
	int const stride = size / 2;

	// in-place reduction
	for (int i = 0; i < stride; i++)
	{
		data[i] += data[i + stride];
	}

	// call recursively
	return cpuSumRecursive(data, stride);
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
	// This is version 2 with the changes given in problem 3 
	// Problem 3: compare the performance of this version and the original and explain the difference using nvprof metrics

	// set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x * 8;

	// unrolling 8
	if (idx + 7 * blockDim.x < n)
	{
		int *ptr = g_idata + idx;
		int tmp = 0;
		// Increment tmp 8 times with values strided by blockDim.x 
		for (int i = 0; i < 8; i++)
		{
			tmp += *ptr;
			ptr += blockDim.x;
		}

		g_idata[idx] = tmp;
	}

	__syncthreads();

	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			idata[tid] += idata[tid + stride];
		}

		// synchronize within threadblock
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
	int dev = 0;
	cudaDeviceProp deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("%s starting reduction at ", argv[0]);
	printf_s("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	bool bResult = false;

	// initialization
	int size = 1 << 24;   // total number of elements to reduce

	printf_s("    with array size %d  ", size);

	// execution configuration
	int blocksize = 512;   // initial block size

	if (argc > 1)
	{
		blocksize = atoi(argv[1]);   // block size from command line argument
	}

	dim3 block(blocksize, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);
	printf_s("grid %d block %d\n", grid.x, block.x);

	// allocate host memory
	size_t bytes = size * sizeof(int);

	int *h_idata = (int *)malloc(bytes);
	int *h_odata = (int *)malloc(grid.x * sizeof(int));
	int *tmp = (int *)malloc(bytes);

	// fill host array with random integers 
	srand(time(NULL));

	for (int i = 0; i < size; i++)
	{
		h_idata[i] = (int)(rand() % 100);
	}

	memcpy(tmp, h_idata, bytes);

	// allocate device memory
	int *d_idata = NULL;
	int *d_odata = NULL;
	checkCuda(cudaMalloc((void **)&d_idata, bytes));
	checkCuda(cudaMalloc((void **)&d_odata, grid.x * sizeof(int)));

	cpusum = cpuSumRecursive(tmp, size);
	printf_s("cpu_sum: %d\n", cpusum);

	checkCuda(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaDeviceSynchronize());

	reduceUnrolling8 << <grid.x / 8, block >> >(d_idata, d_odata, size);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

	int gpusum = 0;
	for (int i = 0; i < grid.x / 8; i++) gpusum += h_odata[i];

	printf_s("Urolling 8 version 2 gpu sum: %d <<<grid %d block %d>>>\n", gpusum, grid.x / 8, block.x);

	// free host memory
	free(h_idata);
	free(h_odata);

	// free device memory
	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));

	// reset device
	checkCuda(cudaDeviceReset());

	// check the results
	bResult = (gpusum == cpusum);

	if (!bResult) printf_s("Test failed!\n");

	return EXIT_SUCCESS;


}