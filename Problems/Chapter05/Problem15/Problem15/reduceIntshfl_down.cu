#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


#define DIM		 128
#define SMEMDIM	 4

// function for checking the CUDA runtime API results.
inline
void checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		printf_s("Error: %s : %d", __FILE__, __LINE__);
		printf_s("CUDA Runtime Error: %d: %s\n", result, cudaGetErrorString(result));
		exit(1);
	}
#endif
}

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
	if (size == 1) return data[0];

	int const stride = size / 2;

	for (int i = 0; i < stride; i++)
		data[i] += data[i + stride];

	return recursiveReduce(data, stride);
}

__inline__ __device__ int warpReduce(int localSum)
{
	localSum += __shfl_down(localSum, 16);
	localSum += __shfl_down(localSum, 8);
	localSum += __shfl_down(localSum, 4);
	localSum += __shfl_down(localSum, 2);
	localSum += __shfl_down(localSum, 1);

	return localSum;
}

__inline__ __device__ int warpReduceFor(int localSum)
{
	for (int i = 16; i > 0; i >>= 1)
		localSum += __shfl_down(localSum, i);

	return localSum;
}

__global__ void reduceShfl(int *g_idata, int *g_odata, unsigned int n)
{
	// shared memory for each warp sum
	__shared__ int smem[SMEMDIM];

	// boundary check
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n) return;

	// calculate lane index and warp index
	int laneIdx = threadIdx.x % warpSize;
	int warpIdx = threadIdx.x / warpSize;

	// blcok-wide warp reduce
	int localSum = warpReduceFor(g_idata[idx]); // just testing warpReduceFor

	// save warp sum to shared memory
	if (laneIdx == 0) smem[warpIdx] = localSum;

	// block synchronization
	__syncthreads();

	// last warp reduce
	if (threadIdx.x < warpSize) localSum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;

	if (warpIdx == 0) localSum = warpReduce(localSum);

	// write result for this block to global mem
	if (threadIdx.x == 0) g_odata[blockIdx.x] = localSum;
}


int main(int argc, char **argv)
{
	// set up device
	int  dev = 0;
	cudaDeviceProp  deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("%s starting reduction on ", argv[0]);
	printf_s("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	bool bResult = false;

	// initialization
	int lShft = 20;

	if (argc > 1) lShft = atoi(argv[1]);

	int nElem = 1 << lShft;
	printf_s("    with array size %d  ", nElem);

	// execution configuration
	int  blocksize = DIM;

	dim3  block(blocksize, 1);
	dim3  grid((nElem + block.x - 1) / block.x, 1);
	printf_s("grid %d block %d\n", grid.x, block.x);

	// allocate host memory
	size_t  nBytes = nElem * sizeof(int);

	int  *h_idata = (int *)malloc(nBytes);
	int  *h_odata = (int *)malloc(grid.x * sizeof(int));
	int  *tmp	  = (int *)malloc(nBytes);

	// initialize the array
	for (int i = 0; i < nElem; i++)
		h_idata[i] = (int)(rand() & 0xFF);

	memcpy(tmp, h_idata, nBytes);

	// allocate device memory
	int  *d_idata, *d_odata;
	checkCuda(cudaMalloc(&d_idata, nBytes));
	checkCuda(cudaMalloc(&d_odata, grid.x * sizeof(int)));

	// cpu reduction
	int cpu_sum = recursiveReduce(tmp, nElem);
	printf_s("cpu reduce          : %d\n", cpu_sum);

	
	// reduce Shfl
	checkCuda(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
	reduceShfl <<<grid.x, block >>>(d_idata, d_odata, nElem);
	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

	int gpu_sum = 0;
	for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

	bResult = (gpu_sum == cpu_sum);

	if (!bResult) printf_s("Test failed!\n");

	printf_s("reduceShfl          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

	// free host memory
	free(h_idata);
	free(h_odata);
	free(tmp);

	// free device memory
	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}