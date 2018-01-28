#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

/*
Removing volatile from warp unrolling and using syncthreads resulted in no performance change the time to do the forced write on volatile memory and the
time taken by syncthreads must be about the same.
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


// Recursive Implementation of Interleaved Pair Approach
int cpuRecursiveReduce(int *data, int const size)
{
	// stop condition
	if (size == 1) return data[0];

	// renew the stride
	int const stride = size / 2;

	// in-place reduction
	for (int i = 0; i < stride; i++)
	{
		data[i] += data[i + stride];
	}

	// call recursively
	return cpuRecursiveReduce(data, stride);
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n)
{
	// set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x * 8;

	// unrolling 8
	if (idx + 7 * blockDim.x < n)
	{
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + 2 * blockDim.x];
		int a4 = g_idata[idx + 3 * blockDim.x];
		int b1 = g_idata[idx + 4 * blockDim.x];
		int b2 = g_idata[idx + 5 * blockDim.x];
		int b3 = g_idata[idx + 6 * blockDim.x];
		int b4 = g_idata[idx + 7 * blockDim.x];
		g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}

	__syncthreads();

	// in-place reduction and complete unroll
	if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

	__syncthreads();

	if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

	__syncthreads();

	if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

	__syncthreads();

	if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

	__syncthreads();

	// unrolling warp
	if (tid < 32)
	{
		int *mem = idata;
		mem[tid] += mem[tid + 32];
		__syncthreads();
		mem[tid] += mem[tid + 16];
		__syncthreads();
		mem[tid] += mem[tid + 8];
		__syncthreads();
		mem[tid] += mem[tid + 4];
		__syncthreads();
		mem[tid] += mem[tid + 2];
		__syncthreads();
		mem[tid] += mem[tid + 1];

	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
	int dev = 0;
	cudaDeviceProp	deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));

	printf_s("%s starting reduction at ", argv[0]);
	printf_s("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	bool bResult = false;

	// initialization
	int size = 1 << 24; // total number of elements to reduce
	printf("    with array size %d  ", size);

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
	time_t t;
	srand(time(&t));

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

	int cpusum = cpuRecursiveReduce(tmp, size);
	printf_s("cpu_sum: %d\n", cpusum);

	checkCuda(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaDeviceSynchronize());

	reduceCompleteUnrollWarps8 << <grid.x / 8, block >> >(d_idata, d_odata, size);
	checkCuda(cudaDeviceSynchronize());

	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
	
	int gpusum = 0;
	for (int i = 0; i < grid.x / 8; i++) gpusum += h_odata[i];

	printf("gpu Cmptnroll8 sum: %d <<<grid %d block %d>>>\n", gpusum, grid.x / 8, block.x);

	free(h_idata);
	free(h_odata);

	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));

	bResult = (gpusum == cpusum);
	if (!bResult) printf_s("Test failed\n");

	return EXIT_SUCCESS;

}