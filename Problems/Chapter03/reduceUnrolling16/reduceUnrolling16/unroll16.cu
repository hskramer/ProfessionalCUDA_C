#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

/*
This kernel improved memory throughput by 9 percent to 218GB/s and instructions per warp by 40 percent to 190 also fully utilized all the 
registersl. Will try a warp unroll next then use PGI on ubuntu and their pragma unroll in both c and fortran.
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


__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n)
{
	// set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x * 16;

	// unrolling 16
	if (idx + 15 * blockDim.x < n)
	{
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + 2 * blockDim.x];
		int a4 = g_idata[idx + 3 * blockDim.x];
		int b1 = g_idata[idx + 4 * blockDim.x];
		int b2 = g_idata[idx + 5 * blockDim.x];
		int b3 = g_idata[idx + 6 * blockDim.x];
		int b4 = g_idata[idx + 7 * blockDim.x];
		int c1 = g_idata[idx + 8 * blockDim.x];
		int c2 = g_idata[idx + 9 * blockDim.x];
		int c3 = g_idata[idx + 10 * blockDim.x];
		int c4 = g_idata[idx + 11 * blockDim.x];
		int d1 = g_idata[idx + 12 * blockDim.x];
		int d2 = g_idata[idx + 13 * blockDim.x];
		int d3 = g_idata[idx + 14 * blockDim.x];
		int d4 = g_idata[idx + 15 * blockDim.x];
		g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 + c1 + c2 + c3 + c4 + d1 + d2 + d3 + d4;

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

	reduceUnrolling16 <<< grid.x / 16, block >>>(d_idata, d_odata, size);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x / 16 * sizeof(int), cudaMemcpyDeviceToHost));

	int gpusum = 0;
	for (int i = 0; i < grid.x / 16; i++)  gpusum += h_odata[i];

	printf_s("gpu unrolling8 sum: %d <<<grid %d block %d >>>\n", gpusum, grid.x / 16, block.x);

	free(h_idata);
	free(h_odata);

	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));

	bResult = (gpusum == cpusum);
	if (!bResult) printf_s("Test failed\n");

	return EXIT_SUCCESS;

}