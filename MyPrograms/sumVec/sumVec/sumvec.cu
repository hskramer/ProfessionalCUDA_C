#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

int cpusumRecurs(int *data, int const size)
{
	// stop condition
	if (size == 1) return data[0];

	// renew the stride
	int const stride = size / 2;

	// in place reduction
	for (int i = 0; i < stride; i++)
	{
		data[i] += data[i + stride];
	}

	// call recursively
	return cpusumRecurs(data, stride);
}


__global__ void gpuNeighborSum(int *g_idata, int *g_odata, unsigned int n)
{
	// set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// convert global data pointer into a local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x;
	
	if (idx >= n) return;

	// in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}

		// synchronize within threadblock
		__syncthreads();

	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(void)
{
	cudaDeviceProp deviceProp;

	int dev = 0;
	int nblock = 2048;
	int nthreads = 512;
	
	int size = 1<<24;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("starting reduction on device %s\n", deviceProp.name);
	printf_s("Array size: %d\n", size);

	dim3 block(nthreads, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);

	bool bResult = false;

	size_t	bytes = size * sizeof(int);

	// allocate host memory
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

	int cpusum = cpusumRecurs(tmp, size);
	printf_s("CPU sum is: %d\n", cpusum);

	// allocate device memory
	int *d_idata = NULL;
	int *d_odata = NULL;

	checkCuda(cudaMalloc((void **)&d_idata, bytes));
	checkCuda(cudaMalloc((void **)&d_odata, grid.x * sizeof(int)));

	checkCuda(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	gpuNeighborSum <<<grid, block >>> (d_idata, d_odata, size);

	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

	int gpusum = 0;
	for (int i = 0; i < grid.x; i++) gpusum += h_odata[i];
	printf_s("gpu sum: %d <<<grid %d, block %d>>>", gpusum, grid.x, block.x);

	free(h_idata);
	free(h_odata);

	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));

	bResult = (gpusum == cpusum);

	if (!bResult) printf_s("Test failed!\n");

	cudaDeviceReset();
	
	return EXIT_SUCCESS;

}