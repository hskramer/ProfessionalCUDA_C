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

// Recursive Implementation of Interleaved Pair Approach
double cpuSumRecurse(double* data, int const size)
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
	return cpuSumRecurse(data, stride);
}

// sum reduction kernel for doubles
__global__ void sumReductionDoubles8(double* g_idata, double* g_odata, unsigned int n)
{
	// set thread ID
	unsigned int  tid = threadIdx.x;
	unsigned int  idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	// convert global data pointer to the local pointer of this block
	double* idata = g_idata + blockIdx.x * blockDim.x * 8;
	
	// unrolling 8
	if (idx + 7 * blockDim.x < n)
	{
		double  a1 = g_idata[idx];
		double  a2 = g_idata[idx + blockDim.x];
		double  a3 = g_idata[idx + 2 * blockDim.x];
		double  a4 = g_idata[idx + 3 * blockDim.x];
		double  b1 = g_idata[idx + 4 * blockDim.x];
		double  b2 = g_idata[idx + 5 * blockDim.x];
		double  b3 = g_idata[idx + 6 * blockDim.x];
		double  b4 = g_idata[idx + 7 * blockDim.x];

		g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
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
	int  dev = 0;
	cudaDeviceProp  deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));

	printf_s("\n%s starting reduction at ", argv[0]);
	printf_s("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	bool  bResult = false;

	int  size = 1 << 24;    // to reduce a vector of this size required the use of double in place of float don't know why the solution has floats
							// and a vector of this size with and a max element of 255 this exceeds the MAX float 

	printf_s("    with array size %d  ", size);

	// execution configuration
	int blocksize = 512;   // initial block size

	if (argc > 1)
	{
		blocksize = atoi(argv[1]);   // block size from command line argument
	}

	dim3  block(blocksize, 1);
	dim3  grid((size + blocksize - 1) / block.x);
	printf_s("grid %d block %d\n", grid.x, block.x);

	// allocate host memory
	size_t  bytes = size * sizeof(double);

	double*  h_idata = (double*)malloc(bytes);
	double*  h_odata = (double*)malloc(bytes);
	double*  tmp = (double*)malloc(bytes);

	// fill host array with random doubles
	time_t  t;
	srand(time(&t));

	for (int i = 0; i < size; i++)
	{
		h_idata[i] = rand() % 25;
	}

	memcpy(tmp, h_idata, bytes);

	// allocate device memory
	double*  d_idata = NULL;
	double*  d_odata = NULL;
	checkCuda(cudaMalloc((void** ) &d_idata, bytes));
	checkCuda(cudaMalloc((void** ) &d_odata, bytes));

	double  cpusum = 0.0f;
	cpusum = cpuSumRecurse(tmp, size);
	printf_s("cpu sum: %f\n", cpusum);

	checkCuda(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaDeviceSynchronize());

	sumReductionDoubles8 <<<grid.x / 8, block >>> (d_idata, d_odata, size);
	checkCuda(cudaDeviceSynchronize());

	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(double), cudaMemcpyDeviceToHost));

	double gpusum = 0.0f;
	for (int i = 0; i < grid.x / 8; i++)  gpusum += h_odata[i];

	printf_s("gpu sum: %f <<<grid %d block %d >>>\n", gpusum, grid.x / 8, block.x);
	
	free(h_idata);
	free(h_odata);
	free(tmp);

	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));

	bResult = (gpusum == cpusum);
	if (!bResult) printf_s("Test failed\n");

	return EXIT_SUCCESS;

}