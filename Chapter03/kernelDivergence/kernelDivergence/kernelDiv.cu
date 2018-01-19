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


/*
* simpleDivergence demonstrates divergent code on the GPU and its impact on
* performance and CUDA metrics.
*/

__global__ void mathKernel1(float *c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;

	if (tid % 2 == 0)
	{
		ia = 100.0f;
	}
	else
	{
		ib = 200.0f;
	}

	c[tid] = ia + ib;
}

__global__ void mathKernel2(float *c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;

	if ((tid / warpSize) % 2 == 0)
	{
		ia = 100.0f;
	}
	else
	{
		ib = 200.0f;
	}

	c[tid] = ia + ib;
}

__global__ void mathKernel3(float *c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;

	bool ipred = (tid % 2 == 0);

	if (ipred)
	{
		ia = 100.0f;
	}

	if (!ipred)
	{
		ib = 200.0f;
	}

	c[tid] = ia + ib;
}

__global__ void mathKernel4(float *c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;

	int itid = tid >> 5;

	if (itid & 0x01 == 0)
	{
		ia = 100.0f;
	}
	else
	{
		ib = 200.0f;
	}

	c[tid] = ia + ib;
}

__global__ void warmingup(float *c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;

	if ((tid / warpSize) % 2 == 0)
	{
		ia = 100.0f;
	}
	else
	{
		ib = 200.0f;
	}

	c[tid] = ia + ib;
}


int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

	// set up data size
	int size = 64;
	int blocksize = 64;

	if (argc > 1) blocksize = atoi(argv[1]);

	if (argc > 2) size = atoi(argv[2]);

	printf("Data size %d ", size);

	// set up execution configuration
	dim3 block(blocksize, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);
	printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

	// allocate gpu memory
	float *d_C;
	size_t nBytes = size * sizeof(float);
	checkCuda(cudaMalloc((float**)&d_C, nBytes));

	// run a warmup kernel to remove overhead
	
	checkCuda(cudaDeviceSynchronize());

	warmingup << <grid, block >> >(d_C);
	checkCuda(cudaDeviceSynchronize());

	// run kernel 1

	mathKernel1 << <grid, block >> >(d_C);
	checkCuda(cudaDeviceSynchronize());


	// run kernel 2
	mathKernel2 << <grid, block >> >(d_C);
	checkCuda(cudaDeviceSynchronize());
	
	// run kernel 3
	mathKernel3 << <grid, block >> >(d_C);
	checkCuda(cudaDeviceSynchronize());

	// run kernel 4
	mathKernel4 << <grid, block >> >(d_C);
	checkCuda(cudaDeviceSynchronize());
	
	// free gpu memory and reset divece
	checkCuda(cudaFree(d_C));
	checkCuda(cudaDeviceReset());
	return 0;
}
