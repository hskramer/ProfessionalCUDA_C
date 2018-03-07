#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define BDIMX	32
#define BDIMY	16
#define PAD		1

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

void printData(char *msg, int *data, int size)
{
	printf_s("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf_s("%5d", data[i]);
	}

	printf_s("\n");
	return;
}

__global__ void setColReadRowPad(int *out)
{
	// declare static shared memory
	__shared__ int tile[BDIMX][BDIMY + PAD];

	// global index
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// convert idx to transposed (row, col)
	unsigned int irow = idx / blockDim.y;
	unsigned int icol = idx % blockDim.y;

	// shared memory store (coalesced)
	tile[threadIdx.x][threadIdx.y] = idx;

	// wait for all threads to complete
	__syncthreads();

	// shared memory load (coalesced)
	out[idx] = tile[irow][icol];

}

int main(int argc, char **argv)
{
	int  dev = 0;
	cudaDeviceProp  deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("> %s starting on", argv[0]);
	printf_s("> device %d: %s\n\n", dev, deviceProp.name);

	// set device
	checkCuda(cudaSetDevice(dev));

	bool  iprintf = 0;
	if (argc > 1)  iprintf = atoi(argv[1]);

	// define matrix
	int  nx = BDIMX;
	int  ny = BDIMY;

	// calculate memory needed
	size_t  nBytes = nx * ny * sizeof(int);

	//allocate device memory
	int  *d_in;
	checkCuda(cudaMalloc(&d_in, nBytes));

	// allocate host memory
	int  *gpuRef = (int *)malloc(nBytes);

	// configure kernel
	dim3  block(BDIMX, BDIMY);
	dim3  grid(1, 1);

	printf_s("setColReadRowPad  <<<grid(%d, %d) block(%d,%d)>>>", grid.x, grid.y, block.x, block.y);
	setColReadRowPad << <grid, block >> > (d_in);
	checkCuda(cudaMemcpy(gpuRef, d_in, nBytes, cudaMemcpyDeviceToHost));


	if (iprintf)  printData("set column read row with pad", gpuRef, nx*ny);

	// free resouces
	checkCuda(cudaFree(d_in));
	free(gpuRef);

	// reset
	checkCuda(cudaDeviceReset());

	return  EXIT_SUCCESS;

}

