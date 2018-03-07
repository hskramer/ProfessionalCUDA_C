#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define BDIMX	32
#define BDIMY	32
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

void printData(char *msg, int *in, const int size)
{
	printf_s("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf_s("%5d", in[i]);
	}

	printf_s("\n");
	return;
}

__global__ void  setColReadRowPad(int *out)
{
	// statically allocate shared memory
	__shared__ int tile[BDIMX][BDIMY + PAD];

	// global memory index
	int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// shared memory store padding the column will result in coalesced stores
	tile[threadIdx.x][threadIdx.y] = idx;

	// wait for all threads to finish
	__syncthreads();

	// read shared memory by row this creates coalesced loads
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

int main(int argc, char **argv)
{
	int  dev = 0;
	cudaDeviceProp  deviceProp;

	// get device information
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("> %s starting on ", argv[0]);
	printf("device %d: %s\n\n", dev, deviceProp.name);

	checkCuda(cudaSetDevice(dev));

	bool iprintf = 0;

	if (argc > 1)	iprintf = atoi(argv[1]);

	// define matrix 32x32 = 1024 total elements
	int  nx = BDIMX;
	int  ny = BDIMY;

	size_t  nBytes = nx * ny * sizeof(int);

	// allocate  memory
	int  *d_in;
	checkCuda(cudaMalloc(&d_in, nBytes));

	int  *gpuRef = (int *)malloc(nBytes);

	// define kernel configuration
	dim3  block(BDIMX, BDIMY);
	dim3  grid(1, 1);

	printf_s("> setColReadRowPad  <<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
	setColReadRowPad <<<grid, block>>> (d_in);
	checkCuda(cudaMemcpy(gpuRef, d_in, nBytes, cudaMemcpyDeviceToHost));
	
	if (iprintf)  printData("set column read row", gpuRef, nx * ny);

	// free memory
	checkCuda(cudaFree(d_in));
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}