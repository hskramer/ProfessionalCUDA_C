#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define BDIMX	32
#define BDIMY	16

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

__global__ void setColReadRow(int *out)
{
	// declare static shared memory
	 __shared__ int tile[BDIMX][BDIMY];

	// mapping from thread index to global memory index
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	//convert idx to transposed coordinates (row, col)
	unsigned int  irow = idx / blockDim.y;
	unsigned int  icol = idx % blockDim.y;

	// store shared memory by column this will have bank conflicts
	tile[threadIdx.x][threadIdx.y] = idx;
	
	// wait for all threads to complete
	__syncthreads();

	// coalesced load by row from shared memory and save in global memory
	out[idx] = tile[irow][icol];

}

void printData(char *msg, int *in, const int size)
{
	printf("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf("%5d", in[i]);
	}

	printf("\n");
	return;
}

int main(int argc, char **argv)
{
	int  dev = 0;
	cudaDeviceProp  deviceProp;

	// get device information
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev)); 
	printf_s("> %s starting", argv[0]);
	printf_s("device %d: %s \n\n", dev, deviceProp.name);

	//set device
	checkCuda(cudaSetDevice(dev));

	// check for input 
	bool  iprintf = 0;

	if (argc > 1)	iprintf = atoi(argv[1]);

	// define matrix 
	int  nx = BDIMX;
	int  ny = BDIMY;

	// allocate  memory
	size_t  nBytes = nx * ny * sizeof(int);

	int  *gpuRef = (int *)malloc(nBytes);

	int  *d_in;
	checkCuda(cudaMalloc(&d_in, nBytes));

	// define kernel configuration
	dim3  block(BDIMX, BDIMY);
	dim3  grid(1, 1);

	printf_s("> setColReadRow  <<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
	setColReadRow <<<grid, block >>> (d_in);
	checkCuda(cudaMemcpy(gpuRef, d_in, nBytes, cudaMemcpyDeviceToHost));

	if (iprintf)  printData("set column read row", gpuRef, nx * ny);

	// free memory
	checkCuda(cudaFree(d_in));
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;


}