#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define BDIMX	32
#define BDIMY	32

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
	printf("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf("%5d", in[i]);
	}

	printf("\n");
	return;
}


__global__ void setColReadRowDyn(int *out)
{
	// static shared memory
	extern __shared__ int tile[];

	// mapping from thread index to global memory index
	unsigned int  row_idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int  col_idx = threadIdx.x * blockDim.y + threadIdx.y;

	// store shared memory by column this will create 32 store bank conflicts
	tile[col_idx] = row_idx;

	// wait for all threads to finish
	__syncthreads();

	// read shared memory by row this creates coalesced loads. This results in a transpose of the matrix
	out[row_idx] = tile[row_idx];
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

	if (argc > 1) iprintf = atoi(argv[1]);

	// define matrix 32x32 = 1024 total elements
	int  nx = BDIMX;
	int  ny = BDIMY;

	size_t  nBytes = nx * ny * sizeof(int);

	// allocate  memory
	int  *d_in;
	checkCuda(cudaMalloc(&d_in, nBytes));

	int  *gpuRef = (int *)malloc(nBytes);

	// define kernel configurationff
	dim3  block(BDIMX, BDIMY);
	dim3  grid(1, 1);

	printf_s("> setColReadRowDyn  <<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
	setColReadRowDyn <<<grid, block, BDIMX*BDIMY*sizeof(int) >>> (d_in);
	checkCuda(cudaMemcpy(gpuRef, d_in, nBytes, cudaMemcpyDeviceToHost));

	if (iprintf)  printData("set column read row", gpuRef, nx * ny);

	// free memory
	checkCuda(cudaFree(d_in));
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}