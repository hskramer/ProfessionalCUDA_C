#include <cuda_runtime.h>
#include <stdio.h>


// function for checking the CUDA runtime API results.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

/*
* Display the dimensionality of a thread block and grid from the host and
* device.
*/

__global__ void checkIndex(void)
{
	printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
	printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

	printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
	printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}

int main(int argc, char **argv)
{
	// define total data element
	int nElem = 36;

	// define grid and block structure
	dim3 block(3);
	dim3 grid((nElem + block.x - 1) / block.x);

	// check grid and block dimension from host side
	printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

	// check grid and block dimension from device side
	checkIndex << <grid, block >> >();

	// reset device before you leave
	checkCuda(cudaDeviceReset());

	return(0);
}
