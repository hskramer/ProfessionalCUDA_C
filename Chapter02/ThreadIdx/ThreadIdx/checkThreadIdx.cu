#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// function for checkCudaing the CUDA runtime API results.
inline
void checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		printf("Error": %s : %d, ", __FILE__, __LINE__);
			printf("CUDA Runtime Error: %d: %s\n", result, cudaGetErrorString(result));
		exit(1);
	}
#endif
}


void printMatrix(int *C, const int nx, const int ny)
{
	int *ic = C;
	printf("\nMatrix: (%d,%d)\n", nx, ny);

	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%3d", ic[ix]);

		}

		ic += nx;
		printf("\n");
	}

	printf("\n");
	return;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index"	" %2d ival %2d\n", 
		threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	// get device information
	int dev = 0;
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// set matrix dimension
	int nx = 8;
	int ny = 6;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	// malloc host memory
	int *h_A;
	h_A = (int *)malloc(nBytes);

	// iniitialize host matrix with integer
	for (int i = 0; i < nxy; i++)
	{
		h_A[i] = i;
	}
	printMatrix(h_A, nx, ny);

	// malloc device memory
	int *d_MatA;
	checkCuda(cudaMalloc((void **)&d_MatA, nBytes));

	// transfer data from host to device
	checkCuda(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

	// set up execution configuration
	dim3 block(4, 2);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// invoke the kernel
	printThreadIndex <<<grid, block >>>(d_MatA, nx, ny);
	checkCuda(cudaGetLastError());

	// free host and devide memory
	checkCuda(cudaFree(d_MatA));
	free(h_A);

	// reset device
	checkCuda(cudaDeviceReset());

	return (0);
}