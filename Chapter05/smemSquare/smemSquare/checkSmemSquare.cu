#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define	BDIMX	32
#define BDIMY	32
#define IPAD	1

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


__global__ void setRowReadRow(int *out)
{
	// static shared memory
	__shared__ int tile[BDIMY][BDIMX];

	// mapping from thread index to global memory index
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// shared memory store operation
	tile[threadIdx.y][threadIdx.x] = idx;

	// wait for all threads to complete
	__syncthreads();

	// shared memory load operation
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out)
{
	// static shared memory
	__shared__ int tile[BDIMX][BDIMY];

	// mapping from thread index to global memory index
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// shared memory store operation
	tile[threadIdx.x][threadIdx.y] = idx;

	// wait for all threads to complete
	__syncthreads();

	// shared memory load operation
	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out)
{
	// static shared memory
	__shared__ int tile[BDIMY][BDIMX];

	// mapping from thread index to global memory index
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// shared memory store operation
	tile[threadIdx.y][threadIdx.x] = idx;

	// wait for all threads to complete
	__syncthreads();

	// shared memory load operation
	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColPad(int *out)
{
	// static shared memory the padding prevents bank conflicts
	// which results in coalesced loads
	__shared__ int  tile[BDIMX][BDIMY + IPAD];

	// mapping from thread index to global memory offset
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// shared memory store operation
	tile[threadIdx.y][threadIdx.x] = idx;

	// wait for all threads to complete
	__syncthreads();

	// shared memory load operation
	out[idx] = tile[threadIdx.x][threadIdx.y];
}


__global__ void setRowReadColDyn(int *out)
{
	// dynamic shared memory
	extern __shared__ int tile[];

	// mapping from thread index to global memory index
	unsigned int  row_idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int  col_idx = threadIdx.x * blockDim.y + threadIdx.y;

	// shared memory store operation
	tile[row_idx] = row_idx;

	// wait for all threads to complete
	__syncthreads();

	// shared memory load operation
	out[row_idx] = tile[col_idx];
}


__global__ void setRowReadColDynPad(int *out)
{
	// dynamic  shared memory
	extern  __shared__ int tile[];

	// mapping from thread index to global memory index
	unsigned int  row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
	unsigned int  col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;;
	unsigned int  g_idx   = threadIdx.y * blockDim.x + threadIdx.x;

	// shared memory store operation
	tile[row_idx] = g_idx;

	// wait for all threads to finish
	__syncthreads();

	// shared memory load operation
	out[g_idx] = tile[col_idx];
}

int main(int argc, char **argv)
{

	int  dev = 0;
	cudaDeviceProp  devProp;

	// get device information
	checkCuda(cudaGetDeviceProperties(&devProp, dev));

	printf_s(" %s starting shared memory access ", argv[0]);
	printf_s("device %d: %s ", dev, devProp.name);
	checkCuda(cudaSetDevice(dev));

	// get device current state
	cudaSharedMemConfig  pConfig;
	checkCuda(cudaDeviceGetSharedMemConfig(&pConfig));
	printf_s("Bank Mode:%s ", pConfig == 1 ? "4-byte\n" : "8-byte\n");

	// set array 
	int  nx = BDIMX;
	int  ny = BDIMX;

	
	// allocate memory 
	size_t  nBytes = nx * ny * sizeof(int);

	int *d_result;
	checkCuda(cudaMalloc(&d_result, nBytes));

	int *gpuRef = (int *)malloc(nBytes);

	// configure kernel
	dim3 block(BDIMX, BDIMY);
	dim3 grid(1, 1);

	printf_s("kernel: setRowReadRow\n");
	printf_s("<<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadRow <<<grid, block >>> (d_result);

	printf_s("kernel: setColReadCol\n");
	printf_s("<<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setColReadCol <<<grid, block >>> (d_result);

	printf_s("kernel: setRowReadCol\n");
	printf_s("<<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadCol <<<grid, block >>> (d_result);

	printf_s("kernel:setRowReadColPad \n");
	printf_s("<<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadColPad <<<grid, block >>> (d_result);

	printf_s("kernel:setRowReadColDyn \n");
	printf_s("<<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadColDyn <<<grid, block, BDIMX * BDIMY * sizeof(int) >>> (d_result);

	printf_s("kernel:setRowReadColDynPad \n");
	printf_s("<<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadColDynPad <<<grid, block, BDIMX * (BDIMY + 1) * sizeof(int) >>> (d_result);

	checkCuda(cudaFree(d_result));
	free(gpuRef);

	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}