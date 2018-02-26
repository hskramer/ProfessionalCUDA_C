#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define BDIMY 32
#define BDIMX 16

#define IPAD 1

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
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

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
	__shared__ int  tile[BDIMY][BDIMX];

	// 2D thread index  to 1D global 
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// convert idx to transposed (row,col)
	unsigned int  irow = idx / blockDim.y;
	unsigned int  icol = idx % blockDim.y;

	// shared memory store op
	tile[threadIdx.y][threadIdx.x] = idx;

	// wait for threads to complete
	__syncthreads();

	// shared memory load/global write
	out[idx] = tile[icol][irow];


}
__global__ void setRowReadColDyn(int *out)
{
	// dynamic shared memory
	extern __shared__ int tile[];

	// mapping from thread index to global memory index
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// convert idx to transposed (row,col)
	unsigned int  irow = idx / blockDim.y;
	unsigned int  icol = idx % blockDim.y;
 
	// convert back to smem idx to access the transposed element
	unsigned int col_idx = icol * blockDim.x + irow;

	// store op
	tile[idx] = idx;

	// wait for threads to complete
	__syncthreads();

	// load op
	out[idx] = tile[col_idx];


}

__global__ void setRowReadColPad(int *out)
{
	__shared__ int tile[BDIMY][BDIMX + IPAD];

	// mapping from thread index to global memory index
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// convert idx to transposed (row,col)
	unsigned int  irow = idx / blockDim.y;
	unsigned int  icol = idx % blockDim.y;
	
	//shared memory store operation
	tile[threadIdx.y][threadIdx.x] = idx;

	// wait for threads to complete
	__syncthreads();

	// shared memory load operation
	out[idx] = tile[icol][irow];

}

__global__ void setRowReadColDynPad(int *out)
{
	// dynamic shared memory
	extern __shared__ int tile[];

	// mapping from thread index to global memory index
	unsigned int  idx = threadIdx.y * blockDim.x + threadIdx.x;

	// convert idx to transposed (row,col)
	unsigned int  irow = idx / blockDim.y;
	unsigned int  icol = idx % blockDim.y;

	unsigned int  row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;

	// convert  back to smem idx to access the transposed element
	unsigned int  col_idx = icol * (blockDim.x + IPAD) + irow;

	// shared memory store operation
	tile[row_idx] = idx;

	// wait for threads to complete
	__syncthreads();

	// shared memory load operation
	out[idx] = tile[col_idx];
}

int main(int argc, char **argv)
{
	// set device
	int  dev = 0;
	cudaDeviceProp  devProp;

	// get device info and set device
	checkCuda(cudaGetDeviceProperties(&devProp, dev));

	printf_s("%s  at ", argv[0]);
	printf_s("device %d: %s ", dev, devProp.name);
	checkCuda(cudaSetDevice(dev));

	//checkCuda(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

	// get device current state
	cudaSharedMemConfig  pConfig;
	checkCuda(cudaDeviceGetSharedMemConfig(&pConfig));
	printf_s("with Bank Mode:%s ", pConfig == 1 ? " 4-byte\n" : " 8-byte\n");

	// set up array size
	int  nx = BDIMX;
	int  ny = BDIMY;

	// allocate device memory
	size_t  nBytes = nx * ny * sizeof(int);

	int  *d_result;
	checkCuda(cudaMalloc(&d_result, nBytes));

	// configure kernel parameters
	dim3  block(BDIMX, BDIMY);
	dim3  grid(1, 1);

	printf_s("kernel: setRowReadRow\n");
	printf_s("<<<grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadRow <<<grid, block >>>(d_result);

	printf_s("kernel: setColReadCol\n");
	printf_s("<<<grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setColReadCol <<<grid, block >>> (d_result);

	printf_s("kernel: setRowReadCol\n");
	printf_s("<<<grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadCol <<<grid, block >>> (d_result);

	printf_s("kernel: setRowReadColDyn\n");
	printf_s("<<<grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadColDyn <<<grid, block, BDIMX * BDIMY * sizeof(int) >>> (d_result);

	printf_s("kernel: setRowReadColPad\n");
	printf_s("<<<grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadColPad <<<grid, block >>> (d_result);

	printf_s("kernel: setRowReadColDynPad\n");
	printf_s("<<<grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	setRowReadColDynPad <<<grid, block, (BDIMX + 1) * BDIMY * sizeof(int) >>> (d_result);

	checkCuda(cudaFree(d_result));

	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}