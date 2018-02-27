#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/*
* Example kernels for transposing a rectangular host array using a variety of
* optimizations, including shared memory, unrolling, and memory padding.
*/


#define BDIMX 32
#define BDIMY 16

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))

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


void initialData(float *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		in[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void printData(float *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%3.0f ", in[i]);
	}

	printf("\n");
	return;
}

void checkResult(float *hostRef, float *gpuRef, int rows, int cols)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int index = INDEX(i, j, cols);
			if (abs(hostRef[index] - gpuRef[index]) > epsilon)
			{
				match = 0;
				printf("different on (%d, %d) (offset=%d) element in transposed matrix: host %f gpu %f\n", i, j, index,
					hostRef[index], gpuRef[index]);
				break;
			}
		}
		if (!match) break;
	}

	if (!match)  printf("Arrays do not match.\n\n");
}

void transposeHost(float *out, float *in, const int nrows, const int ncols)
{
	for (int iy = 0; iy < nrows; ++iy)
	{
		for (int ix = 0; ix < ncols; ++ix)
		{
			out[INDEX(ix, iy, nrows)] = in[INDEX(iy, ix, ncols)];
		}
	}
}

__global__ void copyGmem(float *out, float *in, const int nrows, const int ncols)
{
	// matrix coordinate (ix,iy)
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	// transpose with boundary test
	if (row < nrows && col < ncols)
	{
		// NOTE this is a transpose, not a copy
		out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
	}
}

__global__ void naiveGmem(float *out, float *in, const int nrows, const int ncols)
{
	// matrix coordinate (ix,iy)
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	// transpose with boundary test
	if (row < nrows && col < ncols)
	{
		out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
	}
}

__global__ void naiveGmemUnroll(float *out, float *in, const int nrows, const int ncols)
{
	// Pretend there are twice as many blocks in the x direction
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

	if (row < nrows)
	{
		if (col < ncols)
		{
			out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
		}

		col += blockDim.x;

		if (col < ncols)
		{
			out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
		}
	}
}

__global__ void transposeSmem(float *out, float *in, int nrows, int ncols)
{
	// static shared memory
	__shared__ float tile[BDIMY][BDIMX];

	// coordinate in original matrix
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);

	if (row < nrows && col < ncols)
	{
		// load data from global memory to shared memory
		tile[threadIdx.y][threadIdx.x] = in[offset];
	}

	// thread index in transposed block
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// NOTE - need to transpose row and col on block and thread-block level:
	// 1. swap blocks x-y
	// 2. swap thread x-y assignment (irow and icol calculations above)
	// note col still has continuous threadIdx.x -> coalesced gst
	col = blockIdx.y * blockDim.y + icol;
	row = blockIdx.x * blockDim.x + irow;

	// linear global memory index for transposed matrix
	// NOTE nrows is stride of result, row and col are transposed
	unsigned int transposed_offset = INDEX(row, col, nrows);
	// thread synchronization
	
	__syncthreads();

	// NOTE invert sizes for write check
	if (row < ncols && col < nrows)
	{
		// store data to global memory from shared memory
		out[transposed_offset] = tile[icol][irow]; // NOTE icol,irow not irow,icol
	}
}

__global__ void transposeSmemUnroll(float *out, float *in, const int nrows, const int ncols)
{
	// static 1D shared memory
	__shared__ float tile[BDIMY][BDIMX * 2];

	// coordinate in original matrix
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int row2 = row;
	unsigned int col2 = col + blockDim.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);
	unsigned int offset2 = INDEX(row2, col2, ncols);

	// thread index in transposed block
	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	// linear global memory index for transposed matrix
	unsigned int transposed_offset = INDEX(col, row, nrows);
	unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

	if (row < nrows && col < ncols)
	{
		tile[threadIdx.y][threadIdx.x] = in[offset];
	}
	if (row2 < nrows && col2 < ncols)
	{
		tile[threadIdx.y][blockDim.x + threadIdx.x] = in[offset2];
	}

	__syncthreads();

	if (row < nrows && col < ncols)
	{
		out[transposed_offset] = tile[irow][icol];
	}
	if (row2 < nrows && col2 < ncols)
	{
		out[transposed_offset2] = tile[irow][blockDim.x + icol];
	}
}

__global__ void transposeSmemUnrollPad(float *out, float *in, const int nrows, const int ncols)
{
	// static 1D shared memory with padding
	__shared__ float tile[BDIMY][BDIMX * 2 + IPAD];

	// coordinate in original matrix
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int row2 = row;
	unsigned int col2 = col + blockDim.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);
	unsigned int offset2 = INDEX(row2, col2, ncols);

	// thread index in transposed block
	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	// linear global memory index for transposed matrix
	unsigned int transposed_offset = INDEX(col, row, nrows);
	unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

	if (row < nrows && col < ncols)
	{
		tile[threadIdx.y][threadIdx.x] = in[offset];
	}
	if (row2 < nrows && col2 < ncols)
	{
		tile[threadIdx.y][blockDim.x + threadIdx.x] = in[offset2];
	}

	__syncthreads();

	if (row < nrows && col < ncols)
	{
		out[transposed_offset] = tile[irow][icol];
	}
	if (row2 < nrows && col2 < ncols)
	{
		out[transposed_offset2] = tile[irow][blockDim.x + icol];
	}
}

__global__ void transposeSmemUnrollPadDyn(float *out, float *in, const int nrows, const int ncols)
{
	// dynamic shared memory
	extern __shared__ float tile[];

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int row2 = row;
	unsigned int col2 = col + blockDim.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);
	unsigned int offset2 = INDEX(row2, col2, ncols);

	// thread index in transposed block
	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	unsigned int transposed_offset = INDEX(col, row, nrows);
	unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

	if (row < nrows && col < ncols)
	{
		tile[INDEX(threadIdx.y, threadIdx.x, BDIMX * 2 + IPAD)] = in[offset];
	}
	if (row2 < nrows && col2 < ncols)
	{
		tile[INDEX(threadIdx.y, blockDim.x + threadIdx.x, BDIMX * 2 + IPAD)] =
			in[offset2];
	}

	__syncthreads();

	if (row < nrows && col < ncols)
	{
		out[transposed_offset] = tile[INDEX(irow, icol, BDIMX * 2 + IPAD)];
	}
	if (row2 < nrows && col2 < ncols)
	{
		out[transposed_offset2] = tile[INDEX(irow, blockDim.x + icol, BDIMX * 2 + IPAD)];
	}
}

__global__ void transposeSmemPad(float *out, float *in, int nrows, int ncols)
{
	// static shared memory with padding
	__shared__ float tile[BDIMY][BDIMX + IPAD];

	// coordinate in original matrix
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);

	// thread index in transposed block
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// linear global memory index for transposed matrix
	unsigned int transposed_offset = INDEX(col, row, nrows);

	// transpose with boundary test
	if (row < nrows && col < ncols)
	{
		// load data from global memory to shared memory
		tile[threadIdx.y][threadIdx.x] = in[offset];

		// thread synchronization
		__syncthreads();

		// store data to global memory from shared memory
		out[transposed_offset] = tile[irow][icol];
	}
}

__global__ void transposeSmemDyn(float *out, float *in, int nrows, int ncols)
{
	// dynamic shared memory
	extern __shared__ float tile[];

	// coordinate in original matrix
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);

	// thread index in transposed block
	unsigned int row_idx, col_idx, irow, icol;
	row_idx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = row_idx / blockDim.y;
	icol = row_idx % blockDim.y;
	col_idx = irow * blockDim.x + icol;

	// linear global memory index for transposed matrix
	unsigned int transposed_offset = INDEX(col, row, nrows);

	// transpose with boundary test
	if (row < nrows && col < ncols)
	{
		// load data from global memory to shared memory
		tile[row_idx] = in[offset];

		// thread synchronization
		__syncthreads();

		// store data to global memory from shared memory
		out[transposed_offset] = tile[col_idx];
	}
}

__global__ void transposeSmemPadDyn(float *out, float *in, int nrows, int ncols)
{
	// static shared memory with padding
	extern __shared__ float tile[];

	// coordinate in original matrix
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);

	// thread index in transposed block
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
	unsigned int irow = idx / blockDim.y;
	unsigned int icol = idx % blockDim.y;
	unsigned int col_idx = irow * (blockDim.x + IPAD) + icol;

	// linear global memory index for transposed matrix
	unsigned int transposed_offset = INDEX(col, row, nrows);

	// transpose with boundary test
	if (row < nrows && col < ncols)
	{
		// load data from global memory to shared memory
		tile[row_idx] = in[offset];

		// thread synchronization
		__syncthreads();

		// store data to global memory from shared memory
		out[transposed_offset] = tile[col_idx];
	}
}

int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting transpose at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	bool iprint = 0;

	// set up array size 2048
	int nrows = 1 << 12;
	int ncols = 1 << 12;

	if (argc > 1) iprint = atoi(argv[1]);

	if (argc > 2) nrows = atoi(argv[2]);

	if (argc > 3) ncols = atoi(argv[3]);

	printf(" with matrix nrows %d ncols %d\n", nrows, ncols);
	size_t ncells = nrows * ncols;
	size_t nBytes = ncells * sizeof(float);

	// execution configuration
	dim3 block(BDIMX, BDIMY);
	/*
	* Map CUDA blocks/threads to output space. Map rows in output to same
	* x-value in CUDA, columns to same y-value.
	*/
	dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
	dim3 grid2((grid.x + 2 - 1) / 2, grid.y);

	// allocate host memory
	float *h_A = (float *)malloc(nBytes);
	float *hostRef = (float *)malloc(nBytes);
	float *gpuRef = (float *)malloc(nBytes);

	//  initialize host array
	initialData(h_A, nrows * ncols);

	//  transpose at host side
	transposeHost(hostRef, h_A, nrows, ncols);

	// allocate device memory
	float *d_A, *d_C;
	checkCuda(cudaMalloc(&d_A, nBytes));
	checkCuda(cudaMalloc(&d_C, nBytes));

	// copy data from host to device
	checkCuda(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

	copyGmem << <grid, block >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, nrows * ncols);
	printf("copyGmem    <<< grid (%d,%d) block (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	naiveGmem << <grid, block >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);

	printf("naiveGmem   <<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);

	naiveGmemUnroll << <grid2, block >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	printf("naiveGmemUnroll   <<< grid (%d,%d) block (%d,%d)>>>\n", grid2.x, grid2.y, block.x, block.y);

	transposeSmem << <grid, block >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	printf("transposeSmem   <<< grid (%d,%d) block (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	transposeSmemPad << <grid, block >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	printf("transposeSmemPad   <<< grid (%d,%d) block (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	transposeSmemDyn << <grid, block, BDIMX*BDIMY * sizeof(float) >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	printf("transposeSmemDyn <<< grid (%d,%d) block (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	transposeSmemPadDyn << <grid, block, (BDIMX + IPAD) * BDIMY * sizeof(float) >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	printf("transposeSmemPadDyn   <<< grid (%d,%d) block (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	transposeSmemUnroll << <grid2, block >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	printf("transposeSmemUnroll    <<< grid (%d,%d) block (%d,%d)>>>\n", grid2.x, grid2.y, block.x, block.y);

	transposeSmemUnrollPad << <grid2, block >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	printf("transposeSmemUnrollPad     <<< grid (%d,%d) block(%d,%d)>>>\n", grid2.x, grid2.y, block.x, block.y);

	transposeSmemUnrollPadDyn << <grid2, block, >> >(d_C, d_A, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	printf("transposeSmemUnrollPadDyn    <<< grid (%d,%d) block (%d,%d)>>>\n", grid2.x, grid2.y, block.x, block.y);

	// free host and device memory
	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_C));
	free(h_A);
	free(hostRef);
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());
	return EXIT_SUCCESS;
}
