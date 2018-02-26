#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/* I typed most of these in from the versions given in the book the change
*  in performance of the kerneks using shared memory was negligable and
*  actually increased in some cases. This was probably due to the speed
*  of my global memory unlike the author my throughput never changed much
*  regardless of which version I used.
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
		printf_s("%3.0f ", in[i]);
	}

	printf_s("\n");
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
			if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
				match = 0;
				printf_s("different on (%d, %d) (offset=%d) element in transposed matrix: host %f gpu %f\n", i, j, index, hostRef[index], gpuRef[index]);
				break;
			}
		}
		if (!match) break;
	}

	if (!match)  printf_s("Arrays do not match.\n\n");
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


__global__ void copyGmem(float *out, float *in, const int nx, const int ny)
{
	// matrix coordinate(ix,iy)
	unsigned int  ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int  iy = blockIdx.y * blockDim.y + threadIdx.y;

	// used original kernel given in text the one provided in the download transposeRectangle is an exacr copy of naiveGmem kernel
	// which destroyed his idea of an upper/lower bound against which performance could be judged.
	if (ix < nx && iy < ny)
	{
		out[iy*nx + ix] = in[iy*nx + ix];
	}
}

__global__ void naiveGmem(float *out, float *in, const int nx, const int ny)
{
	// matrix coordinate (ix,iy)
	unsigned int  ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int  iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[ix*ny + iy] = in[iy*nx + ix];
	}
}

__global__ void transposeSmem(float *out, float *in, const int nx, const int ny)
{
	__shared__ float  tile[BDIMY][BDIMX];

	// original matrix coordinate (ix,iy)
	unsigned int  ix, iy, ti, to;

	ix = blockIdx.x * blockDim.x + threadIdx.x;
	iy = blockIdx.y * blockDim.y + threadIdx.y;

	// linear global memory index for original matrix
	ti = iy * nx + ix;

	// thread index in transposed block
	unsigned int  bidx, irow, icol;

	bidx = threadIdx.y*blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	unsigned int ix_col, iy_row;
	ix_col = blockIdx.y * blockDim.y + icol;
	iy_row = blockIdx.x * blockDim.x + irow;

	// linear
	to = iy_row * ny + ix_col;
	
	if (ix_col < nx && iy_row < ny)
	{
		tile[threadIdx.y][threadIdx.x] = in[ti];

		__syncthreads();

		out[to] = tile[icol][irow];
	}
	
}

__global__ void transposeSmemPad(float *out, float *in, const int nx, const int ny)
{
	// static 1D shared memory with padding
	__shared__ float tile[BDIMY * (BDIMX + IPAD)];

	// coordinate in original matrix
	unsigned int  ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int  iy = blockIdx.y * blockDim.y + threadIdx.y;

	// linear global memory index for original matrix
	unsigned int ti = iy * nx + ix;

	// thread index in transposed block
	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	unsigned int  ix2 = blockIdx.y * blockDim.y + icol;
	unsigned int  iy2 = blockIdx.x * blockDim.x + irow;

	// linear global memory index for transposed matrix
	unsigned int  to = iy2 * ny + ix2;

	if (ix + blockDim.x < nx && iy < ny)
	{
		// load one row from global memory to shared memory
		unsigned int  row_idx = threadIdx.y * (blockDim.x  + IPAD) + threadIdx.x;

		tile[row_idx] = in[ti];

		__syncthreads();

		// store one rowsto global memory from one column of shared memory
		unsigned int  col_idx = icol * (blockDim.x + IPAD) + irow;

		out[to] = tile[col_idx];
	}

	
}

__global__ void transposeSmemUnrollPad(float *out, float *in, const int nx, const int ny)
{
	// static 1D shared memory with padding
	__shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

	// coordinate in original matrix
	unsigned int  ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int  iy = blockIdx.y * blockDim.y + threadIdx.y;

	// linear global memory index for original matrix
	unsigned int ti   = iy * nx + ix;	
	
	// thread index in transposed block
	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	unsigned int  ix2 = blockIdx.y * blockDim.y + icol;
	unsigned int  iy2 = 2 * blockIdx.x * blockDim.x + irow;

	// linear global memory index for transposed matrix
	unsigned int  to = iy2 * ny + ix2;

	if (ix + blockDim.x < nx && iy < ny)
	{
		// load two rows from global memory to shared memory
		unsigned int  row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
		
		tile[row_idx]		  = in[ti];
		tile[row_idx + BDIMX] = in[ti + BDIMX];

		__syncthreads();

		// store two rows to global memory from two columns of shared memory
		unsigned int  col_idx = icol * (blockDim.x * 2 + IPAD) + irow;

		out[to] = tile[col_idx];
		out[to + ny * BDIMX] = tile[col_idx + BDIMX];
	}
}

__global__ void transposeSmemUnrollPadDyn(float *out, float *in, const int nx, const int ny)
{
	// static 1D shared memory with padding
	extern __shared__ float tile[];

	// coordinate in original matrix
	unsigned int  ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int  iy = blockIdx.y * blockDim.y + threadIdx.y;

	// linear global memory index for original matrix
	unsigned int ti = iy * nx + ix;

	// thread index in transposed block
	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	unsigned int  ix2 = blockIdx.y * blockDim.y + icol;
	unsigned int  iy2 = 2 * blockIdx.x * blockDim.x + irow;

	// linear global memory index for transposed matrix
	unsigned int  to = iy2 * ny + ix2;

	if (ix + blockDim.x < nx && iy < ny)
	{
		// load two rows from global memory to shared memory
		unsigned int  row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;

		tile[row_idx] = in[ti];
		tile[row_idx + BDIMX] = in[ti + BDIMX];

		__syncthreads();

		// store two rows to global memory from two columns of shared memory
		unsigned int  col_idx = icol * (blockDim.x * 2 + IPAD) + irow;

		out[to] = tile[col_idx];
		out[to + ny * BDIMX] = tile[col_idx + BDIMX];
	}
}

int main(int argc, char **argv)
{
	// set up device
	int  dev = 0;
	cudaDeviceProp  deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("%s starting transpose at ", argv[0]);
	printf_s("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	bool  iprint = 0;

	// set size of matrix
	int  nrows = 1 << 12;
	int  ncols = 1 << 12;

	if (argc > 1) iprint = atoi(argv[1]);

	if (argc > 2) nrows  = atoi(argv[2]);

	if (argc > 3) ncols  = atoi(argv[3]);

	printf_s(" with matrix nrows %d ncols %d\n", nrows, ncols);

	size_t  ncells = nrows * ncols;
	size_t  nBytes = ncells * sizeof(float);

	// set kernel parameters
	dim3  block(BDIMX, BDIMY);
	/*
	* Map CUDA blocks/threads to output space. Map rows in output to same
	* x-value in CUDA, columns to same y-value.
	*/
	dim3  grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
	dim3  grid2((grid.x + 2 - 1) / 2, grid.y);

	// allocate host memory
	float  *h_a		 = (float *)malloc(nBytes);
	float  *hostRef  = (float *)malloc(nBytes);
	float  *gpuRef	 = (float *)malloc(nBytes);

	// initialize host data
	initialData(h_a, nrows * ncols);

	//  transpose at host side
	transposeHost(hostRef, h_a, nrows, ncols);

	// allocate device memory
	float  *d_a, *d_result;
	checkCuda(cudaMalloc(&d_a, nBytes));
	checkCuda(cudaMalloc(&d_result, nBytes));

	//  copy kernel just for performance comparison
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	copyGmem <<<grid, block >>> (d_result, d_a, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_result, nBytes, cudaMemcpyDeviceToHost));
	
	if (iprint) printData(gpuRef, nrows * ncols);

	// kernels too fast for basic cpu timers have used MS high-resolution timers, but its easier to use NSIGHT 
	printf_s("copyGmem       <<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);

	// naive transpose
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	naiveGmem <<<grid, block >>> (d_result, d_a, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_result, nBytes, cudaMemcpyDeviceToHost));
	
	if (iprint) printData(gpuRef, nrows * ncols);

	printf_s("naiveGmem       <<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);
	checkResult(hostRef, gpuRef, nrows, ncols);

	// naive shared memory transpose
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	transposeSmem <<<grid, block >>> (d_result, d_a, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_result, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, nrows * ncols);

	printf_s("transposeSmem     <<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);
	checkResult(hostRef, gpuRef, nrows, ncols);
	
	// shared memory with  pad
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	transposeSmemPad <<<grid, block >>> (d_result, d_a, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_result, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, nrows * ncols);

	printf_s("transposeSmemPad    <<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x, grid.y, block.x, block.y);
	checkResult(hostRef, gpuRef, nrows, ncols);
	
	// shared memory unroll with pad
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	transposeSmemUnrollPad <<<grid2, block >>> (d_result, d_a, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_result, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, nrows * ncols);

	printf_s("transposeSmemUnrollPad  <<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x/2, grid.y, block.x, block.y);
	checkResult(hostRef, gpuRef, nrows, ncols);

	// dynamically allocated shared memory unroll with pad 
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	transposeSmemUnrollPadDyn <<<grid2, block, (BDIMX * 2 + IPAD) * BDIMY * sizeof(float) >>> (d_result, d_a, nrows, ncols);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(gpuRef, d_result, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, nrows * ncols);

	printf_s("transposeSmemUnrollPadDyn  <<< grid (%d,%d) block (%d,%d)>>>\n\n", grid.x/2, grid.y, block.x, block.y);
	checkResult(hostRef, gpuRef, nrows, ncols);

	// free memory
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_result));
	free(hostRef);
	free(gpuRef);
	free(h_a);

	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}