#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>
#include <cuda_runtime.h>

/*
* Various memory access pattern optimizations applied to a matrix transpose
* kernel.
*/

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
		in[i] = (float)(rand() & 0xFF) / 10.0f; //100.0f;
	}

	return;
}

void printData(float *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		printf_s("%dth element: %f\n", i, in[i]);
	}

	return;
}

void checkResult(float *hostRef, float *gpuRef, const int size)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < size; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf_s("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
			break;
		}

	}

	if (!match)  printf_s("Arrays do not match.\n\n");
}

void transposeHost(float *out, float *in, const int nx, const int ny)
{
	for (int iy = 0; iy < ny; ++iy)
	{
		for (int ix = 0; ix < nx; ++ix)
		{
			out[ix * ny + iy] = in[iy * nx + ix];
		}
	}
}

__global__ void warmup(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}

// case 0 copy kernel: access data in rows, simulates the same amount of memory  ops as transpose with coalesced access
__global__ void copyRow(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}

// case 1 copy kernel: access data in columns, simulates the same amount of memory  ops as transpose with strided access
__global__ void copyCol(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[ix * ny + iy];
	}
}

// case 2 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[iy * nx + ix];
	}
}

// case 3 transpose kernel: read in columns and write in rows
__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[ix * ny + iy];
	}
}

// case 4 transpose kernel: read in rows and write in columns + unroll 4 blocks
__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix; // access in rows
	unsigned int to = ix * ny + iy; // access in columns

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[to] = in[ti];
		out[to + ny * blockDim.x] = in[ti + blockDim.x];
		out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
		out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
	}
}

// case 5 transpose kernel: read in columns and write in rows + unroll 4 blocks
__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix; // access in rows
	unsigned int to = ix * ny + iy; // access in columns

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[ti] = in[to];
		out[ti + blockDim.x] = in[to + blockDim.x * ny];
		out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
		out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
	}
}

/*
* case 6 :  transpose kernel: read in rows and write in colunms + diagonal
* coordinate transform
*/
__global__ void transposeDiagonalRow(float *out, float *in, const int nx, const int ny)
{
	unsigned int blk_y = blockIdx.x;
	unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

	unsigned int ix = blockDim.x * blk_x + threadIdx.x;
	unsigned int iy = blockDim.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[iy * nx + ix];
	}
}

/*
* case 7 :  transpose kernel: read in columns and write in row + diagonal
* coordinate transform.
*/
__global__ void transposeDiagonalCol(float *out, float *in, const int nx, const int ny)
{
	unsigned int blk_y = blockIdx.x;
	unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

	unsigned int ix = blockDim.x * blk_x + threadIdx.x;
	unsigned int iy = blockDim.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[ix * ny + iy];
	}
}

// main functions
int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("%s starting transpose at ", argv[0]);
	printf_s("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// set up array size 2048
	int nx = 1 << 11;
	int ny = 1 << 11;

	// select a kernel and block size
	int iKernel = 0;
	int blockx = 16;
	int blocky = 16;

	if (argc > 1) iKernel = atoi(argv[1]);

	if (argc > 2) blockx = atoi(argv[2]);

	if (argc > 3) blocky = atoi(argv[3]);

	if (argc > 4) nx = atoi(argv[4]);

	if (argc > 5) ny = atoi(argv[5]);

	printf_s(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
	size_t nBytes = nx * ny * sizeof(float);

	// execution configuration
	dim3 block(blockx, blocky);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// allocate host memory
	float *h_A		= (float *)malloc(nBytes);
	float *hostRef	= (float *)malloc(nBytes);
	float *gpuRef	= (float *)malloc(nBytes);

	// initialize host array
	initialData(h_A, nx * ny);

	// transpose at host side
	transposeHost(hostRef, h_A, nx, ny);

	// allocate device memory
	float *d_A, *d_C;
	checkCuda(cudaMalloc((float**)&d_A, nBytes));
	checkCuda(cudaMalloc((float**)&d_C, nBytes));

	// copy data from host to device
	checkCuda(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

	// warmup to avoide startup overhead
	clock_t	iStart, iStop;
	iStart = clock();

	warmup << <grid, block >> >(d_C, d_A, nx, ny);
	checkCuda(cudaDeviceSynchronize());
	iStop = clock();
	float iElaps = float(iStop - iStart) / CLOCKS_PER_SEC;
	printf_s("warmup         elapsed %f sec\n", iElaps);
	checkCuda(cudaGetLastError());

	// kernel pointer and descriptor
	void(*kernel)(float *, float *, int, int);
	char *kernelName;

	// set up kernel
	switch (iKernel)
	{
	case 0:
		kernel = &copyRow;
		kernelName = "CopyRow       ";
		break;

	case 1:
		kernel = &copyCol;
		kernelName = "CopyCol       ";
		break;

	case 2:
		kernel = &transposeNaiveRow;
		kernelName = "NaiveRow      ";
		break;

	case 3:
		kernel = &transposeNaiveCol;
		kernelName = "NaiveCol      ";
		break;

	case 4:
		kernel = &transposeUnroll4Row;
		kernelName = "Unroll4Row    ";
		grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
		break;

	case 5:
		kernel = &transposeUnroll4Col;
		kernelName = "Unroll4Col    ";
		grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
		break;

	case 6:
		kernel = &transposeDiagonalRow;
		kernelName = "DiagonalRow   ";
		break;

	case 7:
		kernel = &transposeDiagonalCol;
		kernelName = "DiagonalCol   ";
		break;
	}

	// run kernel
	iStart = clock();
	kernel << <grid, block >> >(d_C, d_A, nx, ny);
	checkCuda(cudaDeviceSynchronize());
	iStop = clock();
    iElaps = float(iStop - iStart) / CLOCKS_PER_SEC;
	printf_s("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> \n", kernelName, iElaps, grid.x, grid.y, block.x, block.y);
	checkCuda(cudaGetLastError());

	// check kernel results
	if (iKernel > 1)
	{
		checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
		checkResult(hostRef, gpuRef, nx * ny);
	}

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
