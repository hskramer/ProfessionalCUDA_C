#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


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

void initialData(float *ip, const int size)
{
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
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
			printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
				gpuRef[i]);
			break;
		}

	}

	if (!match)  printf("Arrays do not match.\n\n");
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

__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny)
{
	unsigned int  ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
	unsigned int  iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int  ti = iy * nx + ix;
	unsigned int  to = ix * ny + iy;

	if (iy < ny)
	{
		out[to]						  = in[ti];
		out[to + ny * blockDim.x]	  = in[ti + blockDim.x];
		out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
		out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
	}
}

__global__ void transposeUnroll8Row(float *out, float *in, const int nx, const int ny)
{
	unsigned int  ix = blockDim.x * blockIdx.x * 8 + threadIdx.x;
	unsigned int  iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int  ti = iy * nx + ix;
	unsigned int  to = ix * ny + iy;

	if (iy < ny)
	{
		out[to]						  = in[ti];
		out[to + ny * blockDim.x]	  = in[ti + blockDim.x];
		out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
		out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
		out[to + ny * 4 * blockDim.x] = in[ti + 4 * blockDim.x];
		out[to + ny * 5 * blockDim.x] = in[ti + 5 * blockDim.x];
		out[to + ny * 6 * blockDim.x] = in[ti + 6 * blockDim.x];
		out[to + ny * 7 * blockDim.x] = in[ti + 7 * blockDim.x];
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

	// set default array size to 2048x2048

	int  nx = 1 << 11;
	int  ny = 1 << 11;

	// block size
	int  blockx = 16;
	int  blocky = 16;

	// setup for user entered parameters

	if (argc > 1)  blockx = atoi(argv[1]);

	if (argc > 2)  blocky = atoi(argv[2]);

	if (argc > 3)  nx = atoi(argv[3]);

	if (argc > 4)  ny = atoi(argv[4]);

	printf_s(" with matrix nx %d ny %d with\n", nx, ny);

	// allocate host memory
	size_t  nBytes = nx * ny * sizeof(float);

	float *h_in = (float*)malloc(nBytes);
	float *hostRef = (float*)malloc(nBytes);
	float *gpuRef = (float*)malloc(nBytes);

	// allocate device memory
	float *d_in, *d_out;
	checkCuda(cudaMalloc(&d_in, nBytes));
	checkCuda(cudaMalloc(&d_out, nBytes));

	// intialize host matrix
	initialData(h_in, nx * ny);
	transposeHost(hostRef, h_in, nx, ny);

	// kernel launch configurations
	dim3  block(blockx, blocky);
	dim3  grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	dim3  grid4((nx + block.x - 1) / (block.x * 4), (ny + block.y - 1) / block.y);
	dim3  grid8((nx + block.x - 1) / (block.x * 8), (ny + block.y - 1) / block.y);


		// transfer host data to device
	checkCuda(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

	// launch warmup kernel
	warmup <<<grid, block >>> (d_out, d_in, nx, ny);
	checkCuda(cudaDeviceSynchronize());

	// Unroll4Row kernel
	transposeUnroll4Row <<<grid4, block >>> (d_out, d_in, nx, ny);
	checkCuda(cudaDeviceSynchronize());
	printf_s("transposeUnroll4Row  <<< grid (%d,%d) block (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	// transfer back to host
	checkCuda(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

	// check results
	checkResult(hostRef, gpuRef, nx * ny);

	transposeUnroll8Row <<<grid8, block >>> (d_out, d_in, nx, ny);
	checkCuda(cudaDeviceSynchronize());
	printf_s("transposeUnroll8Row  <<< grid (%d,%d) block (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	// transfer back to host
	checkCuda(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

	// check results
	checkResult(hostRef, gpuRef, nx * ny);

	checkCuda(cudaFree(d_in));
	checkCuda(cudaFree(d_out));
	free(h_in);
	free(hostRef);
	free(gpuRef);

	checkCuda(cudaDeviceReset());

	return	EXIT_SUCCESS;

}