#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>

/*
* Various memory access pattern optimizations applied to a matrix transpose
* kernel.
*/

#define BDIMX 16
#define BDIMY 16

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
		printf("%dth element: %f\n", i, in[i]);
	}

	return;
}

void Result(float *hostRef, float *gpuRef, const int size, int showme)
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

		if (showme && i > size / 2 && i < size / 2 + 5)
		{
			// printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
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

__global__ void transposeUnroll8Row(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x * 8 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix; // access in rows
	unsigned int to = ix * ny + iy; // access in columns

	if (ix + 7 * blockDim.x < nx && iy < ny)
	{
		out[to] = in[ti];
		out[to + ny * blockDim.x] = in[ti + blockDim.x];
		out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
		out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
		out[to + ny * 4 * blockDim.x] = in[ti + 4 * blockDim.x];
		out[to + ny * 5 * blockDim.x] = in[ti + 5 * blockDim.x];
		out[to + ny * 6 * blockDim.x] = in[ti + 6 * blockDim.x];
		out[to + ny * 7 * blockDim.x] = in[ti + 7 * blockDim.x];
	}
}



// main functions
int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("%s starting transpose at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	cudaSetDevice(dev);

	// set up array size 2048
	int nx = 1 << 11;
	int ny = 1 << 11;

	// select a kernel and block siz
	int blockx = 16;
	int blocky = 16;


	if (argc > 1) blockx = atoi(argv[1]);

	if (argc > 2) blocky = atoi(argv[2]);

	if (argc > 3) nx = atoi(argv[3]);

	if (argc > 4) ny = atoi(argv[4]);

	printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny);
	size_t nBytes = nx * ny * sizeof(float);

	// execution configuration
	dim3 block(blockx, blocky);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	dim3 grid8((nx + block.x - 1) / (block.x * 8), (ny + block.y - 1) /	(block.y * 8));

	// allocate host memory
	float *h_A = (float *)malloc(nBytes);
	float *hostRef = (float *)malloc(nBytes);
	float *gpuRef = (float *)malloc(nBytes);

	// initialize host array
	initialData(h_A, nx * ny);

	// transpose at host side
	transposeHost(hostRef, h_A, nx, ny);

	// allocate device memory
	float *d_A, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	// copy data from host to device
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

	// transposeUnroll8Row

	transposeUnroll8Row << <grid8, block >> >(d_C, d_A, nx, ny);
	cudaDeviceSynchronize();

	printf("transposeUnroll8Row elapsed <<< grid (%d,%d) block (%d,%d)>>>\n",grid.x, grid.y, block.x, block.y);
	cudaGetLastError();
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	Result(hostRef, gpuRef, nx * ny, 1);
	


	// free host and device memory
	cudaFree(d_A);
	cudaFree(d_C);
	free(h_A);
	free(hostRef);
	free(gpuRef);

	// reset device
	(cudaDeviceReset());
	return EXIT_SUCCESS;
}
