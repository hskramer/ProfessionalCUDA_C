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

__global__ void transposeRow(float *out, float *in, const int nx, const int ny)
{
	unsigned int  ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int  iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int  row = iy * gridDim.x * blockIdx.x + ix;

	if (row < ny)
	{
		int  rowBegin = row * nx;
		int  rowEnd = (row + 1) * nx;
		int  colIndex = row;

		for (int i = rowBegin; i < rowEnd; i++)
		{
			out[colIndex] = in[i];
			colIndex += nx;
		}
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

	// set array size
	int  nx = 1 << 11;
	int  ny = 1 << 11;

	// block size
	int  blockx = 16;
	int  blocky = 16;

	// setup for user entered parameters
	if (argc > 1)  nx = atoi(argv[1]);

	if (argc > 2)  ny = atoi(argv[2]);

	if (argc > 3)  blockx = atoi(argv[3]);

	if (argc > 4)  blocky = atoi(argv[4]);

	printf_s(" with matrix nx %d ny %d with kernel %d\n", nx, ny);


	// allocate host memory
	size_t  nBytes = nx * ny * sizeof(float);

	float  *h_in	= (float *)malloc(nBytes);
	float  *hostRef = (float *)malloc(nBytes);
	float  *gpuRef  = (float *)malloc(nBytes);

	// allocate device memory
	float  *d_in, *d_out;
    checkCuda(cudaMalloc((float **)&d_in, nBytes));
	checkCuda(cudaMalloc((float **)&d_out, nBytes));

	// initialize matrix
	initialData(h_in, nx * ny);

	// transpose host data 
	transposeHost(hostRef, h_in, nx, ny);

	// copy host date to device
	checkCuda(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

	// set kernel parameters
	dim3  block(blockx, blocky);
	dim3  grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// warmup kernel
	warmup <<<grid, block >>> (d_out, d_in, nx, ny);
	checkCuda(cudaDeviceSynchronize());

	// launch kernel
	transposeRow <<<grid, block >>> (d_out, d_in, nx, ny);
	checkCuda(cudaDeviceSynchronize());

	// transfer back to host
	checkCuda(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

	// check result
	checkResult(hostRef, gpuRef, nx * ny);

	// free memory
	checkCuda(cudaFree(d_in));
	checkCuda(cudaFree(d_out));

	free(h_in);
	free(hostRef);
	free(gpuRef);

	return EXIT_SUCCESS;
	
}
