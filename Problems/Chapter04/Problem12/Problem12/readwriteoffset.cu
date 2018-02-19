#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* Will have to build and test on my ubuntu machine with a 3.5 compute card pascal cards 32 byte
* read/write too tolerant of offsets to show significant difference.
*/


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


void checkResult(float *hostRef, float *gpuRef, int offset, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = offset; i < N; i++)
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

void initialData(float *ip, int size)
{
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 100.0f;
	}

	return;
}

void sumArraysOnHost(float *a, float *b, float *c, int offset, const int N)
{
	for (int idx = offset; idx < N; idx++)
	{
		c[idx] = a[idx] + b[idx];
	}
}

__global__ void warmup(float *a, float *b, float *c, int offset, const int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int k = i + offset;

	if (k < N)  c[k] = a[k] + b[k];
}

__global__ void writeOffset(float *a, float *b, float *c, int offset, const int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int k = i + offset;

	if (k < N) c[k] = a[i] + b[i];
}

__global__ void readOffset(float *a, float *b, float *c, int offset, const int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int k = i + offset;

	if (k < N) c[i] = a[k] + a[k];
}

__global__ void readWriteOffset(float *a, float *b, float *c, int offset, const int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int k = i + offset;

	if (k < N)  c[k] = a[k] + b[k];
}

int main(int argc, char **argv)
{
	int  dev = 0;
	cudaDeviceProp  devprop;

	checkCuda(cudaGetDeviceProperties(&devprop, dev));

	printf_s("%s starting reduction at ", argv[0]);
	printf_s("device %d: %s ", dev, devprop.name);

	checkCuda(cudaSetDevice(dev));

	// set array size
	int  nElem = 1 << 22;
	printf_s(" with array size %d\n", nElem);

	// set offset
	int offset = 0;
	int blocksize = 512;

	if (argc > 1)  offset = atoi(argv[1]);

	if (argc > 2)  blocksize = atoi(argv[2]);

	size_t  nBytes = nElem * sizeof(float);

	// allocate host memory
	float  *h_a		= (float *)malloc(nBytes);
	float  *h_b		= (float *)malloc(nBytes);
	float  *hostRef = (float *)malloc(nBytes);
	float  *gpuRef	= (float *)malloc(nBytes);

	// initialize host array
	initialData(h_a, nElem);
	memcpy(h_b, h_a, nBytes);
	
	// summary at host side
	sumArraysOnHost(h_a, h_b, hostRef, offset, nElem);

	// kernel configuration
	dim3  block(blocksize, 1);
	dim3  grid((nElem + block.x - 1) / block.x);

	// allocate device memory
	float  *d_a, *d_b, *d_c;
	checkCuda(cudaMalloc((float**)&d_a, nBytes));
	checkCuda(cudaMalloc((float**)&d_b, nBytes));
	checkCuda(cudaMalloc((float**)&d_c, nBytes));

	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice));
	
	// warm up
	warmup <<<grid, block >>> (d_a, d_b, d_c, offset, nElem);
	checkCuda(cudaDeviceSynchronize());

	printf_s("Warmup         <<< %4d, %4d >>> offset %4d\n", grid.x, block.x, offset);

	// kernel 1: read offset
	readOffset <<<grid, block >>> (d_a, d_c, d_b, offset, nElem);
	checkCuda(cudaDeviceSynchronize());

	printf_s("ReadOffset  <<< %4d, %4d >>> offset %4d\n", grid.x, block.x, offset);

	// copy kernel result back to host side and check device results
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, offset, nElem);

	// kernel 2: write offset
	writeOffset<<<grid, block>>>(d_a, d_b, d_c, offset, nElem);
	checkCuda(cudaDeviceSynchronize());

	printf_s("WriteOffset  <<< %4d, %4d >>> offset %4d\n", grid.x, block.x, offset);

	// copy kernel result back to host side and check device results
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, offset, nElem);
	
	// kernel 3: read/write offset
	readWriteOffset<<<grid, block>>>(d_a, d_b, d_c, offset, nElem);
	checkCuda(cudaDeviceSynchronize());

	printf_s("readWriteOffset  <<< %4d, %4d >>> offset %4d\n", grid.x, block.x, offset);

	// copy kernel result back to host side and check device results
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, offset, nElem);

	// free memory
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));
	checkCuda(cudaFree(d_c));

	free(h_a);
	free(h_b);
	free(hostRef);
	free(gpuRef);

	checkCuda(cudaDeviceReset());

	return	EXIT_SUCCESS;

}