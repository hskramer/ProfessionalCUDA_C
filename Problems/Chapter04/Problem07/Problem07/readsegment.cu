#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double  epsilon = 1.0E-8;
	bool bResult = true;
	for (int i = 0; i < N; i++)
	{
		if (hostRef[i] - gpuRef[i] > epsilon)
		{
			printf_s("Arrays do not match.\n");
			printf_s("host: %5.2f  gpu: %5.2f at current location %d\n", hostRef[i], gpuRef[i], i);
			bResult = false;
			break;
		}
	}

	if (bResult)
	{
		printf_s("Arrays match\n");
		return;
	}
}

void initialData(float *ip, int size)
{
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 100.0f;
	}

	return;
}


void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset)
{
	for (int idx = offset, k = 0; idx < n; idx++, k++)
	{
		C[k] = A[idx] + B[idx];
	}
}

__global__ void warmup(float *A, float *B, float *C, const int n, int offset)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int k = i + offset;

	if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffset(float *A, float *B, float *C, const int n,
	int offset)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int k = i + offset;

	if (k < n) C[i] = A[k] + B[k];
}

int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting reduction at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// set up array size
	int nElem = 1 << 22; // total number of elements to reduce
	printf(" with array size %d\n", nElem);
	size_t nBytes = nElem * sizeof(float);

	// set up offset for summary
	int blocksize = 512;
	int offset = 0;

	if (argc > 1) offset = atoi(argv[1]);

	if (argc > 2) blocksize = atoi(argv[2]);

	// execution configuration
	dim3 block(blocksize, 1);
	dim3 grid((nElem + block.x - 1) / block.x, 1);

	// allocate host memory
	float *h_a		= (float *)malloc(nBytes);
	float *h_b		= (float *)malloc(nBytes);
	float *hostRef	= (float *)malloc(nBytes);
	float *gpuRef	= (float *)malloc(nBytes);

	//  initialize host array
	initialData(h_a, nElem);
	memcpy(h_b, h_a, nBytes);

	//  summary at host side
	sumArraysOnHost(h_a, h_b, hostRef, nElem, offset);

	// allocate device memory
	float *d_a, *d_b, *d_c;
	checkCuda(cudaMalloc((float**)&d_a, nBytes));
	checkCuda(cudaMalloc((float**)&d_b, nBytes));
	checkCuda(cudaMalloc((float**)&d_c, nBytes));

	// copy data from host to device
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_b, h_a, nBytes, cudaMemcpyHostToDevice));

	//  kernel 1:
	warmup <<<grid, block >>>(d_a, d_b, d_c, nElem, offset);
	checkCuda(cudaDeviceSynchronize());
	
	printf("warmup  <<< %4d, %4d >>> offset %4d\n", grid.x, block.x, offset);

	readOffset <<<grid, block >>>(d_a, d_b, d_c, nElem, offset);
	checkCuda(cudaDeviceSynchronize());

	printf("readOffset  <<< %4d, %4d >>> offset %4d\n", grid.x, block.x, offset);

	// copy kernel result back to host side and check device results
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nElem - offset);

	// free host and device memory
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));
	checkCuda(cudaFree(d_c));
	free(h_a);
	free(h_b);

	// reset device
	checkCuda(cudaDeviceReset());
	return EXIT_SUCCESS;
}