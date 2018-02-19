#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


/*
* This example demonstrates the use of zero-copy memory to remove the need to
* explicitly issue a memcpy operation between the host and device. By mapping
* host, page-locked memory into the device's address space, the address can
* directly reference a host array and transfer its contents over the PCIe bus.
*
* This example compares performing a vector addition with and without zero-copy
* memory.
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


void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
				gpuRef[i], i);
			break;
		}
	}

	return;
}

void initialData(float *ip, int size)
{
	int i;

	for (i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void sumArraysOnHost(float *a, float *b, float *c, const int N)
{
	for (int idx = 0; idx < N; idx++)
	{
		c[idx] = a[idx] + b[idx];
	}
}

__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	checkCuda(cudaSetDevice(dev));

	// get device properties
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));

	// check if support mapped memory
	if (!deviceProp.canMapHostMemory)
	{
		printf("Device %d does not support mapping CPU host memory!\n", dev);
		checkCuda(cudaDeviceReset());
		exit(EXIT_SUCCESS);
	}

	printf("Using Device %d: %s ", dev, deviceProp.name);

	// set up data size of vectors
	int ipower = 12;

	if (argc > 1) ipower = atoi(argv[1]);

	int nElem = 1 << ipower;
	size_t nBytes = nElem * sizeof(float);

	if (ipower < 18)
	{
		printf("Vector size %d power %d  nbytes  %3.0f KB\n", nElem, ipower,
			(float)nBytes / (1024.0f));
	}
	else
	{
		printf("Vector size %d power %d  nbytes  %3.0f MB\n", nElem, ipower,
			(float)nBytes / (1024.0f * 1024.0f));
	}

	// part 1: using device memory
	// malloc host memory
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_A, nElem);
	initialData(h_B, nElem);

	// add vector at host side for result check
	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	// malloc device global memory
	float *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc((float**)&d_A, nBytes));
	checkCuda(cudaMalloc((float**)&d_B, nBytes));
	checkCuda(cudaMalloc((float**)&d_C, nBytes));

	// transfer data from host to device
	checkCuda(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

	// set up execution configuration
	int iLen = 512;
	dim3 block(iLen);
	dim3 grid((nElem + block.x - 1) / block.x);

	sumArrays <<<grid, block >>>(d_A, d_B, d_C, nElem);

	// copy kernel result back to host side
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free device global memory
	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_B));

	// free host memory
	free(h_A);
	free(h_B);

	// part 2: using zerocopy memory for array A and B
	// allocate zerocpy memory
	checkCuda(cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped));
	checkCuda(cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped));

	// initialize data at host side
	initialData(h_A, nElem);
	initialData(h_B, nElem);

	// pass the pointer to device
	checkCuda(cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0));
	checkCuda(cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0));

	// add at host side for result checks
	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	// execute kernel with zero copy memory
	sumArraysZeroCopy <<<grid, block >>>(d_A, d_B, d_C, nElem);

	// copy kernel result back to host side
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free  memory
	checkCuda(cudaFree(d_C));
	checkCuda(cudaFreeHost(h_A));
	checkCuda(cudaFreeHost(h_B));

	free(hostRef);
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());
	return EXIT_SUCCESS;
}
