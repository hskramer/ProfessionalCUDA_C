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
* memory and unified memory .
*/


/* Unified memory performance was exellent about the same as using all device pointers 
*  without the explicit memory transfer zero-copy memory performed poorly, but still
*  will be needed on devices that don't support unified memory.
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
			printf_s("Arrays do not match!\n");
			printf_s("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
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

__global__ void sumArrays(float *a, float *b, float *c, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) c[i] = a[i] + b[i];
}

__global__ void sumArraysZeroCopy(float *a, float *b, float *c, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) c[i] = a[i] + b[i];
}

__global__ void sumArraysUnified(float *a, float *b, float *c, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) c[i] = a[i] + b[i];
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
		printf_s("Device %d does not support mapping CPU host memory!\n", dev);
		checkCuda(cudaDeviceReset());
		exit(EXIT_SUCCESS);
	}

	if (!deviceProp.managedMemory)
	{
		printf_s("Device %d does not support unified memory!\n", dev);
		checkCuda(cudaDeviceReset());
		exit(EXIT_SUCCESS);
	}

	printf_s("Using Device %d: %s ", dev, deviceProp.name);

	// set up data size of vectors
	int ipower = 12;

	if (argc > 1) ipower = atoi(argv[1]);

	int nElem = 1 << ipower;
	size_t nBytes = nElem * sizeof(float);

	if (ipower < 18)
	{
		printf_s("Vector size %d power %d  nbytes  %3.0f KB\n", nElem, ipower, (float)nBytes / (1024.0f));
	}
	else
	{
		printf_s("Vector size %d power %d  nbytes  %3.0f MB\n", nElem, ipower, (float)nBytes / (1024.0f * 1024.0f));
	}

	// part 1: using device memory
	// malloc host memory
	float *h_a, *h_b, *hostRef, *gpuRef;
	h_a		= (float *)malloc(nBytes);
	h_b		= (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef  = (float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_a, nElem);
	initialData(h_b, nElem);

	// add vector at host side for result check
	sumArraysOnHost(h_a, h_b, hostRef, nElem);

	// malloc device global memory
	float *d_a, *d_b, *d_c;
	checkCuda(cudaMalloc((float**)&d_a, nBytes));
	checkCuda(cudaMalloc((float**)&d_b, nBytes));
	checkCuda(cudaMalloc((float**)&d_c, nBytes));

	// transfer data from host to device
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice));

	// set up execution configuration
	int iLen = 512;
	dim3 block(iLen);
	dim3 grid((nElem + block.x - 1) / block.x);

	sumArrays <<<grid, block >>>(d_a, d_b, d_c, nElem);

	// copy kernel result back to host side
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free device global memory
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));

	// free host memory
	free(h_a);
	free(h_b);

	// part 2: using zerocopy memory for array A and B
	// allocate zerocpy memory
	checkCuda(cudaHostAlloc((void **)&h_a, nBytes, cudaHostAllocMapped));
	checkCuda(cudaHostAlloc((void **)&h_b, nBytes, cudaHostAllocMapped));

	// initialize data at host side
	initialData(h_a, nElem);
	initialData(h_b, nElem);

	// pass the pointer to device
	checkCuda(cudaHostGetDevicePointer((void **)&d_a, (void *)h_a, 0));
	checkCuda(cudaHostGetDevicePointer((void **)&d_b, (void *)h_b, 0));

	// add at host side for result checks
	sumArraysOnHost(h_a, h_b, hostRef, nElem);

	// execute kernel with zero copy memory
	sumArraysZeroCopy <<<grid, block>>>(d_a, d_b, d_c, nElem);

	// copy kernel result back to host side
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free  memory and prepare to use unified memory
	checkCuda(cudaFree(d_c));
	checkCuda(cudaFreeHost(h_a));
	checkCuda(cudaFreeHost(h_b));

	// part 3: using unified memory for all arrays
	float  *umem_a, *umem_b, *umem_c;

	checkCuda(cudaMallocManaged(&umem_a, nBytes));
	checkCuda(cudaMallocManaged(&umem_b, nBytes));
	checkCuda(cudaMallocManaged(&umem_c, nBytes));

	// initialize data at host side
	initialData(umem_a, nElem);
	initialData(umem_b, nElem);

	// sum arrays on host for result check
	sumArraysOnHost(umem_a, umem_b, hostRef, nElem);

	// execute kernel with unified memory
	sumArraysUnified <<<grid, block >>> (umem_a, umem_b, umem_c, nElem);
	checkCuda(cudaDeviceSynchronize());

	checkResult(hostRef, umem_c, nElem);

	// free unified memory
	checkCuda(cudaFree(umem_a));
	checkCuda(cudaFree(umem_b));
	checkCuda(cudaFree(umem_c));

	free(hostRef);
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());
	return EXIT_SUCCESS;
}
