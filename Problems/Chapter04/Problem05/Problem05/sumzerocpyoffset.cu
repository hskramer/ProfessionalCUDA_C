#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* zero copy memory vs pinned memory:
*  the primary difference is that zero copy mem is mapped into the devices memory space i.e.(universal virtual address space)
*  Pinned memory allows for greater transfer speeds between host and device because it's guaranteed to be page-locked. I could
*  have used the host pointers in the kernel call but that was not the point of the exercise thats next.  Had to modify sumHost
*  and checkResult to deal with the offset. Solution failed when I entered offsets, also removed check for for small power
*  since you can't change it from command line only from source and I decided not to make it a command line option. My gpu
*  has the pascal architecture which is always 32byte aligned performance for 0,16,32,.. offset were the same odd offsets
*  doubled kernel times even offsets not a multiple of 32 and less than 8 such as 6 performed well.
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
	int i;

	for (i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void sumHost(float *a, float *b, float *c, int offset, const int N)
{
	for (int i = offset, k = 0; i < N; i++, k++)
	{
		c[k] = a[i] + b[i];
	}
}

__global__ void warmup(float *a, float *b, float *c, int offset, const int N)
{
	int  tid = blockIdx.x * blockDim.x + threadIdx.x;
	int  k = tid + offset;

	if (k < N) c[tid] = a[k] + b[k];
}

__global__ void sumArraysOffset(float *a, float *b, float *c, int offset, const int N)

{
	int  tid = blockIdx.x * blockDim.x + threadIdx.x;
	int  k = tid + offset;

	if (k < N) c[tid] = a[k] + b[k];
}

int main(int argc, char **argv)
{
	int  dev = 0;
	cudaDeviceProp  devProp;

	// set up device
	checkCuda(cudaSetDevice(dev));

	// get device properties
	checkCuda(cudaGetDeviceProperties(&devProp, dev));

	// check if support mapped memory
	if (!devProp.canMapHostMemory)
	{
		printf_s("Device %d does not support mapping CPU host memory!\n", dev);
		checkCuda(cudaDeviceReset());
		exit(EXIT_SUCCESS);
	}


	// get device properties
	checkCuda(cudaGetDeviceProperties(&devProp, dev));
	printf_s("Program use: enter an offset value or hit enter for default zero.\n");
	printf_s("Using Device %d: %s ", dev, devProp.name);

	// set up data size of vectors
	int lShft = 22;
	int offset = 0;

	if (argc > 1)  offset = atoi(argv[1]);

	int  nElem = (1 << lShft) + offset;
	size_t  nBytes = nElem * sizeof(float);

	printf_s("Vector size %d memory used  %3.0f MB\n", nElem, (float)nBytes / (1024.0f * 1024.0f));


	// part 1: using device memory and standard memory

	// malloc host memory
	float  *h_a, *h_b, *hostRef, *gpuRef;
	h_a		=	(float *)malloc(nBytes);
	h_b		=	(float *)malloc(nBytes);
	hostRef =	(float *)malloc(nBytes);
	gpuRef	=	(float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_a, nElem);
	initialData(h_b, nElem);

	// sum on host to validate gpu sum
	sumHost(h_a, h_b, hostRef, offset, nElem);

	// cudaMalloc device memory
	float *d_a, *d_b, *d_c;
	checkCuda(cudaMalloc((float**)&d_a, nBytes));
	checkCuda(cudaMalloc((float**)&d_b, nBytes));
	checkCuda(cudaMalloc((float**)&d_c, nBytes));

	// copy data to device
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice));

	// setup kernel execution parameters
	int  nthreads = 512;
	dim3  block(nthreads);
	dim3  grid((nElem + block.x - 1) / block.x);

	warmup << <grid, block >> > (d_a, d_b, d_c, offset, nElem);

	// free memory
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));

	free(h_a);
	free(h_b);

	// part 2: using zero copy memory for array's A and B

	checkCuda(cudaHostAlloc((void**)&h_a, nBytes, cudaHostAllocMapped));
	checkCuda(cudaHostAlloc((void**)&h_b, nBytes, cudaHostAllocMapped));

	// initialize data at host side
	initialData(h_a, nElem);
	initialData(h_b, nElem);


	// get device pointers 
	checkCuda(cudaHostGetDevicePointer((void**)&d_a, (void*)h_a, 0));
	checkCuda(cudaHostGetDevicePointer((void**)&d_b, (void*)h_b, 0));

	// sum on host to validate gpu sum
	sumHost(h_a, h_b, hostRef, offset, nElem);

	// execute kernel with zero copy memory
	sumArraysOffset << <grid, block >> >(d_a, d_b, d_c, offset, nElem);

	// copy kernel result back to host side
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElem - offset);

	// free  memory
	checkCuda(cudaFree(d_c));
	checkCuda(cudaFreeHost(h_a));
	checkCuda(cudaFreeHost(h_b));

	free(hostRef);
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;
}


