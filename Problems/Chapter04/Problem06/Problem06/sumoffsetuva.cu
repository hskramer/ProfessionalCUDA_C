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
	int i;

	for (i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void sumOnHost(float *a, float *b, float *c, int offset, const int N)
{
	for (int i = offset, k = 0; i < N; i++, k++)
	{
		c[k] = a[i] + b[i];
	}
}

__global__ void memSum(float *a, float *b, float *c, const int N)
{
	int  tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N) c[tid] = a[tid] + b[tid];
}

__global__ void memSumZero(float *a, float *b, float *c, int offset, const int N)
{
	int  tid = blockIdx.x * blockDim.x + threadIdx.x;
	int  k = tid + offset;

	if (k < N) c[tid] = a[k] + b[k];
}

__global__ void memSumZeroUVA(float *a, float *b, float *c, int offset, const int N)
{
	int  tid = blockIdx.x * blockDim.x + threadIdx.x;
	int  k = tid + offset;

	if (k < N) c[tid] = a[k] + b[k];
}

int main(int argc, char **argv)
{
	int  dev = 0;
	cudaDeviceProp  devProp;

	// set device and retrieve properties
	checkCuda(cudaSetDevice(dev));
	checkCuda(cudaGetDeviceProperties(&devProp, dev));

	// check if support mapped memory
	if (!devProp.canMapHostMemory)
	{
		printf_s("Device %d does not support mapping CPU host memory!\n", dev);
		checkCuda(cudaDeviceReset());
		exit(EXIT_SUCCESS);
	}

	printf_s("Program use: enter an offset value or hit enter for default of zero.\n");
	printf_s("Using device %d:  %s\n", dev, devProp.name);

	// set up data size of vectors
	int lShft = 24;
	int offset = 0;

	if (argc > 1)  offset = atoi(argv[1]);

	int  nElem = 1<< lShft;
	size_t  nBytes = nElem * sizeof(float);

	printf_s("Vector size %d memory used  %3.0f MB\n", nElem, (float)nBytes / (1024.0f * 1024.0f));

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

	// add vectors host side for result checks
	sumOnHost(h_a, h_b, hostRef, offset, nElem);

	// allocate device memory
	float  *d_a, *d_b, *d_c;
	checkCuda(cudaMalloc((float**)&d_a, nBytes));
	checkCuda(cudaMalloc((float**)&d_b, nBytes));
	checkCuda(cudaMalloc((float**)&d_c, nBytes));

	//checkCuda(cudaMemcpy((void**)&d_a, (void*)h_a, nBytes, cudaMemcpyHostToDevice));
	//checkCuda(cudaMemcpy((void**)&d_b, (void*)h_b, nBytes, cudaMemcpyHostToDevice));

	// configure kernel launch variables
	int  nthreads = 512;
	dim3  block(nthreads);
	dim3  grid((nElem + block.x - 1) / block.x);

	memSum <<<grid, block >>> (d_a, d_b, d_c, nElem);
	checkCuda(cudaDeviceSynchronize());

	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	printf_s("Sum device memory outcome ");
	checkResult(hostRef, gpuRef, nElem);

	
	// free device global memory 
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));

	// free host memory
	free(h_a);
	free(h_b);

	// part 2: pinned memory without using UVA
	// allocate pinned memory
	unsigned int  flag = cudaHostAllocMapped;
	float  *h_c;
	checkCuda(cudaHostAlloc((void**)&h_a, nBytes, flag));
	checkCuda(cudaHostAlloc((void**)&h_b, nBytes, flag));
	checkCuda(cudaHostAlloc((void**)&h_c, nBytes, flag));// want to try using only host memory

	
	// initialize data at host side
	initialData(h_a, nElem);
	initialData(h_b, nElem);

	checkCuda(cudaHostGetDevicePointer((void**)&d_a, (void*)h_a, 0));
	checkCuda(cudaHostGetDevicePointer((void**)&d_b, (void*)h_b, 0));


	// add at host side for result checks
	sumOnHost(h_a, h_b, hostRef, offset, nElem);

	// execute kernel with zero copy memory
	memSumZero <<<grid, block >>> (d_a, d_b, d_c, offset, nElem);
	checkCuda(cudaDeviceSynchronize());


// copy back results and check
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	printf_s("Zero copy kernel outcome ");
	checkResult(hostRef, gpuRef, nElem - offset);
	
	memset(gpuRef, 0, nBytes);

	// execute kernel with zero copy memory UVA
	memSumZeroUVA <<<grid, block >>> (h_a, h_b, d_c, offset, nElem);
	checkCuda(cudaDeviceSynchronize());

	// copy back results and check
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	printf_s("Zero copy UVA kernel outcome ");
	checkResult(hostRef, gpuRef, nElem - offset);


	// execute kernel with zero copy memory UVA and all in host memory
	// this to me represents the the most effective use of unified virtual addressing (note this is not unified memory)
	// this kernel is executed with all host memory pointers. This eliminates the need to transfer the results back.
	// I checked for implicit memory transfers and there were none.
	memSumZeroUVA << <grid, block >> > (h_a, h_b, h_c, offset, nElem);
	checkCuda(cudaDeviceSynchronize());

	// copy back results and check
	//checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	printf_s("Zero copy UVA kernel outcome all host memory ");
	checkResult(hostRef, h_c, nElem - offset);

	// free device memory
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));
	checkCuda(cudaFree(d_c));

	// free host memory
	checkCuda(cudaFreeHost(h_a));
	checkCuda(cudaFreeHost(h_b));
	free(hostRef);
	free(gpuRef);

	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;
	
}