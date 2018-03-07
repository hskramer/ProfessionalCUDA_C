#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

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
	float epsilon = 1.0E-8;

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
	
	printf_s("arrays match\n");

	return;
}

void initializeData(float* ip, int const size)
{
	time_t t;
	srand(time(&t));
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void sumHost(float* a, float* b, float* c, int const N)
{
	for (int i = 0; i < N; i++)
	{
		c[i] = a[i] + b[i];
	}
}

__global__ void sumGPU(float* a, float* b, float* c, int const N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)  c[idx] = a[idx] + b[idx];
}

__global__ void sumGPUZeroMem(float* a, float* b, float* c, int const N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)	c[idx] = a[idx] + b[idx];
}

int main(int argc, char** argv)
{
	int  dev = 0;
	cudaDeviceProp  devProp;

	// set device and get properties
	checkCuda(cudaSetDevice(dev));
	checkCuda(cudaGetDeviceProperties(&devProp, dev));

	// check for MapHostMemory support
	if(!devProp.canMapHostMemory)
	{
		printf("Device %d does not support mapping CPU host memory!\n", dev);
		checkCuda(cudaDeviceReset());
		exit(EXIT_SUCCESS);
	}

	printf_s("This program sums two arrays of random integers and stores it in a third using zero copy memory\n");
	printf_s("Enter a number between 11 and 27 or hit enter for default (1024 elements).\n");

	int  lShift =  10;

	if (argc > 1)
	{
		lShift = atoi(argv[1]);
	}

	printf_s("\nUsing device %d: %s ", dev, devProp.name);

	int  numElems = 1 << lShift;

	size_t  nbytes = numElems * sizeof(float);

	if (lShift < 18)
	{
		printf_s("Vector size %d requires %3.0f KB of memory\n", numElems, (float)nbytes / (1024.0f));
	}
	else
	{
		printf_s("Vector size %d requires %3.0f MB of memory\n", numElems, (float)nbytes / (1024.0f * 1024.0f));
	}

	
	// part 1 using host memory and device memory with a typical copy
	// malloc host memory
	float  *h_a,  *h_b, *hostRef, *gpuRef;
	h_a		= (float*) malloc(nbytes);
	h_b		= (float*)malloc(nbytes);
	hostRef	= (float*)malloc(nbytes);
	gpuRef = (float*)malloc(nbytes);

	// intialize host vectors with data
	initializeData(h_a, numElems);
	initializeData(h_b, numElems);

	sumHost(h_a, h_b, hostRef, numElems);

	// cudaMalloc device memory
	float  *d_a, *d_b, *d_c;
	checkCuda(cudaMalloc((float**)&d_a, nbytes));
	checkCuda(cudaMalloc((float**)&d_b, nbytes));
	checkCuda(cudaMalloc((float**)&d_c, nbytes));

	checkCuda(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice));
	
	// define variables for kernel
	int  nthreads = 512;
	dim3  block(nthreads);
	dim3  grid((numElems + block.x - 1) / block.x);

	//execute standard kernel
	sumGPU <<<grid, block >>> (d_a, d_b, d_c, numElems);
	checkCuda(cudaDeviceSynchronize());
	printf_s("Sum using device memory \n");

	// copy kernel results back to host
	checkCuda(cudaMemcpy(gpuRef, d_c, nbytes, cudaMemcpyDeviceToHost));

	// compare results 
	checkResult(hostRef, gpuRef, numElems);

	// free device memory
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));

	// free host memory
	free(h_a);
	free(h_b);

	// part 2 using zero copy memory
	unsigned int flags = cudaHostAllocMapped;
	checkCuda(cudaHostAlloc((void**)&h_a, nbytes, flags));
	checkCuda(cudaHostAlloc((void**)&h_b, nbytes, flags));

	// intialize host vectors with data
	initializeData(h_a, numElems);
	initializeData(h_b, numElems);

	//checkCuda(cudaSetDeviceFlags(cudaDeviceMapHost));
	checkCuda(cudaHostGetDevicePointer((void**)&d_a, (void*)h_a, 0));
	checkCuda(cudaHostGetDevicePointer((void**)&d_b, (void*)h_b, 0));

	sumHost(h_a, h_b, hostRef, numElems);

	// execute zero copy kernel
	sumGPUZeroMem <<<grid, block >>> (d_a, d_b, d_c, numElems);
	checkCuda(cudaDeviceSynchronize());
	printf_s("Sum using zero copy memory \n");
	
	// copy kernel results back to host
	checkCuda(cudaMemcpy(gpuRef, d_c, nbytes, cudaMemcpyDeviceToHost));

	// compare results 
	checkResult(hostRef, gpuRef, numElems);

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