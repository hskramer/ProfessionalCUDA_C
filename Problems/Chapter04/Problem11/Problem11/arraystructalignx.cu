#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
* Compared Array of structure with read/write of both x and y and this version read/write x only
* both had an efficiency of 50%. The align 8 version maintained a 100% efficiency (problem 11b).
*/

#define LEN 1<<22

struct innerStruct {
	float x;
	float y;
};

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


void initialInnerStruct(innerStruct *ip, int size)
{
	for (int i = 0; i < size; i++)
	{
		ip[i].x = (float)(rand() & 0xFF) / 100.0f;		
	}

	return;
}

void testInnerStructHost(innerStruct *a, innerStruct *c, const int N)
{
	for (int idx = 0; idx < N; idx++)
	{
		c[idx].x = a[idx].x + 10.f;		
	}

	return;
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i].x - gpuRef[i].x) > epsilon)
		{
			match = 0;
			printf_s("different on %dth element: host %f gpu %f\n", i, hostRef[i].x, gpuRef[i].x);
			break;
		}

	}

	if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void testInnerStruct(innerStruct *data, innerStruct * result, const int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		innerStruct tmp = data[i];
		tmp.x += 10.f;
		result[i] = tmp;
	}
}

__global__ void warmup(innerStruct *data, innerStruct * result, const int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		innerStruct tmp = data[i];
		tmp.x += 10.f;
		result[i] = tmp;
	}
}

int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("%s test struct of array at ", argv[0]);
	printf_s("device %d: %s \n", dev, deviceProp.name);

	checkCuda(cudaSetDevice(dev));

	// allocate host memory
	int nElem = LEN;
	size_t nBytes = nElem * sizeof(innerStruct);

	innerStruct     *h_a = (innerStruct *)malloc(nBytes);
	innerStruct *hostRef = (innerStruct *)malloc(nBytes);
	innerStruct *gpuRef  = (innerStruct *)malloc(nBytes);

	// initialize host array
	initialInnerStruct(h_a, nElem);
	testInnerStructHost(h_a, hostRef, nElem);

	// allocate device memory
	innerStruct *d_a, *d_c;
	checkCuda(cudaMalloc((innerStruct**)&d_a, nBytes));
	checkCuda(cudaMalloc((innerStruct**)&d_c, nBytes));

	// copy data from host to device
	checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));

	// set up offset for summaryAU: It is blocksize not offset.
	int blocksize = 256;

	if (argc > 1) blocksize = atoi(argv[1]);

	// execution configuration
	dim3 block(blocksize, 1);
	dim3 grid((nElem + block.x - 1) / block.x, 1);

	// kernel 1: warmup

	warmup <<<grid, block >>>(d_a, d_c, nElem);
	checkCuda(cudaDeviceSynchronize());

	printf_s("warmup   <<< %3d, %3d >>>\n", grid.x, block.x);
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	checkInnerStruct(hostRef, gpuRef, nElem);

	// kernel 2: testInnerStruct x only

	testInnerStruct <<<grid, block >>>(d_a, d_c, nElem);
	checkCuda(cudaDeviceSynchronize());

	printf_s("innerstruct   <<< %3d, %3d >>>\n", grid.x, block.x);
	checkCuda(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));
	checkInnerStruct(hostRef, gpuRef, nElem);

	// free memories both host and device
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_c));
	free(h_a);
	free(hostRef);
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;
}
