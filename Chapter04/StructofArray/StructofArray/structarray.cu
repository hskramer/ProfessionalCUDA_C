#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>
#include <cuda_runtime.h>


/*
* A simple example of using a structore of arrays to store data on the device.
* This example is used to study the impact on performance of data layout on the
* GPU.
* global load/store efficiency now 100% despite this kernel runtimes are the same
* as the array of structures even when I pushed LEN to 24
* SoA: contiguous reads for x and y
*/

#define LEN 1<<22

struct InnerArray {
	float x[LEN];
	float y[LEN];
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


// functions for inner array outer struct
void initialInnerArray(InnerArray *ip, int size)
{
	for (int i = 0; i < size; i++)
	{
		ip->x[i] = (float)(rand() & 0xFF) / 100.0f;
		ip->y[i] = (float)(rand() & 0xFF) / 100.0f;
	}

	return;
}

void testInnerArrayHost(InnerArray *A, InnerArray *C, const int n)
{
	for (int idx = 0; idx < n; idx++)
	{
		C->x[idx] = A->x[idx] + 10.f;
		C->y[idx] = A->y[idx] + 20.f;
	}

	return;
}


void printfHostResult(InnerArray *C, const int n)
{
	for (int idx = 0; idx < n; idx++)
	{
		printf_s("printout idx %d:  x %f y %f\n", idx, C->x[idx], C->y[idx]);
	}

	return;
}

void checkInnerArray(InnerArray *hostRef, InnerArray *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon)
		{
			match = 0;
			printf_s("different on x %dth element: host %f gpu %f\n", i, hostRef->x[i], gpuRef->x[i]);
			break;
		}

		if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon)
		{
			match = 0;
			printf_s("different on y %dth element: host %f gpu %f\n", i, hostRef->y[i], gpuRef->y[i]);
			break;
		}
	}

	if (!match)  printf_s("Arrays do not match.\n\n");
}

__global__ void testInnerArray(InnerArray *data, InnerArray * result, const int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		float tmpx = data->x[i];
		float tmpy = data->y[i];

		tmpx += 10.f;
		tmpy += 20.f;
		result->x[i] = tmpx;
		result->y[i] = tmpy;
	}
}

__global__ void warmup(InnerArray *data, InnerArray * result, const int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		float tmpx = data->x[i];
		float tmpy = data->y[i];
		tmpx += 10.f;
		tmpy += 20.f;
		result->x[i] = tmpx;
		result->y[i] = tmpy;
	}
}

// test for array of struct
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
	size_t nBytes = sizeof(InnerArray);

	InnerArray     *h_A = (InnerArray *)malloc(nBytes);
	InnerArray *hostRef = (InnerArray *)malloc(nBytes);
	InnerArray *gpuRef  = (InnerArray *)malloc(nBytes);

	// initialize host array
	initialInnerArray(h_A, nElem);
	testInnerArrayHost(h_A, hostRef, nElem);

	// allocate device memory
	InnerArray *d_A, *d_C;
	checkCuda(cudaMalloc((InnerArray**)&d_A, nBytes));
	checkCuda(cudaMalloc((InnerArray**)&d_C, nBytes));

	// copy data from host to device
	checkCuda(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

	// set up offset for summary
	int blocksize = 256;

	if (argc > 1) blocksize = atoi(argv[1]);

	// execution configuration
	dim3 block(blocksize, 1);
	dim3 grid((nElem + block.x - 1) / block.x, 1);

	// kernel 1:
	clock_t	iStart, iStop;
	iStart = clock();
	warmup <<<grid, block >>>(d_A, d_C, nElem);
	checkCuda(cudaDeviceSynchronize());

	iStop = clock();
	double iElaps = double(iStop - iStart) / CLOCKS_PER_SEC;
	printf_s("warmup      <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x, iElaps);
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkInnerArray(hostRef, gpuRef, nElem);
	checkCuda(cudaGetLastError());

	iStart = clock();
	testInnerArray << <grid, block >> >(d_A, d_C, nElem);
	checkCuda(cudaDeviceSynchronize());

	iStop = clock();
	iElaps = double(iStop - iStart) / CLOCKS_PER_SEC;
	printf_s("innerarray   <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x, iElaps);
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkInnerArray(hostRef, gpuRef, nElem);
	checkCuda(cudaGetLastError());

	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_C));
	free(h_A);
	free(hostRef);
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());
	return EXIT_SUCCESS;
}