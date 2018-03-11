#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda.h>


#define NSTREAM 8
#define BDIM 128

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

void initialData(float *ip, int size)
{
	int i;

	for (i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
	for (int idx = 0; idx < N; idx++)
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		for (int i = 0; i < N; ++i)
		{
			C[idx] = A[idx] + B[idx];
		}
	}
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf_s("Arrays do not match!\n");
			printf_s("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}

	if (match) printf_s("Arrays match.\n\n");
}

int main(int argc, char **argv)
{
	printf_s("> %s Starting...\n", argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("> Using Device %d: %s\n", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// check if device support hyper-q
	if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
	{
		if (deviceProp.concurrentKernels == 0)
		{
			printf_s("> GPU does not support concurrent kernel execution (SM 3.5 or higher required)\n");
			printf_s("> CUDA kernel runs will be serialized\n");
		}
		else
		{
			printf_s("> GPU does not support HyperQ\n");
			printf_s("> CUDA kernel runs will have limited concurrency\n");
		}
	}

	printf_s("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

	printf_s("> with streams = %d\n", NSTREAM);

	// set up data size of vectors
	int nElem = 1 << 19;
	printf_s("> vector size = %d\n", nElem);
	size_t nBytes = nElem * sizeof(float);

	// malloc pinned host memory for async memcpy
	float *h_A, *h_B, *hostRef, *gpuRef;
	checkCuda(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault));
	checkCuda(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault));
	checkCuda(cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault));
	checkCuda(cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault));

	// initialize data at host side
	initialData(h_A, nElem);
	initialData(h_B, nElem);

	// add vector at host side for result checks
	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	// malloc device global memory
	float *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc((float**)&d_A, nBytes));
	checkCuda(cudaMalloc((float**)&d_B, nBytes));
	checkCuda(cudaMalloc((float**)&d_C, nBytes));

	cudaEvent_t start, stop;
	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));

	// invoke kernel at host side
	dim3 block(BDIM);
	dim3 grid((nElem + block.x - 1) / block.x);
	printf_s("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x, block.y);

	// sequential operation
	checkCuda(cudaEventRecord(start, 0));
	checkCuda(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop));
	float memcpy_h2d_time;
	checkCuda(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

	checkCuda(cudaEventRecord(start, 0));
	sumArrays << <grid, block >> >(d_A, d_B, d_C, nElem);
	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop));
	float kernel_time;
	checkCuda(cudaEventElapsedTime(&kernel_time, start, stop));

	checkCuda(cudaEventRecord(start, 0));
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop));
	float memcpy_d2h_time;
	checkCuda(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
	float itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

	printf_s("\n");
	printf_s(" Measured timings (throughput):\n");
	printf_s(" Memcpy host to device\t: %f ms (%f GB/s)\n", memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
	printf_s(" Memcpy device to host\t: %f ms (%f GB/s)\n", memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
	printf_s(" Kernel\t\t\t: %f ms (%f GB/s)\n", kernel_time, (nBytes * 2e-6) / kernel_time);
	printf_s(" Total\t\t\t: %f ms (%f GB/s)\n", itotal, (nBytes * 2e-6) / itotal);

	// grid parallel operation
	int iElem = nElem / NSTREAM;
	size_t iBytes = iElem * sizeof(float);
	grid.x = (iElem + block.x - 1) / block.x;

	cudaStream_t stream[NSTREAM];

	for (int i = 0; i < NSTREAM; ++i)
	{
		checkCuda(cudaStreamCreate(&stream[i]));
	}

	checkCuda(cudaEventRecord(start, 0));

	// initiate all asynchronous transfers to the device
	for (int i = 0; i < NSTREAM; ++i)
	{
		int ioffset = i * iElem;
		checkCuda(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
		checkCuda(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
	}

	// launch a kernel in each stream
	for (int i = 0; i < NSTREAM; ++i)
	{
		int ioffset = i * iElem;
		sumArrays <<<grid, block, 0, stream[i] >>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iElem);
	}

	// enqueue asynchronous transfers from the device
	for (int i = 0; i < NSTREAM; ++i)
	{
		int ioffset = i * iElem;
		checkCuda(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]));
	}


	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop));
	float execution_time;
	checkCuda(cudaEventElapsedTime(&execution_time, start, stop));

	printf_s("\n");
	printf_s("Actual results from overlapped data transfers:\n");
	printf_s(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM, execution_time, (nBytes * 2e-6) / execution_time);
	printf_s(" speedup                : %f \n", ((itotal - execution_time) * 100.0f) / itotal);

	// check kernel error
	checkCuda(cudaGetLastError());

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free device global memory
	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_B));
	checkCuda(cudaFree(d_C));

	// free host memory
	checkCuda(cudaFreeHost(h_A));
	checkCuda(cudaFreeHost(h_B));
	checkCuda(cudaFreeHost(hostRef));
	checkCuda(cudaFreeHost(gpuRef));

	// destroy events
	checkCuda(cudaEventDestroy(start));
	checkCuda(cudaEventDestroy(stop));

	// destroy streams
	for (int i = 0; i < NSTREAM; ++i)
	{
		checkCuda(cudaStreamDestroy(stream[i]));
	}

	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;
}