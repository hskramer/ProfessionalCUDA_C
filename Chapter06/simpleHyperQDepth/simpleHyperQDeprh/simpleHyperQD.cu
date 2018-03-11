#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


#define N 300000
#define NSTREAM 4

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


__global__ void kernel_1()
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_2()
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_3()
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_4()
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

int main(int argc, char **argv)
{
	int n_streams = NSTREAM;
	int isize = 1;
	int iblock = 1;
	int bigcase = 0;

	// get argument from command line
	if (argc > 1) n_streams = atoi(argv[1]);

	if (argc > 2) bigcase = atoi(argv[2]);

	float elapsed_time;

	int dev = 0;
	cudaDeviceProp deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name, n_streams);
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

	// Allocate and initialize an array of stream handles
	cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));

	for (int i = 0; i < n_streams; i++)
	{
		checkCuda(cudaStreamCreate(&(streams[i])));
	}

	// run kernel with more threads
	if (bigcase == 1)
	{
		iblock = 512;
		isize = 1 << 18;
	}

	// set up execution configuration
	dim3 block(iblock);
	dim3 grid(isize / iblock);
	printf_s("> grid %d block %d\n", grid.x, block.x);

	// creat events
	cudaEvent_t start, stop;
	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));

	// record start event
	checkCuda(cudaEventRecord(start, 0));

	// dispatch job with depth first ordering
	for (int i = 0; i < n_streams; i++)
	{
		kernel_1 <<<grid, block, 0, streams[i] >>>();
		kernel_2 <<<grid, block, 0, streams[i] >>>();
		kernel_3 <<<grid, block, 0, streams[i] >>>();
		kernel_4 <<<grid, block, 0, streams[i] >>>();
	}

	// record stop event
	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop));

	// calculate elapsed time
	checkCuda(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf_s("Measured time for parallel execution = %.3fs\n", elapsed_time / 1000.0f);

	// release all stream
	for (int i = 0; i < n_streams; i++)
	{
		checkCuda(cudaStreamDestroy(streams[i]));
	}

	free(streams);

	// destroy events
	checkCuda(cudaEventDestroy(start));
	checkCuda(cudaEventDestroy(stop));

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;
}