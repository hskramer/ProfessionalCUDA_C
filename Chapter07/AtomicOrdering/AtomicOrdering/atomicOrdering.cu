#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
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


/**
* Utility function for printing the contents of an array.
**/
static void print_read_results(int *h_arr, int *d_arr, int N, const char *label)
{
	int i;
	int maxNumToPrint = 10;
	int nToPrint = N > maxNumToPrint ? maxNumToPrint : N;
	checkCuda(cudaMemcpy(h_arr, d_arr, nToPrint * sizeof(int), cudaMemcpyDeviceToHost));
	printf_s("Threads performing %s operations read values", label);

	for (i = 0; i < nToPrint; i++)
	{
		printf_s(" %d", h_arr[i]);
	}

	printf_s("\n");
}

/**
* This version of the kernel uses atomic operations to safely increment a
* shared variable from multiple threads.
**/
__global__ void atomics(int *shared_var, int *values_read, int N, int iters)
{
	int i;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= N) return;

	values_read[tid] = atomicAdd(shared_var, 1);

	for (i = 0; i < iters; i++)
	{
		atomicAdd(shared_var, 1);
	}
}

/**
* This version of the kernel performs the same increments as atomics() but in
* an unsafe manner.
**/
__global__ void unsafe(int *shared_var, int *values_read, int N, int iters)
{
	int i;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= N) return;

	int old = *shared_var;
	*shared_var = old + 1;
	values_read[tid] = old;

	for (i = 0; i < iters; i++)
	{
		int old = *shared_var;
		*shared_var = old + 1;
	}
}

int main(int argc, char **argv)
{
	int N = 64;
	int block = 32;
	int runs = 30;
	int iters = 100000;
	int r;
	int *d_shared_var;
	int h_shared_var_atomic, h_shared_var_unsafe;
	int *d_values_read_atomic;
	int *d_values_read_unsafe;
	int *h_values_read;

	checkCuda(cudaMalloc((void **)&d_shared_var, sizeof(int)));
	checkCuda(cudaMalloc((void **)&d_values_read_atomic, N * sizeof(int)));
	checkCuda(cudaMalloc((void **)&d_values_read_unsafe, N * sizeof(int)));

	h_values_read = (int *)malloc(N * sizeof(int));


	for (r = 0; r < runs; r++)
	{

		checkCuda(cudaMemset(d_shared_var, 0x00, sizeof(int)));
		atomics <<<N / block, block >>>(d_shared_var, d_values_read_atomic, N, iters);
		checkCuda(cudaDeviceSynchronize());
		
		checkCuda(cudaMemcpy(&h_shared_var_atomic, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost));

		checkCuda(cudaMemset(d_shared_var, 0x00, sizeof(int)));
		unsafe <<<N / block, block >>>(d_shared_var, d_values_read_unsafe, N, iters);
		checkCuda(cudaDeviceSynchronize());

		checkCuda(cudaMemcpy(&h_shared_var_unsafe, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost));
	}

	printf_s("In total, %d runs using atomic operations\n",runs);
	printf_s("  Using atomic operations also produced an output of %d\n", h_shared_var_atomic);
	printf_s("In total, %d runs using unsafe operations\n", runs);
	printf_s("  Using unsafe operations also produced an output of %d\n", h_shared_var_unsafe);

	print_read_results(h_values_read, d_values_read_atomic, N, "atomic");
	print_read_results(h_values_read, d_values_read_unsafe, N, "unsafe");

	return 0;
}
