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

__device__ int myAtomicMin(int *address, int val)
{
	int  guess = *address;
	int  oldVal = atomicCAS(address, guess, guess < val ? guess : val);

	//loop while guess is greater than val
	while (oldVal != guess)
	{
		guess = oldVal;
		oldVal = atomicCAS(address, guess, guess < val ? guess : val);

	}

	return oldVal;
}


__global__ void kernel(int *sharedInteger, int kval)
{
	myAtomicMin(sharedInteger, kval);
}

__global__ void min_kernel(int *address, int val)
{
	 atomicMin(address, val);	
}

int main(int argc, char **argv)
{
	int  h_val;
	int  *d_val;

	checkCuda(cudaMalloc(&d_val, sizeof(int)));
	checkCuda(cudaMemset(d_val, 0x11, sizeof(int)));

	kernel <<<1, 1 >>> (d_val, 18);

	//min_kernel << <1, 1 >> > (d_val, 18);

	checkCuda(cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost));

	printf_s("The min value is %d\n", h_val);

	return 0;
}