
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


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



__device__ int myAtomicAdd(int *addr, int incr)
{
	// Create an initial guess for the value stored at *address.
	int guess = *addr;
	int oldValue = atomicCAS(addr, guess, guess + incr);

	// Loop while the guess is incorrect.
	while (oldValue != guess)
	{
		guess = oldValue;
		oldValue = atomicCAS(addr, guess, guess + incr);
	}

	return oldValue;
}

__global__ void kernel(int *sharedInteger)
{
	myAtomicAdd(sharedInteger, 1);
}

int main(int argc, char **argv)
{
	int  h_sharedInt;
	int  *d_sharedInt;

	checkCuda(cudaMalloc(&d_sharedInt, sizeof(int)));
	checkCuda(cudaMemset(d_sharedInt, 0x00, sizeof(int)));

	kernel <<<4, 128 >>> (d_sharedInt);

	checkCuda(cudaMemcpy(&h_sharedInt, d_sharedInt, sizeof(int), cudaMemcpyDeviceToHost));
	printf_s("4 x 128 increments led to value of %d\n", h_sharedInt);

	return 0;

}