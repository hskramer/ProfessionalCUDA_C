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

__device__ double myAtomicDblAdd(double *address, double incr)
{
	// Convert address to point to supported type and size
	unsigned long long int *typedAddress = (unsigned long long int *)address;

	// Store expected and desired double as unsigned long long int
	double  currentVal = *address;


	// Need to be carefull here you need to use the function __double_as_longlong and __longlong_as_double
	// which reinterpret the values as opposed to converting from on form to another, a subtle but important difference.
	unsigned long long int expected = __double_as_longlong(currentVal);
	unsigned long long int desired  = __double_as_longlong(currentVal + incr);
	
	unsigned long long int oldIntValue = atomicCAS(typedAddress, expected, desired);

	while (oldIntValue != expected)
	{
		expected = oldIntValue;

		desired = __double_as_longlong(__longlong_as_double(oldIntValue) + incr);

		oldIntValue = atomicCAS(typedAddress, expected, desired);

	}

	return __longlong_as_double(oldIntValue);
}

__global__ void myKernel(double *sharedDouble)
{
	myAtomicDblAdd(sharedDouble, 1.0);
}

int main(int argc, char **argv)
{
	double  h_double;
	double *d_double;

	checkCuda(cudaMalloc(&d_double, sizeof(double)));
	checkCuda(cudaMemset(d_double, 0x00, sizeof(double)));

	myKernel <<<4, 128 >>> (d_double);

	checkCuda(cudaMemcpy(&h_double, d_double, sizeof(double), cudaMemcpyDeviceToHost));

	printf_s("4 x 128 increments led to a value of %e", h_double);

	return 0;

}