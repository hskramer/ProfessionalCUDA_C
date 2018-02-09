#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
* An example of using a statically declared global variable (devData) to store
* a floating-point value on the device.
*/

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


__device__ float devData[5];

__global__ void checkGlobalVariable()
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// display value before kernel change 
	printf("The device value: %f and thread id: %d\n", devData[tid], tid);
	if (tid < 5)
	{
		devData[tid] *= tid;
	}

}

int main(void)
{
	// initialize the global variable
	float values[5] = { 3.14f, 3.14f, 3.14f, 3.14f, 3.14f };

	checkCuda(cudaMemcpyToSymbol(devData, values, 5 * sizeof(float)));
	printf("Host:   copied %f to the global array\n\n", values[0]);
	
	
	// invoke the kernel
	checkGlobalVariable <<<1, 5>>>();

	// copy the global variable back to the host
	checkCuda(cudaMemcpyFromSymbol(values, devData, 5 * sizeof(float)));
	for (int i = 0; i < 5; i++)
	{
		printf("Host: the value changed by the kernel to %f\n", values[i]);
	}

	checkCuda(cudaDeviceReset());
	return EXIT_SUCCESS;
}
