#include <stdio.h>
#include <cuda_runtime.h>

inline
void checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		printf("Error: %s : %d", __FILE__, __LINE__);
		printf("CUDA Runtime Error: %d: %s\n", result, cudaGetErrorString(result));
		exit(1);
	}
#endif
}

__device__ float devData;

__global__ void checkGlobalVar()
{
	// display original value
	printf("Device global var: %f\n", devData);

	// alter the value
	devData += 2.0f;
}

int main(void)
{
	// initialize global variable
	float  value = 3.14f;

	checkCuda(cudaMemcpyToSymbol(devData, &value, sizeof(float)));

	printf_s("Host:   copied %f to the global variable\n", value);

	// invoke the kernel
	checkGlobalVar <<<1, 1 >>>();

	// copy the global variable back to the host
	checkCuda(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
	printf_s("Host:   the value changed by the kernel to %f\n", value);

	checkCuda(cudaDeviceReset());

	return 0;
}