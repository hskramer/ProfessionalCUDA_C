#include <stdio.h>
#include <stdlib.h>
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

__device__ float devData[1];

__global__ void checkGlobalVar()
{
	// display the original value
	printf("Device:  value of global variable is %f\n", devData[0]);

	// alter global value
	devData[0] += 2.0f;
}

int main(void)
{
	// initialize variables
	float  value = 3.14f;
	float  *devPtr = NULL;
	
	// get address and copy to device
	checkCuda(cudaGetSymbolAddress((void**)&devPtr, devData));
	checkCuda(cudaMemcpy(devPtr, &value, sizeof(float), cudaMemcpyHostToDevice));
	printf_s("Host:  copied %f to the global variable\n", value);

	// invoke kernel
	checkGlobalVar <<<1, 1>>> ();
	
	// copy variable back to host
	checkCuda(cudaMemcpy(&value, devPtr, sizeof(float), cudaMemcpyDeviceToHost));
	printf_s("Host:  the value changed by the kernel to %f\n", value);

	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}