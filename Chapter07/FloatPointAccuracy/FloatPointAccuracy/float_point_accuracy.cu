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
* Save the single- and double-precision representation of 12.1 from the device
* into global memory. That global memory is then copied back to the host for
* later analysis.
**/


__global__ void kernel(float *F, double *D)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid == 0)
	{
		*F = 12.1;
		*D = 12.1;
	}
}

int main(int argc, char **argv)
{
	float *deviceF;
	float h_deviceF;
	double *deviceD;
	double h_deviceD;

	float hostF = 12.1;
	double hostD = 12.1;

	checkCuda(cudaMalloc((void **)&deviceF, sizeof(float)));
	checkCuda(cudaMalloc((void **)&deviceD, sizeof(double)));
	kernel << <1, 32 >> >(deviceF, deviceD);
	checkCuda(cudaMemcpy(&h_deviceF, deviceF, sizeof(float), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(&h_deviceD, deviceD, sizeof(double), cudaMemcpyDeviceToHost));

	printf_s("Host single-precision representation of 12.1   = %.20f\n", hostF);
	printf_s("Host double-precision representation of 12.1   = %.20f\n", hostD);
	printf_s("Device single-precision representation of 12.1 = %.20f\n", hostF);
	printf_s("Device double-precision representation of 12.1 = %.20f\n", hostD);
	printf_s("Device and host single-precision representation equal? %s\n", hostF == h_deviceF ? "yes" : "no");
	printf_s("Device and host double-precision representation equal? %s\n", hostD == h_deviceD ? "yes" : "no");

	return 0;
}

