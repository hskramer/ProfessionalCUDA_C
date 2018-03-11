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


__global__ void fmad_kernel(double x, double y, double *out)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid == 0)
	{
		*out = x * x + y;
	}
}

double host_fmad_kernel(double x, double y)
{
	return x * x + y;
}

int main(int argc, char **argv)
{
	double *d_out, h_out;
	double x = 2.891903;
	double y = -3.980364;

	double host_value = host_fmad_kernel(x, y);
	checkCuda(cudaMalloc((void **)&d_out, sizeof(double)));
	fmad_kernel << <1, 32 >> >(x, y, d_out);
	checkCuda(cudaMemcpy(&h_out, d_out, sizeof(double),
		cudaMemcpyDeviceToHost));

	if (host_value == h_out)
	{
		printf_s("The device output the same value as the host.\n");
	}
	else
	{
		printf_s("The device output a different value than the host, diff=%e.\n", abs(host_value - h_out));
	}

	return 0;
}
