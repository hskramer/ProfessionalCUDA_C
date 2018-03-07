#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define BDIMX 16
#define SEGM  4

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

void printData(double *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		printf_s("%2.2f ", in[i]);
	}

	printf_s("\n");
}

__global__ void double_shfl_wrap(double *d_out, double *d_in, int const offset)
{
	double value = d_in[threadIdx.x];
	value = __shfl(value, threadIdx.x + offset, BDIMX);
	d_out[threadIdx.x] = value;
}

int main(int argc, char **argv)
{
	int  dev = 0;

	cudaDeviceProp  deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));

	printf_s("> %s starting ", argv[0]);
	printf_s("> on device %d: %s\n", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// set number of elements and the size of the array's
	int  nElem = BDIMX;
	double  h_iData[BDIMX], h_oData[BDIMX];

	for (int i = 0; i < nElem; i++) h_iData[i] = i * 1.0;

	printf("intial data\t\t: ");
	printData(h_iData, nElem);


	// configure kernel
	int  block = BDIMX;

	// allocate device memory
	size_t  nBytes = nElem * sizeof(double);
	double  *d_iData, *d_oData;

	checkCuda(cudaMalloc(&d_iData, nBytes));
	checkCuda(cudaMalloc(&d_oData, nBytes));

	// run kernels
	checkCuda(cudaMemcpy(d_iData, h_iData, nBytes, cudaMemcpyHostToDevice));

	double_shfl_wrap <<<1, block >>> (d_oData, d_iData, 4);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(h_oData, d_oData, nBytes, cudaMemcpyDeviceToHost));

	printf_s("double shfl wrap\t: ");
	printData(h_oData, nElem);

	// free memory
	checkCuda(cudaFree(d_iData));
	checkCuda(cudaFree(d_oData));

	return 0;

}