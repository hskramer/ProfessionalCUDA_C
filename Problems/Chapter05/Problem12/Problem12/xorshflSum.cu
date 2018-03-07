#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define BDIMX	16

/* If your comfortable using  mask's this one is straight forward a mask of one switches every other number so 
*  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  becomes
*  1  0  3  2  5  4  7  6  9  8  11  10  13  12  15  14 addition is commutative so the sum of adjacent numbers will be the same.
*  1  1  5  5  9  9  13 13 17 17 21  21  25  25  29  29
*/

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

void printData(int *data, int isize)
{
	for (int i = 0; i < isize; i++)
	{
		printf_s("%2d ", data[i]);
	}
	printf_s("\n");
}


__global__ void shfl_xor_sum(int *d_out, int *d_in)
{
	int value = d_in[threadIdx.x];
	value += __shfl_xor(value, 1, BDIMX);
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
	int  h_iData[BDIMX], h_oData[BDIMX];

	for (int i = 0; i < nElem; i++) h_iData[i] = i;

	printf("intial data\t\t: ");
	printData(h_iData, nElem);


	// configure kernel
	int  block = BDIMX;

	// allocate device memory
	size_t  nBytes = nElem * sizeof(int);
	int  *d_iData, *d_oData;

	checkCuda(cudaMalloc(&d_iData, nBytes));
	checkCuda(cudaMalloc(&d_oData, nBytes));

	// run kernels
	checkCuda(cudaMemcpy(d_iData, h_iData, nBytes, cudaMemcpyHostToDevice));

	shfl_xor_sum <<<1, block >>> (d_oData, d_iData);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(h_oData, d_oData, nBytes, cudaMemcpyDeviceToHost));

	printf_s("shfl xor\t\t: ");
	printData(h_oData, nElem);

	// free memory
	checkCuda(cudaFree(d_iData));
	checkCuda(cudaFree(d_oData));

	return EXIT_SUCCESS;

}


