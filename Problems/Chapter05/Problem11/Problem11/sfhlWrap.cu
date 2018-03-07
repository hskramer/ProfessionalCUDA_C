#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define  BDIMX 16

/* This problem is simple once you understand that the initial values are the landID's (laneID = threadIdx.x % 32)
*  if a width is given it becomes % width. In this case we are using a width of 16 anf are initial values are
*  0...15 and from here we add the values held in the the srclane's of the remaining half warp plus the offset two.
*  The easy way to think about it is (laneID + 2) % 16 + initial value. (16 + 2) % 16 = 2 + 0 = 2 the first result and our last
*  is (31 + 2) % 16 = 1 + 15 = 16 are last result. If you change BDIMX to any multiple of two it becomes very clear.
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

__global__ void shfl_wrap(int *d_out, int *d_in)
{
	int value = d_in[threadIdx.x]; 
	value +=__shfl(value, threadIdx.x + 2, BDIMX); 
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

	shfl_wrap <<<1, block >>> (d_oData, d_iData);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaMemcpy(h_oData, d_oData, nBytes, cudaMemcpyDeviceToHost));

	printf_s("shfl wrap\t\t: ");
	printData(h_oData, nElem);

	// free memory
	checkCuda(cudaFree(d_iData));
	checkCuda(cudaFree(d_oData));

	return EXIT_SUCCESS;

}