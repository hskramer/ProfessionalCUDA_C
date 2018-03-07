#include <stdio.h>
#include <stdlib.h>
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

void printData(int *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		printf_s("%2d ", in[i]);
	}

	printf_s("\n");
}

__global__ void test_shfl_broadcast(int *d_out, int *d_in, int const srcLane)
{
	int value = d_in[threadIdx.x];

	value = __shfl(value, srcLane, BDIMX);
	d_out[threadIdx.x] = value;
}

__global__ void test_shfl_up(int *d_out, int *d_in, unsigned int const delta)
{
	int  value = d_in[threadIdx.x];

	value = __shfl_up(value, delta, BDIMX);
	d_out[threadIdx.x] = value;
}

__global__ void test_shfl_down(int *d_out, int *d_in, unsigned int delta)
{
	int value = d_in[threadIdx.x];

	value = __shfl_down(value, delta, BDIMX);
	d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor(int *d_out, int *d_in, int const mask)
{
	int value = d_in[threadIdx.x];

	value = __shfl_xor(value, mask, BDIMX);
	d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor_array(int *d_out, int *d_in, int const mask)
{
	int idx = threadIdx.x * SEGM;
	int value[SEGM];

	for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

	value[0] = __shfl_xor(value[0], mask, BDIMX);
	value[1] = __shfl_xor(value[1], mask, BDIMX);
	value[2] = __shfl_xor(value[2], mask, BDIMX);
	value[3] = __shfl_xor(value[3], mask, BDIMX);

	for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}

__inline__ __device__
void swap(int *value, int laneIdx, int mask, int firstIdx, int secondIdx)
{
	bool pred = ((laneIdx / mask + 1) == 1);

	if (pred)
	{
		int tmp = value[firstIdx];
		value[firstIdx] = value[secondIdx];
		value[secondIdx] = tmp;
	}

	value[secondIdx] = __shfl_xor(value[secondIdx], mask, BDIMX);

	if (pred)
	{
		int tmp = value[firstIdx];
		value[firstIdx] = value[secondIdx];
		value[secondIdx] = tmp;
	}
}

__global__ void test_shfl_swap(int *d_out, int *d_in, int const mask, int firstIdx,	int secondIdx)
{
	int idx = threadIdx.x * SEGM;
	int value[SEGM];

	for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

	swap(value, threadIdx.x, mask, firstIdx, secondIdx);

	for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}


int main(int argck, char **argv)
{
	int  dev = 0;
	bool  iPrintout = 1;

	cudaDeviceProp  deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("> %s starting. ", argv[0]);
	printf_s("on Device %d: %s\n", dev, deviceProp.name);
	
	checkCuda(cudaSetDevice(dev));

	// define variables and set values
	int  nElem = BDIMX;
	int  h_inData[BDIMX], h_outData[BDIMX];

	for (int i = 0; i < nElem; i++)  h_inData[i] = i;

	if (iPrintout)
	{
		printf_s("initialData\t\t: ");
		printData(h_inData, nElem);
	}

	size_t  nBytes = nElem * sizeof(int);
	int  *d_inData, *d_outData;

	checkCuda(cudaMalloc((int**)&d_inData, nBytes));
	checkCuda(cudaMalloc((int**)&d_outData, nBytes));

	checkCuda(cudaMemcpy(d_inData, h_inData, nBytes, cudaMemcpyHostToDevice));

	int  block = BDIMX;

	// shfl bcast
	test_shfl_broadcast <<<1, block >>>(d_outData, d_inData, 2);
	checkCuda(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

	if (iPrintout)
	{
		printf_s("shfl bcast\t\t: ");
		printData(h_outData, BDIMX);
	}

	// shfl up
	test_shfl_up <<<1, block >>> (d_outData, d_inData, 2);
	checkCuda(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

	if (iPrintout)
	{
		printf_s("shfl up \t\t: ");
		printData(h_outData, BDIMX);
	}

	// shfl down
	test_shfl_down <<<1, BDIMX >>> (d_outData, d_inData, 2);
	checkCuda(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

	if (iPrintout)
	{
		printf_s("shfl down \t\t: ");
		printData(h_outData, BDIMX);
	}

	// shfl with wrap does not work as given in text or download

	// shfl xor with 1 will result in adjacent threads exchanging values(butterfly exchange)
	test_shfl_xor <<<1, BDIMX >>> (d_outData, d_inData, 1);
	checkCuda(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

	if (iPrintout)
	{
		printf_s("shfl xor \t\t: ");
		printData(h_outData, BDIMX);
	}

	// test shfl xor with an array each thread will exchange data with another thread determined by a mask
	test_shfl_xor_array <<<1, BDIMX / SEGM >>> (d_outData, d_inData, 1);
	checkCuda(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

	if (iPrintout)
	{
		printf("shfl array 1\t\t: ");
		printData(h_outData, nElem);
	}

	//  swap using __shfl_xor
	test_shfl_swap <<<1, block / SEGM >>>(d_outData, d_inData, 1, 0, 3);
	checkCuda(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));
	

	if (iPrintout)
	{
		printf("shfl swap inline\t: ");
		printData(h_outData, nElem);
	}
	
	// finishing
	checkCuda(cudaFree(d_inData));
	checkCuda(cudaFree(d_outData));
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}