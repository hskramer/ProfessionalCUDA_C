#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#include <ctime>
#include <cuda_runtime.h>

inline
void checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		printf("Error": %s : %d, ", __FILE__, __LINE__);
			printf("CUDA Runtime Error: %d: %s\n", result, cudaGetErrorString(result));
		exit(1);
	}
#endif
}

void checkCudaResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
				gpuRef[i], i);
			break;
		}
	}

	if (match) printf("Arrays match.\n\n");

	return;
}

void initialData(float *ip, int size)
{
	// generate different seed for random number
	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
	for (int idx = 0; idx < N; idx++)
	{
		C[idx] = A[idx] + B[idx];
	}
}
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// set up data size of vectors
	int nElem = 1 << 24;
	printf("Vector size %d\n", nElem);

	// malloc host memory
	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;

	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	float iElaps;
	clock_t	iStart, iStop;

	// initialize data at host side
	iStart = clock();
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	iStop = clock();
	iElaps = float(iStop - iStart) / CLOCKS_PER_SEC;

	printf("initialData Time elapsed %f sec\n", iElaps);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// add vector at host side for result checkCudas
	iStart = clock();
	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	iStop = clock();
	iElaps = float(iStop - iStart) / CLOCKS_PER_SEC;

	printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

	// malloc device global memory
	float *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc((float**)&d_A, nBytes));
	checkCuda(cudaMalloc((float**)&d_B, nBytes));
	checkCuda(cudaMalloc((float**)&d_C, nBytes));

	// transfer data from host to device
	checkCuda(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

	// invoke kernel at host side
	int iLen = 256;
	dim3 block(iLen);
	dim3 grid((nElem + block.x - 1) / block.x);

	iStart = clock();
	sumArraysOnGPU << <grid, block >> >(d_A, d_B, d_C, nElem);
	checkCuda(cudaDeviceSynchronize());
	iStop = clock();
	iElaps = float(iStop - iStart) / CLOCKS_PER_SEC;

	printf("sumArraysOnGPU <<< %d, %d >>>  Time elapsed %f sec\n", grid.x, block.x, iElaps);

	// checkCuda kernel error
	checkCuda(cudaGetLastError());

	// copy kernel result back to host side
	checkCuda(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	// checkCuda device results
	checkCudaResult(hostRef, gpuRef, nElem);

	// free device global memory
	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_B));
	checkCuda(cudaFree(d_C));

	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	return(0);
}
