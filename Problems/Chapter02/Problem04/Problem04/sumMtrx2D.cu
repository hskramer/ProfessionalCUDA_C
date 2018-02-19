#include <stdio.h>
#include <stdlib.h>
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


void initialData(float *ip, const int size)
{
	int i;

	for (i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;

	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			ic[ix] = ia[ix] + ib[ix];

		}

		ia += nx;
		ib += nx;
		ic += nx;
	}

	return;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
			break;
		}
	}

	if (match)
		printf("Arrays match.\n\n");
	else
		printf("Arrays do not match.\n\n");
}

// grid 2D block 1D
__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = blockIdx.y;
	unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny)
		MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// set up data size of matrix
	int nx = 1 << 14;
	int ny = 1 << 14;

	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	printf("Matrix size: nx %d ny %d\n", nx, ny);

	// malloc host memory
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	
	// add matrix at host side for result checks
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);


	// malloc device global memory
	float *d_MatA, *d_MatB, *d_MatC;
	CHECK(cudaMalloc((void **)&d_MatA, nBytes));
	CHECK(cudaMalloc((void **)&d_MatB, nBytes));
	CHECK(cudaMalloc((void **)&d_MatC, nBytes));

	// transfer data from host to device
	CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

	// invoke kernel at host side
	int dimx = 32;
	dim3 block(dimx, 1);
	dim3 grid((nx + block.x - 1) / block.x, ny);

	
	sumMatrixOnGPUMix <<<grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
	CHECK(cudaDeviceSynchronize());
	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nxy);

	// free device global memory
	CHECK(cudaFree(d_MatA));
	CHECK(cudaFree(d_MatB));
	CHECK(cudaFree(d_MatC));

	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	// reset device
	CHECK(cudaDeviceReset());

	return 0;
}