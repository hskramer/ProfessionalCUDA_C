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

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
			printf("Arrays do not match.\n\n");
			break;
		}
	}
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = iy * NX + ix;

	if (ix < NX && iy < NY)
	{
		C[idx] = A[idx] + B[idx];
	}
}

int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	checkCuda(cudaSetDevice(dev));

	// set up data size of matrix
	int nx = 1 << 14;
	int ny = 1 << 14;

	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	// malloc host memory
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_A, nxy);
	initialData(h_B, nxy);


	// add matrix at host side for result checkCudas
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

	// malloc device global memory
	float *d_MatA, *d_MatB, *d_MatC;
	checkCuda(cudaMalloc((void **)&d_MatA, nBytes));
	checkCuda(cudaMalloc((void **)&d_MatB, nBytes));
	checkCuda(cudaMalloc((void **)&d_MatC, nBytes));

	// transfer data from host to device
	checkCuda(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

	// invoke kernel at host side
	int dimx = 32;
	int dimy = 32;

	if (argc > 2)
	{
		dimx = atoi(argv[1]);
		dimy = atoi(argv[2]);
	}

	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// execute the kernel
	checkCuda(cudaDeviceSynchronize());
	sumMatrixOnGPU2D << <grid, block >> >(d_MatA, d_MatB, d_MatC, nx, ny);
	checkCuda(cudaDeviceSynchronize());

	printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
	checkCuda(cudaGetLastError());

	// copy kernel result back to host side
	checkCuda(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nxy);

	// free device global memory
	checkCuda(cudaFree(d_MatA));
	checkCuda(cudaFree(d_MatB));
	checkCuda(cudaFree(d_MatC));

	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());

	return 0;
}