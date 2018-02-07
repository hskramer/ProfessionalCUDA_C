#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


/*
* This example demonstrates using explicit CUDA memory transfer to implement
* matrix addition. This code contrasts with sumMatrixGPUManaged.cu, where CUDA
* managed memory is used to remove all explicit memory transfers and abstract
* away the concept of physicall separate address spaces.
*/

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
			printf_s("host %f gpu %f\n", hostRef[i], gpuRef[i]);
			break;
		}
	}

	if (!match)
	{
		printf_s("Arrays do not match.\n\n");
	}
}

// grid 2D block 2D
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny)
	{
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}

int main(int argc, char **argv)
{
	printf_s("%s Starting ", argv[0]);

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("using Device %d: %s\n", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// set up data size of matrix
	int nx, ny;
	int ishift = 12;

	if (argc > 1) ishift = atoi(argv[1]);

	nx = ny = 1 << ishift;

	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	printf_s("Matrix size: nx %d ny %d\n", nx, ny);

	// malloc host memory
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A	     = (float *)malloc(nBytes);
	h_B		 = (float *)malloc(nBytes);
	hostRef  = (float *)malloc(nBytes);
	gpuRef   = (float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_A, nxy);
	initialData(h_B, nxy);

	// add matrix at host side for result checks
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

	// malloc device global memory
	float *d_MatA, *d_MatB, *d_MatC;
	checkCuda(cudaMalloc((void **)&d_MatA, nBytes));
	checkCuda(cudaMalloc((void **)&d_MatB, nBytes));
	checkCuda(cudaMalloc((void **)&d_MatC, nBytes));

	// invoke kernel at host side
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// init device data to 0.0f, then warm-up kernel to obtain accurate timing
	// result
	checkCuda(cudaMemset(d_MatA, 0.0f, nBytes));
	checkCuda(cudaMemset(d_MatB, 0.0f, nBytes));
	sumMatrixGPU <<<grid, block >>>(d_MatA, d_MatB, d_MatC, 1, 1);


	// transfer data from host to device
	checkCuda(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));


	sumMatrixGPU <<<grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);

	checkCuda(cudaDeviceSynchronize());
	printf_s("sumMatrix on gpu : <<<(%d,%d), (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	checkCuda(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

	// check kernel error
	checkCuda(cudaGetLastError());

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

	return (0);
}