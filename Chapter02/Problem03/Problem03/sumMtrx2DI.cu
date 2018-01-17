#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

inline
void checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		printf("Error: %s : %d" , __FILE__, __LINE__);
			printf("CUDA Runtime Error: %d: %s\n", result, cudaGetErrorString(result));
		exit(1);
	}
#endif
}

void initialData(int *ip, const int size)
{
	int i;

	srand(time(NULL));

	for (i = 0; i < size; i++)
	{
		ip[i] = (int)(rand() % 25);
	}

	return;
}

void sumMatrixOnHost(int *A, int *B, int *C, const int nx, const int ny)
{
	int *ia = A;
	int *ib = B;
	int *ic = C;

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


void checkResult(int *hostRef, int *gpuRef, const int N)
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

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
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
	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// set up data size of matrix
	int nx = 1 << 14;
	int ny = 1 << 14;

	int nxy = nx * ny;
	int nBytes = nxy * sizeof(int);
	printf("Matrix size: nx %d ny %d\n", nx, ny);

	// malloc host memory
	int *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (int *)malloc(nBytes);
	h_B = (int *)malloc(nBytes);
	hostRef = (int *)malloc(nBytes);
	gpuRef = (int *)malloc(nBytes);

	// initialize data at host side

	initialData(h_A, nxy);
	initialData(h_B, nxy);

	// add matrix at host side for result checks

	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

	// malloc device global memory
	int *d_MatA, *d_MatB, *d_MatC;
	checkCuda(cudaMalloc((void **)&d_MatA, nBytes));
	checkCuda(cudaMalloc((void **)&d_MatB, nBytes));
	checkCuda(cudaMalloc((void **)&d_MatC, nBytes));

	// transfer data from host to device
	checkCuda(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

	// invoke kernel at host side
	int dimx = 32;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);


	sumMatrixOnGPU2D << <grid, block >> >(d_MatA, d_MatB, d_MatC, nx, ny);
	checkCuda(cudaDeviceSynchronize());

	printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
	// check kernel error
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

	return (0);
}