#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


/*
* This example demonstrates the use of CUDA managed memory to implement matrix
* addition. In this example, arbitrary pointers can be dereferenced on the host
* and device. CUDA will automatically manage the transfer of data to and from
* the GPU as needed by the application. There is no need for the programmer to
* use cudaMemcpy, cudaHostGetDevicePointer, or any other CUDA API involved with
* explicitly transferring data. In addition, because CUDA managed memory is not
* forced to reside in a single place it can be transferred to the optimal
* memory space and not require round-trips over the PCIe bus every time a
* cross-device reference is performed (as is required with zero copy and UVA).
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
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
	int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
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
	int ishift = 13;

	if (argc > 1) ishift = atoi(argv[1]);

	nx = ny = 1 << ishift;

	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	printf_s("Matrix size: nx %d ny %d\n", nx, ny);

	// malloc host memory
	float *A, *B, *hostRef, *gpuRef;
	checkCuda(cudaMallocManaged(&A, nBytes));
	checkCuda(cudaMallocManaged(&B, nBytes));
	checkCuda(cudaMallocManaged(&gpuRef, nBytes));
	checkCuda(cudaMallocManaged(&hostRef, nBytes));

	// initialize data at host side
	
	initialData(A, nxy);
	initialData(B, nxy);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// add matrix at host side for result checks
	sumMatrixOnHost(A, B, hostRef, nx, ny);

	// invoke kernel at host side
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);



	// warm-up kernel, with unified memory all pages will migrate from host to
	// device
	sumMatrixGPU << <grid, block >> >(A, B, gpuRef, 1, 1);
	
	sumMatrixGPU <<<grid, block >>>(A, B, gpuRef, nx, ny);

	checkCuda(cudaDeviceSynchronize());
	printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

	// check kernel error
	checkCuda(cudaGetLastError());

	// check device results
	checkResult(hostRef, gpuRef, nxy);

	// free device global memory
	checkCuda(cudaFree(A));
	checkCuda(cudaFree(B));
	checkCuda(cudaFree(hostRef));
	checkCuda(cudaFree(gpuRef));

	// reset device
	checkCuda(cudaDeviceReset());

	return (0);
}
