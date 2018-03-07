#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


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
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}


void checkResult(float *hostRef, float *gpuRef, const int size)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < size; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf_s("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
			break;
		}

	}

	if (!match)  printf_s("Arrays do not match.\n\n");
}

__global__ void transposeDiagonalColUnroll4(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int blk_y = blockIdx.x;
	unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

	unsigned int ix_stride = blockDim.x * blk_x;

	unsigned int ix = ix_stride * 4 + threadIdx.x;
	unsigned int iy = blockDim.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix]					= in[ix * ny + iy];
		out[iy * nx + ix + blockDim.x]		= in[(ix + blockDim.x) * ny + iy];
		out[iy * nx + ix + 2 * blockDim.x]	= in[(ix + 2 * blockDim.x) * ny + iy];
		out[iy * nx + ix + 3 * blockDim.x]	= in[(ix + 3 * blockDim.x) * ny + iy];
	}
}

int main(int argc, char **argv)
{

	 // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


	
    // set up array size 2048
    int nx = 1 << 11;
    int ny = 1 << 11;

    // select a kernel and block size
    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) iKernel = atoi(argv[1]);

    if (argc > 2) blockx  = atoi(argv[2]);

    if (argc > 3) blocky  = atoi(argv[3]);

    if (argc > 4) nx  = atoi(argv[4]);

    if (argc > 5) ny  = atoi(argv[5]);



}