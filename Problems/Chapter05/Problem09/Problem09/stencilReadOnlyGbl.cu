#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


/* The global memory kernel is extremely slow compared to the other two taking almost 1500us to complete while
*  the other two finish in 550 or less. The constant memory kernel is the fastest which makes it clear that when
*  it can be utilized.
*/

#define	 RADIUS 4
#define  BDIM   64 // for best performance

// constant memory
__constant__ float coef[RADIUS + 1];

// FD coeffecient
#define a0     0.00000f
#define a1     0.80000f
#define a2    -0.20000f
#define a3     0.03809f
#define a4    -0.00357f

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

void initialData(float *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		in[i] = (float)(rand() & 0xFF) / 100.0f;
	}
}

void printData(float *in, const int size)
{
	for (int i = RADIUS; i < size; i++)
	{
		printf_s("%f ", in[i]);
	}

	printf_s("\n");
}

void setup_coef_constant(void)
{
	const float hc_coef[] = { a0, a1, a2, a3, a4 };
	checkCuda(cudaMemcpyToSymbol(coef, hc_coef, (RADIUS + 1) * sizeof(float)));
}

void cpu_stencil_1d(float *in, float *out, int isize)
{
	for (int i = RADIUS; i <= isize; i++)
	{
		float tmp = a1 * (in[i + 1] - in[i - 1])
			+ a2 * (in[i + 2] - in[i - 2])
			+ a3 * (in[i + 3] - in[i - 3])
			+ a4 * (in[i + 4] - in[i - 4]);
		out[i] = tmp;
	}
}

void checkResult(float *hostRef, float *gpuRef, const int size)
{
	double epsilon = 1.0E-6;
	bool match = 1;

	for (int i = RADIUS; i < size; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf_s("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
			break;
		}
	}

	if (!match) printf_s("Arrays do not match.\n\n");
}

__global__ void stencil_1d(float *in, float *out, const int N)
{
	// shared memory
	__shared__ float smem[BDIM + 2 * RADIUS];

	//index to global memory
	int  idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < N)
	{
		// index to shared memory for stencil calculatioin
		int sidx = threadIdx.x + RADIUS;

		// Read data from global memory into shared memory
		smem[sidx] = in[idx];

		// read halo part to shared memory
		if (threadIdx.x < RADIUS)
		{
			smem[sidx - RADIUS] = in[idx - RADIUS];
			smem[sidx + BDIM]   = in[idx + BDIM];
		}

		// Synchronize (ensure all data is available)
		__syncthreads();

		// Apply the stencil
		float  tmp = 0.0f;

#pragma unroll
		for (int i = 1; i <= RADIUS; i++)
		{
			tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
		}

		// Store the result
		out[idx] = tmp;
		
		idx += gridDim.x * blockDim.x;
	}
}

__global__ void stencil_1d_read_only(float *in, float *out, const float *__restrict__ dcoef, const int N)
{
	// shared memory
	__shared__ float smem[BDIM + 2 * RADIUS];

	//index to global memory
	int  idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	while (idx < N)
	{
		// index to shared memory for stencil calculatioin
		int sidx = threadIdx.x + RADIUS;

		// Read data from global memory into shared memory
		smem[sidx] = in[idx];
		// read halo part to shared memory

		if (threadIdx.x < RADIUS)
		{
			smem[sidx - RADIUS] = in[idx - RADIUS];
			smem[sidx + BDIM]   = in[idx + BDIM];
		}

		// Synchronize (ensure all data is available)
		__syncthreads();

		// Apply the stencil
		float  tmp = 0.0f;

#pragma unroll
		for (int i = 1; i <= RADIUS; i++)
		{
			tmp += dcoef[i] * (smem[sidx + i] - smem[sidx - i]);
		}

		// Store the result
		out[idx] = tmp;

		idx += gridDim.x * blockDim.x;
	}
}

__global__ void stencil_1d_global(float *in, float *out, float *dcoef, const int N)
{
	__shared__ float smem[BDIM + 2 * RADIUS];
	//index to global memory
	int  idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < N)
	{
		// index to shared memory
		int sidx = threadIdx.x + RADIUS;

		// Read data from global memory into shared memory
		smem[sidx] = in[idx];
		// read halo part to shared memory

		if (threadIdx.x < RADIUS)
		{
			smem[sidx - RADIUS] = in[idx - RADIUS];
			smem[sidx + BDIM]   = in[idx + BDIM];
		}

		// Synchronize (ensure all data is available)
		__syncthreads();

		// Apply the stencil
		float  tmp = 0.0f;

#pragma unroll
		for (int i = 1; i <= RADIUS; i++)
		{
			tmp += dcoef[i] * (smem[sidx + i] - smem[sidx - i]);
		}

		// Store the result
		out[idx] = tmp;

		idx += gridDim.x * blockDim.x;
	}
}

int main(int argc, char **argv)
{
	int  dev = 0;
	cudaDeviceProp  deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
	printf_s("%s starting 1D stencil calculation at ", argv[0]);
	printf_s("device %d: %s ", dev, deviceProp.name);
	checkCuda(cudaSetDevice(dev));

	// set up data size
	int isize = 1 << 24;

	size_t nBytes = (isize + 2 * RADIUS) * sizeof(float);
	printf_s("array size: %d ", isize);

	bool iprint = 0;

	// allocate host memory
	float *h_in		 = (float *)malloc(nBytes);
	float *hostRef	 = (float *)malloc(nBytes);
	float *gpuRef	 = (float *)malloc(nBytes);

	// allocate device memory
	float  *d_in, *d_out, *d_coef;
	checkCuda(cudaMalloc(&d_in, nBytes));
	checkCuda(cudaMalloc(&d_out, nBytes));
	checkCuda(cudaMalloc(&d_coef, (RADIUS + 1) * sizeof(float)));

	// set up coefficient in global memory
	const float h_coef[] = { a0, a1, a2, a3, a4 };
	checkCuda(cudaMemcpy(d_coef, h_coef, (RADIUS + 1) * sizeof(float), cudaMemcpyHostToDevice));

	// initialize host array
	initialData(h_in, isize + 2 * RADIUS);

	// Copy to device
	checkCuda(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

	// set up constant memory
	setup_coef_constant();

	// launch configuration
	cudaDeviceProp info;
	checkCuda(cudaGetDeviceProperties(&info, 0));

	dim3 block(BDIM, 1);
	dim3 grid(info.maxGridSize[0] < isize / block.x ? info.maxGridSize[0] : isize / block.x, 1);
	printf_s("(grid, block) %d,%d \n ", grid.x, block.x);

	// Launch stencil_1d() kernel on GPU
	stencil_1d <<<grid, block >>>(d_in + RADIUS, d_out + RADIUS, isize);

	// Copy result back to host
	checkCuda(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

	// apply cpu stencil
	cpu_stencil_1d(h_in, hostRef, isize);

	// check results
	checkResult(hostRef, gpuRef, isize);

	// launch read only cache kernel
	stencil_1d_read_only <<<grid, block >>>(d_in + RADIUS, d_out + RADIUS, d_coef, isize);
	checkCuda(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, isize);

	// launch global memory kernel
	stencil_1d_global <<<grid, block >>>(d_in + RADIUS, d_out + RADIUS, d_coef, isize);
	checkCuda(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, isize);

	// print out results
	if (iprint)
	{
		printData(gpuRef, isize);
		printData(hostRef, isize);
	}

	// Cleanup
	checkCuda(cudaFree(d_in));
	checkCuda(cudaFree(d_out));
	checkCuda(cudaFree(d_coef));
	free(h_in);
	free(hostRef);
	free(gpuRef);

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}