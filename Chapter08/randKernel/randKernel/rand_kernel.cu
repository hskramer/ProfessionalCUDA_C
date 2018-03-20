#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

/*
* This example demonstrates two techniques for using the cuRAND host and device
* API to generate random numbers for CUDA kernels to consume.
*/

int threads_per_block = 256;
int blocks_per_grid = 30;

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



/*
* host_api_kernel consumes pre-generated random values from the cuRAND host API
* to perform some dummy computation.
*/
__global__ void host_api_kernel(float *randomValues, float *out, int N)
{
	int i;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int nthreads = gridDim.x * blockDim.x;

	for (i = tid; i < N; i += nthreads)
	{
		float rand = randomValues[i];
		rand = rand * 2;
		out[i] = rand;
	}
}

/*
* device_api_kernel uses the cuRAND device API to generate random numbers
* on-the-fly on the GPU, and then performs some dummy computation using them.
*/
__global__ void device_api_kernel(curandState *states, float *out, int N)
{
	int i;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int nthreads = gridDim.x * blockDim.x;
	curandState *state = states + tid;

	curand_init(9584, tid, 0, state);

	for (i = tid; i < N; i += nthreads)
	{
		float rand = curand_uniform(state);
		rand = rand * 2;
		out[i] = rand;
	}
}

/*
* use_host_api is an examples usage of the cuRAND host API to generate random
* values to be consumed on the device.
*/
void use_host_api(int N)
{
	int i;
	curandGenerator_t randGen;
	float *dRand, *dOut, *hOut;

	// Create cuRAND generator (i.e. handle)
	curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);

	// Allocate device memory to store the random values and output
	checkCuda(cudaMalloc((void **)&dRand, sizeof(float) * N));
	checkCuda(cudaMalloc((void **)&dOut, sizeof(float) * N));
	hOut = (float *)malloc(sizeof(float) * N);

	// Generate N random values from a uniform distribution
	curandGenerateUniform(randGen, dRand, N);

	// Consume the values generated by curandGenerateUniform
	host_api_kernel <<<blocks_per_grid, threads_per_block >>>(dRand, dOut, N);

	// Retrieve outputs
	checkCuda(cudaMemcpy(hOut, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));

	printf("Sampling of output from host API:\n");

	for (i = 0; i < 10; i++)
	{
		printf("%2.4f\n", hOut[i]);
	}

	printf("...\n");

	free(hOut);
	checkCuda(cudaFree(dRand));
	checkCuda(cudaFree(dOut));
	curandDestroyGenerator(randGen);
}

/*
* use_device_api is an examples usage of the cuRAND device API to use the GPU
* to generate random values on the fly from inside a CUDA kernel.
*/
void use_device_api(int N)
{
	int i;
	static curandState *states = NULL;
	float *dOut, *hOut;

	/*
	* Allocate device memory to store the output and cuRAND device state
	* objects (which are analogous to handles, but on the GPU).
	*/
	checkCuda(cudaMalloc((void **)&dOut, sizeof(float) * N));
	checkCuda(cudaMalloc((void **)&states, sizeof(curandState) * threads_per_block * blocks_per_grid));
	hOut = (float *)malloc(sizeof(float) * N);

	// Execute a kernel that generates and consumes its own random numbers
	device_api_kernel <<<blocks_per_grid, threads_per_block >>>(states, dOut, N);

	// Retrieve the results
	checkCuda(cudaMemcpy(hOut, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));

	printf("Sampling of output from device API:\n");

	for (i = 0; i < 10; i++)
	{
		printf("%2.4f\n", hOut[i]);
	}

	printf("...\n");

	free(hOut);
	checkCuda(cudaFree(dOut));
	checkCuda(cudaFree(states));
}

int main(int argc, char **argv)
{
	int N = 8388608;

	use_host_api(N);
	use_device_api(N);

	return 0;
}
