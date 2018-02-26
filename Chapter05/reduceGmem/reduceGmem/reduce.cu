#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define DIM 256

extern __shared__ int smem[];

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


// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
	if (size == 1) return data[0];

	int const stride = size / 2;

	for (int i = 0; i < stride; i++)
		data[i] += data[i + stride];

	return recursiveReduce(data, stride);
}

__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n)
{
	// set thread ID
	unsigned int tid = threadIdx.x;
	int *idata = g_idata + blockIdx.x * blockDim.x;

	// boundary check
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n) return;

	/* These if statements are not needed when using a thread DIM of 128 or in my case 256,
	 * kernel performance increased over 1.30 times.
	/*/ 
	//if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

	//__syncthreads();

	//if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

	//__syncthreads();

	if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

	__syncthreads();

	if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

	__syncthreads();

	// unrolling warp
	if (tid < 32)
	{
		volatile int *vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n)
{
	__shared__ int smem[DIM];

	// set thread ID
	unsigned int tid = threadIdx.x;

	// boundary check
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n) return;

	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x;

	// set to smem by each threads
	smem[tid] = idata[tid];
	__syncthreads();

	// in-place reduction in shared memory you don't need these if statements 
	// and they only slow down the kernel the results match.
	//if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];

	//__syncthreads();

	//if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];

	//__syncthreads();

	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];

	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];

	__syncthreads();

	// unrolling warp
	if (tid < 32)
	{
		volatile int *vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

// unroll4 + complete unroll for loop + gmem
__global__ void reduceGmemUnroll4(int *g_idata, int *g_odata, unsigned int n)
{
	// set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x * 4;

	// unrolling 4
	if (idx < n)
	{
		int a1, a2, a3, a4;
		a1 = a2 = a3 = a4 = 0;
		a1 = g_idata[idx];
		if (idx + blockDim.x < n)	  a2 = g_idata[idx + blockDim.x];
		if (idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
		if (idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];

		g_idata[idx] = a1 + a2 + a3 + a4;
	}

	__syncthreads();

	// I'm removing unnecessary if/syncthreads statements.
	
	if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

	__syncthreads();

	if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

	__syncthreads();

	// unrolling warp
	if (tid < 32)
	{
		volatile int *vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmemUnroll8(int *g_idata, int *g_odata, unsigned int n)
{
	// static shared memory
	__shared__ int smem[DIM];

	// set thread ID
	unsigned int tid = threadIdx.x;

	// global index, 8 blocks of input data processed at a time
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	// unrolling 4 blocks
	int tmpSum = 0;

	// since I created 2X the number of threads I decided to do an unroll eight there was  performance benefit
	// boundary check
	if (idx < n)
	{
		int a1, a2, a3, a4;
		int b1, b2, b3, b4;
		a1 = a2 = a3 = a4 = 0;
		b1 = b2 = b3 = b4 = 0;
		a1 = g_idata[idx];
		if (idx + blockDim.x < n)	  a2 = g_idata[idx + blockDim.x];
		if (idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
		if (idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];
		if (idx + 4 * blockDim.x < n) b1 = g_idata[idx + 4 * blockDim.x];
		if (idx + 5 * blockDim.x < n) b2 = g_idata[idx + 5 * blockDim.x];
		if (idx + 6 * blockDim.x < n) b3 = g_idata[idx + 6 * blockDim.x];
		if (idx + 7 * blockDim.x < n) b4 = g_idata[idx + 7 * blockDim.x];

		tmpSum = a1 + a2 + a3 + a4 +b1 + b2 + b3 + b4;
	}

	smem[tid] = tmpSum;
	__syncthreads();

	// removed if/sync statements

	if (blockDim.x >= 256 && tid < 128)  smem[tid] += smem[tid + 128];

	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)   smem[tid] += smem[tid + 64];

	__syncthreads();

	// unrolling warp
	if (tid < 32)
	{
		volatile int *vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnrollDyn8(int *g_idata, int *g_odata, unsigned int n)
{
	// using dynamically allocated shared memory

	// set thread ID
	unsigned int tid = threadIdx.x;

	// global index, 8 blocks of input data processed at a time
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	// unrolling 4 blocks
	int tmpSum = 0;

	// since I created 2X the number of threads I decided to do an unroll eight there was  performance benefit
	// boundary check
	if (idx < n)
	{
		int a1, a2, a3, a4;
		int b1, b2, b3, b4;
		a1 = a2 = a3 = a4 = 0;
		b1 = b2 = b3 = b4 = 0;
		a1 = g_idata[idx];
		if (idx + blockDim.x < n)	  a2 = g_idata[idx + blockDim.x];
		if (idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
		if (idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];
		if (idx + 4 * blockDim.x < n) b1 = g_idata[idx + 4 * blockDim.x];
		if (idx + 5 * blockDim.x < n) b2 = g_idata[idx + 5 * blockDim.x];
		if (idx + 6 * blockDim.x < n) b3 = g_idata[idx + 6 * blockDim.x];
		if (idx + 7 * blockDim.x < n) b4 = g_idata[idx + 7 * blockDim.x];

		tmpSum = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}

	smem[tid] = tmpSum;
	__syncthreads();

	// removed if/sync statements

	if (blockDim.x >= 256 && tid < 128)  smem[tid] += smem[tid + 128];

	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)   smem[tid] += smem[tid + 64];

	__syncthreads();

	// unrolling warp
	if (tid < 32)
	{
		volatile int *vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

int main(int argc, char **argv)
{
	// set up device
	int  dev = 0;
	cudaDeviceProp  deviceProp;

	checkCuda(cudaGetDeviceProperties(&deviceProp, dev));

	printf_s("%s starting reduction at ", argv[0]);
	printf_s("device %d: %s", dev, deviceProp);
	checkCuda(cudaSetDevice(dev));

	bool  bResult = false;

	// initialization
	int  nElem = 1 << 24; // number elements
	printf("    with array size %d  ", nElem);

	// execution configuration
	int  blocksize = DIM;   // initial block size

	dim3  block(blocksize, 1);
	dim3  grid((nElem + block.x - 1) / block.x, 1);
	printf("grid %d block %d\n", grid.x, block.x);
	 
	// allocate host memory
	size_t  nBytes = nElem * sizeof(int);
	int  *h_idata  = (int *)malloc(nBytes);
	int  *h_odata  = (int *)malloc(grid.x * sizeof(int));
	int  *tmp	   = (int *)malloc(nBytes);


	// initialize the array
	for (int i = 0; i < nElem; i++)
	{
		h_idata[i] = (int)(rand() & 0xFF);
	}

	memcpy(tmp, h_idata, nBytes);

	int gpu_sum = 0;

	//allocate device memory
	int  *d_idata;
	int  *d_odata;

	checkCuda(cudaMalloc(&d_idata, nBytes));
	checkCuda(cudaMalloc(&d_odata, grid.x * sizeof(int)));

	// cpu reduction
	int cpu_sum = recursiveReduce(tmp, nElem);
	printf("cpu reduce             : %d\n", cpu_sum);

	// reduce global memory
	checkCuda(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
	reduceGmem <<<grid.x, block >>> (d_idata, d_odata, nElem);
	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
	printf_s("reduceGmem             : %d  <<<grid %d  block %d>>>\n", gpu_sum, grid.x, block.x);

	// reduce shared memory is 1.6 times as fast as global memory
	checkCuda(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
	reduceSmem <<<grid.x, block >>>(d_idata, d_odata, nElem);
	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
	printf("reduceSmem             : %d  <<<grid %d  block %d>>>\n", gpu_sum, grid.x, block.x);


	// reduce global memory unroll 4 this kernel performed 
	checkCuda(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
	reduceGmemUnroll4<<<grid.x / 4, block >>>(d_idata, d_odata, nElem);
	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));

	gpu_sum = 0;
	for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];
	printf_s("reduceGmemUnroll4      : %d  <<<grid %d  block %d>>>\n", gpu_sum, grid.x / 4, block.x);


	// this kernel was 1.6 times as fast as reduce shared and 1.5 times global unroll 4
	checkCuda(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
	reduceSmemUnroll8 <<<grid.x / 8, block >>>(d_idata, d_odata, nElem);
	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

	gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
	printf_s("reduceSmemUnroll8      : %d  <<<grid %d   block %d>>>\n", gpu_sum, grid.x / 8, block.x);

	// performance was exactly the same as the static kernel 
	checkCuda(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
	reduceSmemUnrollDyn8 <<<grid.x / 8, block, DIM * sizeof(int) >>>(d_idata, d_odata, nElem);
	checkCuda(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

	gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
	printf_s("reduceSmemUnrollDyn8   : %d  <<<grid %d   block %d>>>\n", gpu_sum, grid.x / 8, block.x);


	// free device memory
	checkCuda(cudaFree(d_idata));
	checkCuda(cudaFree(d_odata));

	// reset device
	checkCuda(cudaDeviceReset());

	// check the results
	bResult = (gpu_sum == cpu_sum);
	if (!bResult) printf("Test failed!\n");

	free(h_idata);
	free(h_odata);
	free(tmp);

	return  EXIT_SUCCESS;

}