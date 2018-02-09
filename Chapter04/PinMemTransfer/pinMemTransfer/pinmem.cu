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

int main(int argc, char** argv)
{
	// set up device 
	int  dev = 0;
	int  lShft = 22;

	checkCuda(cudaSetDevice(dev));

	if (argc > 1)	lShft = atoi(argv[1]);

	// memory size
	unsigned int  isize = 1 << lShft;
	unsigned int  nbytes = isize * sizeof(float);

	//get device information
	cudaDeviceProp  devProp;
	checkCuda(cudaGetDeviceProperties(&devProp, dev));
	printf_s("%s starting at ", argv[0]);
	printf_s("device %d: %s transfering  %d float's using %5.2fMB of memory from host to device and back.\n", dev, devProp.name, isize, nbytes / (1024.0f * 1024.0f));

	// allocate the host memory
	float  *h_a = NULL;
	checkCuda(cudaMallocHost((float**)&h_a, nbytes));

	// allocate the device memory
	float *d_a = NULL;
	checkCuda(cudaMalloc((float**)&d_a, nbytes));

	// initialize the host memory
	for (unsigned int i = 0; i < isize; i++) h_a[i] = 0.5f;

	// transfer data from the host to the device
	checkCuda(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

	// transfer data from the device to the host
	checkCuda(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

	// free memory
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFreeHost(h_a));

	// reset device
	checkCuda(cudaDeviceReset());

	return EXIT_SUCCESS;

}

