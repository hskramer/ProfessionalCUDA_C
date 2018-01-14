#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(void)
{
	if(threadIdx.x == 5)
	printf("Hello World from GPU thread %d\n", threadIdx.x);
}

int main(void)
{
	printf("Hello World from CPU.\n");

	helloFromGPU << <1, 10 >> > ();

	cudaDeviceSynchronize();

	return 0;
}