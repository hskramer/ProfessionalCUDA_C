#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


/*
* M = # of rows
* N = # of columns
*/
int M = 1024;
int N = 1024;

void generate_random_dense_matrix(int M, int N, float **outA)
{
	int i, j;
	double rMax = (double)RAND_MAX;
	float *A = (float *)malloc(sizeof(float) * M * N);

	// For each column
	for (j = 0; j < N; j++)
	{
		// For each row
		for (i = 0; i < M; i++)
		{
			double dr = (double)rand();
			A[j * M + i] = (dr / rMax) * 100.0;
		}
	}

	*outA = A;
}

int main(int argc, char **argv)
{
	float	*A, *d_A;
	float	*B, *d_B;
	float	*C, *d_C;
	float	alpha = 3.0f, beta = 5.0f;
		
	cudaStream_t	stream_1;
	cublasHandle_t	handle = 0;

	//generate two matrices
	srand(2468);
	generate_random_dense_matrix(M, N, &A);
	generate_random_dense_matrix(M, N, &B);

	C = (float *)malloc(M * N * sizeof(float));

	//create cublas handle and cuda stream
	CHECK(cudaStreamCreate(&stream_1));
	CHECK_CUBLAS(cublasCreate(&handle));

	//allocate device memoru
	CHECK(cudaMalloc((void **)&d_A, M * N * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_B, M * N * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

	//create cublas stream and set matrix async
	CHECK_CUBLAS(cublasSetStream(handle, stream_1));

	CHECK_CUBLAS(cublasSetMatrixAsync(M, N, sizeof(float), A, M, d_A, M, stream_1));
	CHECK_CUBLAS(cublasSetMatrixAsync(M, N, sizeof(float), B, M, d_B, M, stream_1));
	CHECK_CUBLAS(cublasSetMatrixAsync(M, N, sizeof(float), C, M, d_C, M, stream_1));

	//execute are matrix multiplication
	CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, N, &alpha, d_A, M, d_B, M, &beta, d_C, M));
	CHECK(cudaStreamSynchronize(stream_1));

	 // retrieve results
	CHECK_CUBLAS(cublasGetMatrixAsync(M, N, sizeof(float), d_C, M, C, M, stream_1));

	int i, j;

	for (j = 0; j < 10; j++)
	{
		for (i = 0; i < 10; i++)
		{
			printf_s("%2.2f ", C[j * M + i]);
		}
		printf_s("...\n");
	}


	// free all resources and destroy streams and handle
	free(A);
	free(B);
	free(C);

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));

	CHECK(cudaStreamDestroy(stream_1));

	CHECK_CUBLAS(cublasDestroy(handle));

	return EXIT_SUCCESS;

}