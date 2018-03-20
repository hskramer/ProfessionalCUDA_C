#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>

/*
* This is an example demonstrating usage of the cuSPARSE library to perform a
* sparse matrix-vector multiplication on randomly generated data.
*/

/*
* M = # of rows
* N = # of columns
*/
int M = 5120;
int N = 5120;

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
* Generate a vector of length N with random single-precision floating-point
* values between 0 and 100.
*/
void generate_random_vector(int N, float **outX)
{
	int i;
	double rMax = (double)RAND_MAX;
	float *X = (float *)malloc(sizeof(float) * N);

	for (i = 0; i < N; i++)
	{
		int r = rand();
		double dr = (double)r;
		X[i] = (dr / rMax) * 100.0;
	}

	*outX = X;
}

/*
* Generate random dense matrix A in column-major order, while rounding some
* elements down to zero to ensure it is sparse.
*/
int generate_random_dense_matrix(int M, int N, float **outA)
{
	int i, j;
	double rMax = (double)RAND_MAX;
	float *A = (float *)malloc(sizeof(float) * M * N);
	int totalNnz = 0;

	for (j = 0; j < N; j++)
	{
		for (i = 0; i < M; i++)
		{
			int r = rand();
			float *curr = A + (j * M + i);

			if (r % 3 > 0)
			{
				*curr = 0.0f;
			}
			else
			{
				double dr = (double)r;
				*curr = (dr / rMax) * 100.0;
			}

			if (*curr != 0.0f)
			{
				totalNnz++;
			}
		}
	}

	*outA = A;
	return totalNnz;
}

int main(int argc, char **argv)
{
	int row;
	float *A, *dA;
	int *dNnzPerRow;
	float *dCsrValA;
	int *dCsrRowPtrA;
	int *dCsrColIndA;
	int totalNnz;
	float alpha = 3.0f;
	float beta = 4.0f;
	float *dX, *X;
	float *dY, *Y;

	cusparseHandle_t handle = 0;
	cusparseMatDescr_t descr = 0;

	// Generate input
	srand(9384);
	int trueNnz = generate_random_dense_matrix(M, N, &A);
	generate_random_vector(N, &X);
	generate_random_vector(M, &Y);

	// Create the cuSPARSE handle
	cusparseCreate(&handle);

	// Allocate device memory for vectors and the dense form of the matrix A
	checkCuda(cudaMalloc((void **)&dX, sizeof(float) * N));
	checkCuda(cudaMalloc((void **)&dY, sizeof(float) * M));
	checkCuda(cudaMalloc((void **)&dA, sizeof(float) * M * N));
	checkCuda(cudaMalloc((void **)&dNnzPerRow, sizeof(int) * M));

	// Construct a descriptor of the matrix A by default it creates a general matrix with index base zero
	cusparseCreateMatDescr(&descr);
	

	// Transfer the input vectors and dense matrix A to the device
	checkCuda(cudaMemcpy(dX, X, sizeof(float) * N, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dY, Y, sizeof(float) * M, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));

	// Compute the number of non-zero elements in A
	cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dA, M, dNnzPerRow, &totalNnz);

	if (totalNnz != trueNnz)
	{
		fprintf_s(stderr, "Difference detected between cuSPARSE NNZ and true value: expected %d but got %d\n", trueNnz, totalNnz);
		return 1;
	}

	// Allocate device memory to store the sparse CSR representation of A
	checkCuda(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalNnz));
	checkCuda(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1)));
	checkCuda(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalNnz));

	// Convert A from a dense formatting to a CSR formatting, using the GPU this takes the most time
	cusparseSdense2csr(handle, M, N, descr, dA, M, dNnzPerRow, dCsrValA, dCsrRowPtrA, dCsrColIndA);

	// Perform matrix-vector multiplication with the CSR-formatted matrix A
	cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, totalNnz, &alpha, descr, dCsrValA, dCsrRowPtrA, dCsrColIndA, dX, &beta, dY);

	// Copy the result vector back to the host
	checkCuda(cudaMemcpy(Y, dY, sizeof(float) * M, cudaMemcpyDeviceToHost));

	for (row = 0; row < 10; row++)
	{
		printf_s("%2.2f\n", Y[row]);
	}

	printf_s("...\n");

	free(A);
	free(X);
	free(Y);

	checkCuda(cudaFree(dX));
	checkCuda(cudaFree(dY));
	checkCuda(cudaFree(dA));
	checkCuda(cudaFree(dNnzPerRow));
	checkCuda(cudaFree(dCsrValA));
	checkCuda(cudaFree(dCsrRowPtrA));
	checkCuda(cudaFree(dCsrColIndA));

	cusparseDestroyMatDescr(descr);
	cusparseDestroy(handle);


	return 0;
}
