#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/*
* M = # of rows
* N = # of columns
*/
int M = 1024;
int N = 1024;

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

void print_partial_matrix(float *M, int nrows, int ncols, int max_row, int max_col)
{
	int row, col;

	for (row = 0; row < max_row; row++)
	{
		for (col = 0; col < max_col; col++)
		{
			printf_s("%2.2f ", M[row * ncols + col]);
		}
		printf_s("...\n");
	}
	printf_s("...\n");
}

int main(int argc, char **argv)
{
	float	*A, *d_A;
	float	*B, *d_B;
	float	*C, *d_C;
	float	*d_csrValA;
	int		*d_csrRowPtrA;
	int		*d_csrColIndA;
	int		*d_AnnzPerRow;
	int		nnzATotal;
	const float	alpha = 3.0f, beta = 5.0f;
	cusparseHandle_t	handle = 0;
	cusparseMatDescr_t	descrA = 0;


	// generate the two matrices
	srand(2468);
	int actualNnzA = generate_random_dense_matrix(M, N, &A);
	int actualNnzB = generate_random_dense_matrix(M, N, &B);

	printf_s("A:\n");
	print_partial_matrix(A, M, N, 10, 10);

	printf_s("B:\n");
	print_partial_matrix(B, M, N, 10, 10);

	// Create the cuSPARSE handle and describe matrix
	CHECK_CUSPARSE(cusparseCreate(&handle));
	CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
	CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

	// Allocate device memory for the matrices
	CHECK(cudaMalloc((void **)&d_A, M * N * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_B, M * N * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

	// Transer matrices to device 
	CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, B, M * N * sizeof(float), cudaMemcpyHostToDevice));

	// Allocate device memory for the number of non-zero elements of matrix A
	CHECK(cudaMalloc(&d_AnnzPerRow, M * sizeof(int)));

	// Compute the number of non-zero elements in A
	CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descrA, d_A, M, d_AnnzPerRow, &nnzATotal));

	if (actualNnzA != nnzATotal)
	{
		printf_s("Number of non-zero elements in A: %d don't match number returned by cuSPARSE NNZ: %d\n", actualNnzA, nnzATotal);
		EXIT_SUCCESS;
	}


	// Allocate memory for the csr values
	CHECK(cudaMalloc((void **)&d_csrValA, nnzATotal * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_csrRowPtrA, (M + 1) * sizeof(int)));
	CHECK(cudaMalloc((void **)&d_csrColIndA, nnzATotal * sizeof(int)));

	// convert dense matrix A into csr format for use in csrmm
	CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, descrA, d_A, M, d_AnnzPerRow, d_csrValA, d_csrRowPtrA, d_csrColIndA));

	// do the matrix multiplication with matrix A converted to csr format
	CHECK_CUSPARSE(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, N, nnzATotal, &alpha, descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
		d_B, M, &beta, d_C, M));
	
	// Allocate memory for C and return it
	C = (float *)malloc(M * N * sizeof(float));

	CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

	printf_s("C:\n");
	print_partial_matrix(C, M, N, 10, 10);

	free(A);
	free(B);
	free(C);

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));
	CHECK(cudaFree(d_csrValA));
	CHECK(cudaFree(d_csrRowPtrA));
	CHECK(cudaFree(d_csrColIndA));

	CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
	CHECK_CUSPARSE(cusparseDestroy(handle));


	return EXIT_SUCCESS;

}