#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "common.h"

// function for checking the CUDA runtime API results.


void dense2coo(float *A, int M, int N, float **values, int **row_indices, int **col_indices)
{
	cusparseHandle_t	A_handle = 0;
	cusparseMatDescr_t	A_descr  = 0;


	float	 *d_A, *d_csrValA;
	int		 *d_nnzPerRow, *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;
	int		 nnzTotal;
	

	CHECK_CUSPARSE(cusparseCreate(&A_handle));

	// construct a description of matrix A
	CHECK_CUSPARSE(cusparseCreateMatDescr(&A_descr));
	CHECK_CUSPARSE(cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO));

	CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
	CHECK(cudaMalloc(&d_nnzPerRow, M * sizeof(int)));

	CHECK_CUSPARSE(cusparseSnnz(A_handle, CUSPARSE_DIRECTION_ROW, M, N, A_descr, d_A, M, d_nnzPerRow, &nnzTotal));

	CHECK(cudaMalloc(&d_csrRowPtrA,  (M + 1) * sizeof(int)));
	CHECK(cudaMalloc(&d_csrValA,  nnzTotal   * sizeof(float)));
	CHECK(cudaMalloc(&d_csrColIndA, nnzTotal * sizeof(float)));
	CHECK(cudaMalloc(&d_cooRowIndA, nnzTotal * sizeof(float)));

	CHECK_CUSPARSE(cusparseSdense2csr(A_handle, M, N, A_descr, d_A, M, d_nnzPerRow, d_csrValA, d_csrRowPtrA, d_csrColIndA));

	CHECK_CUSPARSE(cusparseXcsr2coo(A_handle, d_csrRowPtrA, nnzTotal, M, d_cooRowIndA, CUSPARSE_INDEX_BASE_ZERO));

	float *values = (float *)malloc(nnzTotal * sizeof(float));
	int   *row_i  = (int *)malloc(nnzTotal * sizeof(int));
	int   *col_i  = (int *)malloc(nnzTotal * sizeof(int));

	CHECK(cudaMemcpy(values, d_csrValA, nnzTotal * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(row_i, d_cooRowIndA, nnzTotal * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(col_i, d_csrColIndA, nnzTotal * sizeof(int), cudaMemcpyDeviceToHost));

}