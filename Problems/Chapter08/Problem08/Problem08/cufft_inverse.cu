#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>

float M_PI = 3.141592653589f;

int nprints = 30;

/*
* Create N fake samplings along the function cos(x). These samplings will be
* stored as single-precision floating-point values.
*/
void generate_fake_samples(int N, float **out)
{
	int i;
	float *result = (float *)malloc(sizeof(float) * N);
	double delta = M_PI / 20.0;

	for (i = 0; i < N; i++)
	{
		result[i] = cos(i * delta);
	}

	*out = result;
}

/*
* Convert a real-valued vector r of length Nto a complex-valued vector.
*/
void real_to_complex(float *r, cufftComplex **complx, int N)
{
	int i;
	(*complx) = (cufftComplex *)malloc(sizeof(cufftComplex) * N);
		
	for (i = 0; i < N; i++)
	{
		(*complx)[i].x = r[i];
		(*complx)[i].y = 0;
	}
}

int main(int argc, char **argv)
{
	int		i;
	int		N = 2048;
	float	*samples;

	cufftHandle		plan = 0;
	cufftComplex	*d_ComplexSamples, *complexSamples, *complexFreq;


	// generate are data for the FFT
	generate_fake_samples(N, &samples);
	real_to_complex(samples, &complexSamples, N);

	complexFreq = (cufftComplex *)malloc(N * sizeof(cufftComplex));
	if (!complexFreq)
	{
		printf_s("host memory allocation failed");
		return EXIT_FAILURE;
	}

	printf_s("Initial Samples:\n");

	for (i = 0; i < nprints; i++)
	{
		printf_s("  %2.4f\n", samples[i]);
	}

	printf_s("  ...\n");

	// setup the cuFFT plan
	CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

	// allocate device memory and transfer to device
	CHECK(cudaMalloc((void **)&d_ComplexSamples, N * sizeof(cufftComplex)));
	CHECK(cudaMemcpy(d_ComplexSamples, complexSamples, N * sizeof(cufftComplex), cudaMemcpyHostToDevice));

	// execute are FFT forward then immediately reverse it
	CHECK_CUFFT(cufftExecC2C(plan, d_ComplexSamples, d_ComplexSamples , CUFFT_FORWARD));

	CHECK_CUFFT(cufftExecC2C(plan, d_ComplexSamples, d_ComplexSamples, CUFFT_INVERSE));

	// retrieve the results from the gpu
	CHECK(cudaMemcpy(complexFreq, d_ComplexSamples, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	printf_s("Fourier Coefficients:\n");

	float maximum = 0.0f;
	for (i = 0; i < N; i++)
	{
		if (fabs(complexFreq[i].x) > maximum)
		{
			maximum = fabs(complexFreq[i].x);
		}
	}


	for (i = 0; i < nprints; i++)
	{
		printf_s("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq[i].x / maximum, complexFreq[i].y / maximum);
	}

	printf_s("  ...\n");

	free(complexSamples);
	free(complexFreq);
	free(samples);

	CHECK(cudaFree(d_ComplexSamples));
	CHECK_CUFFT(cufftDestroy(plan));

	return EXIT_SUCCESS;

}