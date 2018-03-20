#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define M_PI 3.14159265358979323846

/*
* An example usage of the cuFFT library. This example performs a 1D forward
* FFT.
*/

int nprints = 40;


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
		(*complx)[i].y = (rand()  & 0xFF)/100.0f;
	}
}

int main(int argc, char **argv)
{
	int i;
	int N = 2048;
	float *samples;
	cufftHandle plan = 0;
	cufftComplex *dComplexSamples, *complexSamples, *complexFreq;

	// Input Generation
	generate_fake_samples(N, &samples);
	real_to_complex(samples, &complexSamples, N);
	complexFreq = (cufftComplex *)malloc(sizeof(cufftComplex) * N);
	printf_s("Initial Samples:\n");

	for (i = 0; i < nprints; i++)
	{
		printf_s("  %2.4f\n", samples[i]);
	}

	printf_s("  ...\n");

	// Setup the cuFFT plan
	(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

	// Allocate device memory
	cudaMalloc((void **)&dComplexSamples, sizeof(cufftComplex) * N);

	// Transfer inputs into device memory
	cudaMemcpy(dComplexSamples, complexSamples, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

	// Execute a complex-to-complex 1D FFT
	cufftExecC2C(plan, dComplexSamples, dComplexSamples, CUFFT_FORWARD);

	// Retrieve the results into host memory
	cudaMemcpy(complexFreq, dComplexSamples, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);

	printf_s("Fourier Coefficients:\n");

	for (i = 0; i < nprints; i++)
	{
		printf_s("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq[i].x, complexFreq[i].y);
	}

	printf_s("  ...\n");

	free(samples);
	free(complexSamples);
	free(complexFreq);

	cudaFree(dComplexSamples);
	cufftDestroy(plan);

	return 0;
}