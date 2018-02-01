#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


// check for reduction
double cpuSum(double* data, int const size)
{
	double  sum = 0.0f;
	for (int i = 0; i < size; i++)  sum += data[i];

	return sum;
}

// Recursive Implementation of Interleaved Pair Approach
double recursiveReduce(double *data, int const size)
{
	// terminate check
	if (size == 1) return data[0];

	// renew the stride
	int const stride = size / 2;

	// in-place reduction
	for (int i = 0; i < stride; i++)
	{
		data[i] += data[i + stride];
	}

	// call recursively
	return recursiveReduce(data, stride);
}

int main(int argc, char **argv)
{

	int  size = 1 << 24;    // total number of elements to reduce need to use double in place of floats for a vector of this size

	printf_s("    with array size %d  \n", size);

	// allocate host memory
	size_t  bytes = size * sizeof(double);

	double*  fltarray = (double*)malloc(bytes);
	double*  tmp = (double*)malloc(bytes);

	// fill host array with random doubles
	time_t  t;
	srand(time(&t));

	for (int i = 0; i < size; i++)
	{
		fltarray[i] = rand() % 25;
	}

	memcpy(tmp, fltarray, bytes);

	
	double  cpusum = cpuSum(tmp, size);
	printf_s("cpu sum: %f\n", cpusum);

	double  cpusumR = recursiveReduce(fltarray, size);
	printf_s("cpu sum reduce: %f\n", cpusumR);

	free(fltarray);
	free(tmp);

	return EXIT_SUCCESS;

}