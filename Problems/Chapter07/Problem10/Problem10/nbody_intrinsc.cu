#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/**
* This example implements a very simple two-stage NBody simulation. The goal of
* this sample code is to illustrate the use of all three main concepts from
* Chapter 7 in a single application.
*
* This NBody simulation consists of two main stages: updating particle
* velocities based on calculated acceleration, followed by updating particle
* positions based on the computed velocities.
*
* This example also supports the use of compile-time flags -DSINGLE_PREC and
* -DDOUBLE_PREC to switch between the floating-point types used to store
* particle acceleration, velocity, and position.
*
* Another supported compile-time flag is -DVALIDATE, which turns on executing a
* copy of the same computation on the host side with the same floating-point
* type. Using the host values as a baseline, this application can validate its
* own numerical results. The measure used for validation is the mean distance
* between a particle's position as calculated on the device versus the position
* from the host.
**/

/*
* If neither single- or double-precision is specified, default
* to single-precision.
*/
#ifndef SINGLE_PREC
#ifndef DOUBLE_PREC
#define SINGLE_PREC
#endif
#endif

#ifdef SINGLE_PREC

/* using the intrinsic functions xxx_rn is slower because they  adhere to the IEEE-754 standard when rounding which requires more work
 * SQRT(x)     __sqrtf(x)  faster than __fsqrt_rn but not as accurate --use_fast_math or -prec sqrt = {true|false} determines accuracy vs speed
*/

typedef float real;
#define MAX_DIST    200.0f
#define MAX_SPEED   100.0f
#define MASS        2.0f
#define DT          0.00001f
#define LIMIT_DIST  0.000001f
#define POW(x,y)    __powf(x,y)
#define DIST(x,y)	hypotf(x,y)
#define MUL(x,y)	__fmul_rn(x,y)
#define MADD(x,y,z)	__fmaf_rn(x,y,z)
#define SQRT(x)		__fsqrt_rn(x)
#define RCP(x)		__frcp_rn(x)

#else // SINGLE_PREC

typedef double real;
#define MAX_DIST    200.0
#define MAX_SPEED   100.0
#define MASS        2.0
#define DT          0.00001
#define LIMIT_DIST  0.000001
#define POW(x,y)    __powf(x,y)
#define SQRT(x)     __dsqrt_rn(x)
#define MUL(x,y)	__dmul_rn(x,y)
#define MADD(x,y,z)	__fma_rn(x,y,z)
#define RCP(x)		__drcp_rn(x)
#define DIST(x,y)	hypotf(x,y)


#endif // SINGLE_PREC

#ifdef VALIDATE

/**
* Host implementation of the NBody simulation.
**/
static void h_nbody_update_velocity(real *px, real *py, real *vx,
									real *vy, real *ax, real *ay, int N, int *exceeded_speed, int id)
{
	real total_ax = 0.0f;
	real total_ay = 0.0f;

	real my_x = px[id];
	real my_y = py[id];

	int i = (id + 1) % N;

	while (i != id)
	{
		real other_x = px[i];
		real other_y = py[i];

		real rx = other_x - my_x;
		real ry = other_y - my_y;

		real dist2 = rx * rx + ry * ry;

		if (dist2 < LIMIT_DIST)
		{
			dist2 = LIMIT_DIST;
		}

		real dist6 = dist2 * dist2 * dist2;
		real s = MASS * (1.0f / SQRT(dist6));
		total_ax += rx * s;
		total_ay += ry * s;

		i = (i + 1) % N;
	}

	ax[id] = total_ax;
	ay[id] = total_ay;

	vx[id] = vx[id] + ax[id];
	vy[id] = vy[id] + ay[id];

	real v = SQRT(POW(vx[id], 2.0) + POW(vy[id], 2.0));

	if (v > MAX_SPEED)
	{
		*exceeded_speed = *exceeded_speed + 1;
	}
}

static void h_nbody_update_position(real *px, real *py, real *vx, real *vy, int N, int *beyond_bounds, int id)
{

	px[id] += (vx[id] * DT);
	py[id] += (vy[id] * DT);

	real dist = SQRT(POW(px[id], 2.0) + POW(py[id], 2.0));

	if (dist > MAX_DIST)
	{
		*beyond_bounds = 1;
	}
}
#endif // VALIDATE

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


/**
* CUDA implementation of simple NBody.
**/
__global__ void d_nbody_update_velocity(real *px, real *py,
										real *vx, real *vy,
										real *ax, real *ay, int N, int *exceeded_speed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	real total_ax = 0.0f;
	real total_ay = 0.0f;

	if (tid >= N) return;

	real my_x = px[tid];
	real my_y = py[tid];

	int i = (tid + 1) % N;

	while (i != tid)
	{
		real other_x = px[i];
		real other_y = py[i];

		real rx = other_x - my_x;
		real ry = other_y - my_y;

		real dist2 = MADD(rx, rx, MUL(ry,ry));

		if (dist2 < LIMIT_DIST)
		{
			dist2 = LIMIT_DIST;
		}

		real dist6 = MUL(MUL(dist2, dist2), dist2);
		real s = MUL(MASS,  RCP(SQRT(dist6)));
		total_ax = MADD(rx, s, total_ax);
		total_ay = MADD(ry, s, total_ay);

		i = (i + 1) % N;
	}

	ax[tid] = total_ax;
	ay[tid] = total_ay;

	vx[tid] = vx[tid] + ax[tid];
	vy[tid] = vy[tid] + ay[tid];

	real v = DIST(vx[tid], vy[tid]);

	//real v = SQRT(POW(vx[tid], 2.0) + POW(vy[tid], 2.0));

	if (v > MAX_SPEED)
	{
		atomicAdd(exceeded_speed, 1);
	}
}

__global__ void d_nbody_update_position(real *px, real *py, real *vx, real *vy, int N, int *beyond_bounds)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= N) return;

	px[tid] += (vx[tid] * DT);
	py[tid] += (vy[tid] * DT);

	real dist = DIST(px[tid], py[tid]);
	//real dist = SQRT(POW(px[tid], 2.0) + POW(py[tid], 2.0));

	if (dist > MAX_DIST)
	{
		*beyond_bounds = 1;
	}
}

static void print_points(real *x, real *y, int N)
{
	int i;

	for (i = 0; i < N; i++)
	{
		printf_s("%.20e %.20e\n", x[i], y[i]);
	}
}

int main(int argc, char **argv)
{
	int i;
	int N = 30720;
	int block = 512;
	int iter, niters = 50;
	real *d_px, *d_py;
	real *d_vx, *d_vy;
	real *d_ax, *d_ay;
	real *h_px, *h_py;
	int *d_exceeded_speed, *d_beyond_bounds;
	int exceeded_speed, beyond_bounds;
#ifdef VALIDATE
	int id;
	real *host_px, *host_py;
	real *host_vx, *host_vy;
	real *host_ax, *host_ay;
	int host_exceeded_speed, host_beyond_bounds;
#endif // VALIDATE

#ifdef SINGLE_PREC
	printf_s("Using single-precision floating-point values\n");
#else // SINGLE_PREC
	printf_s("Using double-precision floating-point values\n");
#endif // SINGLE_PREC

#ifdef VALIDATE
	printf_s("Running host simulation. WARNING, this might take a while.\n");
#endif // VALIDATE

	h_px = (real *)malloc(N * sizeof(real));
	h_py = (real *)malloc(N * sizeof(real));

#ifdef VALIDATE
	host_px = (real *)malloc(N * sizeof(real));
	host_py = (real *)malloc(N * sizeof(real));
	host_vx = (real *)malloc(N * sizeof(real));
	host_vy = (real *)malloc(N * sizeof(real));
	host_ax = (real *)malloc(N * sizeof(real));
	host_ay = (real *)malloc(N * sizeof(real));
#endif // VALIDATE

	for (i = 0; i < N; i++)
	{
		real x = (rand() % 200) - 100;
		real y = (rand() % 200) - 100;

		h_px[i] = x;
		h_py[i] = y;
#ifdef VALIDATE
		host_px[i] = x;
		host_py[i] = y;
#endif // VALIDATE
	}

	checkCuda(cudaMalloc((void **)&d_px, N * sizeof(real)));
	checkCuda(cudaMalloc((void **)&d_py, N * sizeof(real)));

	checkCuda(cudaMalloc((void **)&d_vx, N * sizeof(real)));
	checkCuda(cudaMalloc((void **)&d_vy, N * sizeof(real)));

	checkCuda(cudaMalloc((void **)&d_ax, N * sizeof(real)));
	checkCuda(cudaMalloc((void **)&d_ay, N * sizeof(real)));

	checkCuda(cudaMalloc((void **)&d_exceeded_speed, sizeof(int)));
	checkCuda(cudaMalloc((void **)&d_beyond_bounds, sizeof(int)));

	checkCuda(cudaMemcpy(d_px, h_px, N * sizeof(real), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_py, h_py, N * sizeof(real), cudaMemcpyHostToDevice));

	checkCuda(cudaMemset(d_vx, 0x00, N * sizeof(real)));
	checkCuda(cudaMemset(d_vy, 0x00, N * sizeof(real)));
#ifdef VALIDATE
	memset(host_vx, 0x00, N * sizeof(real));
	memset(host_vy, 0x00, N * sizeof(real));
#endif // VALIDATE

	checkCuda(cudaMemset(d_ax, 0x00, N * sizeof(real)));
	checkCuda(cudaMemset(d_ay, 0x00, N * sizeof(real)));
#ifdef VALIDATE
	memset(host_ax, 0x00, N * sizeof(real));
	memset(host_ay, 0x00, N * sizeof(real));
#endif // VALIDATE


	for (iter = 0; iter < niters; iter++)
	{
		checkCuda(cudaMemset(d_exceeded_speed, 0x00, sizeof(int)));
		checkCuda(cudaMemset(d_beyond_bounds, 0x00, sizeof(int)));

		d_nbody_update_velocity <<<N / block, block >>>(d_px, d_py, d_vx, d_vy, d_ax, d_ay, N, d_exceeded_speed);
		d_nbody_update_position <<<N / block, block >>>(d_px, d_py, d_vx, d_vy, N, d_beyond_bounds);

	}

	checkCuda(cudaDeviceSynchronize());

#ifdef VALIDATE

	for (iter = 0; iter < niters; iter++)
	{
		printf_s("iter=%d\n", iter);
		host_exceeded_speed = 0;
		host_beyond_bounds = 0;

#pragma omp parallel for
		for (id = 0; id < N; id++)
		{
			h_nbody_update_velocity(host_px, host_py, host_vx, host_vy,
				host_ax, host_ay, N, &host_exceeded_speed, id);
		}

#pragma omp parallel for
		for (id = 0; id < N; id++)
		{
			h_nbody_update_position(host_px, host_py, host_vx, host_vy, N, &host_beyond_bounds, id);
		}
	}

#endif // VALIDATE

	checkCuda(cudaMemcpy(&exceeded_speed, d_exceeded_speed, sizeof(int), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(&beyond_bounds, d_beyond_bounds, sizeof(int), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_px, d_px, N * sizeof(real), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_py, d_py, N * sizeof(real), cudaMemcpyDeviceToHost));

	print_points(h_px, h_py, 10);
	printf_s("Any points beyond bounds? %s, # points exceeded velocity %d/%d\n", beyond_bounds > 0 ? "true" : "false", exceeded_speed, N);

#ifdef VALIDATE
	double error = 0.0;

	for (i = 0; i < N; i++)
	{
		double dist = sqrt(pow(h_px[i] - host_px[i], 2.0) + pow(h_py[i] - host_py[i], 2.0));
		error += dist;
	}

	error /= N;
	printf_s("Error = %.20e\n", error);
#endif // VALIDATE

	return 0;
}