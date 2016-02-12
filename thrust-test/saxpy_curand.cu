#include <iostream>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <ctime>
#include <cstdio>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

struct saxpy_functor
{
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  { 
    return a * x + y;
  }
  
};

__global__ void f_cudaRand(float *d_out){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);

    d_out[i] = curand_normal(&state);

}



