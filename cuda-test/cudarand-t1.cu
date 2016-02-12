#include <iostream>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <ctime>
#include <cstdio>


__global__ void d_cudaRand(double *d_out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);

    d_out[i] = curand_uniform_double(&state);
}

__global__ void f_cudaRand(float *d_out){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);

    d_out[i] = curand_normal(&state);

}

int main(int argc, char** argv)
{
    size_t N = 1 << 4;
    double *h_v = new double[N];

    double *d_out;
    cudaMalloc((void**)&d_out, N * sizeof(double));

    // generate random numbers
    d_cudaRand << < 1, N >> > (d_out);

    cudaMemcpy(h_v, d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; i++){
        printf("out: %f\n", h_v[i]);
    }

    cudaFree(d_out);
    delete[] h_v;

    return 0;
}