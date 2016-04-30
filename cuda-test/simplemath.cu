
#include <iostream>
#include "stdio.h"

//simple math is a simple test program that can run operations on arrays in parallel.
//some kernels generate numbers and others operate on them.
//the main function shows the basic lay out of how to call and utilise the kernels.
//have fun.

__global__ void operation(float * d_in, float * d_out){
	int threadid = threadIdx.x;
	float f = d_in[threadid];
	d_out[threadid] = f * f; //this is the operation
}

//Why use cpu clock cycles to do something that can be done in parallel on the GPU?
__global__ void genIntSequentialArray(float * d_in){
	//int threadid = blockDim.x*blockIdx.x+threadIdx.x;
	int threadid = threadIdx.x;
	d_in[threadid] = threadid;
}

int main(){
	int ARBITRARY = 64;
	int ARBITRARY_BYTES = ARBITRARY * sizeof(float);
	float * d_in;
	float * d_out;
	float ZEROS[64] = {};
	
	cudaMalloc((void**)&d_in, ARBITRARY_BYTES);
	cudaMalloc((void**)&d_out, ARBITRARY_BYTES);
	cudaMemcpy(d_in, ZEROS, ARBITRARY, cudaMemcpyHostToDevice );

	genIntSequentialArray<<<1,64>>>(d_in);

	float SEQ[64];
	cudaMemcpy(SEQ, d_in, ARBITRARY_BYTES, cudaMemcpyDeviceToHost);
	for( int i = 0 ; i < ARBITRARY ; i++ ){
		printf( "%f" , SEQ[i] );
		printf(((i % 4) != 3 ) ? "\t" : "\n");
	}

	operation<<<1,64>>> (d_in, d_out);
	float OPED[64] = {};
	cudaMemcpy( OPED,d_out, ARBITRARY_BYTES, cudaMemcpyDeviceToHost );
	std::cout << "\n=========================================================================" << std::endl;
	for( int i = 0 ; i < ARBITRARY ; i++ ){
		printf( "%f" , OPED[i] );
		printf(((i % 4) != 3 ) ? "\t" : "\n");
	}
	
}



/*
#include <stdio.h>

__global__ void cube(float * d_out, float * d_in){
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = idx;
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_in;
	float * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}*/