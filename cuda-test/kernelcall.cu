#include <iostream>
#include "CudaByExample/common/book.h"
#include <stdlib.h>

__global__ void kernel(void){
}

__global__ void add (int a, int b, int *c){
	*c = a + b;
}

int main ( void ) {
	/*kernel<<<1,1>>>();
	std::cout<<"Hello, World!\n";
	return 0;*/
	int c;
	int *dev_c;
	int * regular = (int*)malloc(2*sizeof(int));
	regular++;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
	add<<<1,1>>>(2, 7, dev_c);
	add<<<1,1>>>(3, 6, regular);

	HANDLE_ERROR( cudaMemcpy(
		&c,
		dev_c,
		sizeof(int),
		cudaMemcpyDeviceToHost
	));
	printf( "2 + 7 = %d\n", c);
	printf("regular address:%p\n", regular );
	printf("dev_c address: %p\n", dev_c );
	cudaFree(dev_c);

	return 0;
}