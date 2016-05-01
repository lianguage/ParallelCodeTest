//Udacity HW 4
//Radix Sorting

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

      1) Histogram of the number of occurrences o feach digit
      2) Exclusive Prefix Sum of Histogram
      3) Determine relative offset of each digit
           For example [0 0 1 1 0 0 1]
                   ->  [0 1 0 1 2 3 2]
      4) Combine the results of steps 2 &3  to determine the final
         output location for each element and move it there

      LSB Radix sort is an out-of-place sort and you will need to ping-pong values
      between the input and output buffers we have provided.  Make sure the final
      sorted results end up in the output buffer!  Hint: You may need to do a copy
      at the end.

    */

//are these needed?
#include "reference_calc.cpp"
#include "utils.h"

//@max_reduce tested.
__global__ void max_reduce(   unsigned int* const d_input,
                              unsigned int* const d_result,
                              int size
                           ){
   extern __shared__ unsigned int shared[];
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if( index >= size ){ return; }
   int tid = threadIdx.x;
   shared[tid] = d_input[index];
   __syncthreads();

   for( int s = blockDim.x/2 ; s > 0 ; s >>= 1 ){
      if( tid < s){
         shared[tid] = max( shared[tid], shared[tid+s]);
      }
      __syncthreads();
   }

   if(tid == 0){
      d_result[ blockIdx.x ] = shared[tid];
   }
}


//@radix_predicate tested
__global__ void radix_predicate(   unsigned int* const d_input,
                                   //unsigned int* const d_position,
                                   bool* const d_predicate,
                                   int current_bit,
                                   int size
                                   ){
   
   int index = threadIdx.x + blockDim.x*blockIdx.x;
   if(index >= size ){ return; }

   //int position = d_position[index];
   //predicate: (i & 1) == 0;
   unsigned int x = d_input[index];
   x >>= current_bit;
   d_predicate[index] = x&1;
}


__global__ void radix_scan( bool* const d_predicate,
                            unsigned int* const d_input,
                            unsigned int identity,
                            unsigned int * d_middle,
                            int size ){ //identity variable is used to set the first term. This is used so that the indexes for 
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int bos = blockIdx.x * blockDim.x;
	if(index<size) {d_input[index] = d_predicate[index]; }
	else{return;}

	//sum threads
	for( int s = 1 ; s < blockDim.x ; s <<= 1 ){
		unsigned int val = 0;
		int spot = tid - s;
		if( spot >= 0){
	 		val = d_input[spot+bos];
		}
		__syncthreads();
		if( spot >= 0){
			d_input[index] += val;
		}
		__syncthreads();
	}

	//sum blocks
	for ( int s = 1 ; s < gridDim.x ; s <<= 1){
		unsigned int val = 0;
		int spot = bos - (s * blockDim.x) - 1;
		if( spot >= blockDim.x - 1 ){
			val = d_input[spot];
		}
		if ( spot > blockDim.x -1 ){
			d_input[index] += val;
		}
		__syncthreads();
	}

	//shift for exclusive scan
	unsigned int temp = 0;
	if( tid != 0 ){ temp = d_input[index-1]; }
	__syncthreads();
	d_input[index] = temp;
	__syncthreads();

	d_input[index] += identity;
	if( index == size - 1){ *d_middle = d_input[index] + d_predicate[size-1]; }

}

__global__ void radix_invert_predicate( bool* const d_predicate,
                                        int size){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if( index >= size ){ return;}
	d_predicate[index] = !d_predicate[index];
}


__global__ void radix_reposition( unsigned int* const d_position1,
						     unsigned int* const d_position2,
                             bool* const d_predicate,
                             unsigned int* const d_values,
                             int size){

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= size ){ return; }
	unsigned int displacement_index= 0;
	if(d_predicate[index]){
		displacement_index = d_position2[index];
	}
	else{
		displacement_index = d_position1[index];
	}
	unsigned int value = d_values[index];
	__syncthreads();
	d_values[displacement_index] = value;

}



#define MAX_BLOCKSZ 512

using namespace std;

int getBlockSize(int numElems){
	if(numElems > MAX_BLOCKSZ){
		return MAX_BLOCKSZ;
	}
	else{
		return numElems;
	}
}

int getGridSize(int numElems){
	return (numElems + MAX_BLOCKSZ -1)/MAX_BLOCKSZ;
}



//@find_max_reduce tested
int find_max_reduce( unsigned int* const d_input , int numElems ){
	int blockSize = getBlockSize(numElems);
	int gridSize = getGridSize(numElems);
	int outFElems = gridSize * sizeof(unsigned int); //filesize of the output from one iteration of kernel

	unsigned int * d_result; //pointer to where result of max reduce kernel can be stored
	checkCudaErrors(cudaMalloc( (void**)&d_result, outFElems )); 
	unsigned int * d_incopy; //a copy of input
	cudaMalloc( (void**)&d_incopy, sizeof(unsigned int)*numElems );
	checkCudaErrors(cudaMemcpy( d_incopy, d_input, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice ));

	int shareFSize = blockSize * sizeof(unsigned int);
	
	int s = numElems; 
	do{
		int current_gridSz = getGridSize(s);
		max_reduce<<< current_gridSz, blockSize, shareFSize>>>( d_incopy, d_result, s );
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy( d_incopy, d_result, current_gridSz*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		s /= blockSize;
	}while( s > blockSize );
	unsigned int result = 0;
	checkCudaErrors(cudaMemcpy( &result, d_result , sizeof(unsigned int), cudaMemcpyDeviceToHost ));
	return result;
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_numbers,
               unsigned int* const d_position1,
               const size_t numElems)
{

	int filesize = numElems*sizeof(unsigned int);
	//unsigned int* d_inputCopy;
	//unsigned int* d_inPosCopy;
	bool* d_predicate;
	unsigned int* d_position2;

	checkCudaErrors(cudaMalloc((void**)&d_predicate, numElems*sizeof(bool) ));
	checkCudaErrors(cudaMalloc((void**)&d_position2, filesize ));

	//set scan destination to 0;
	checkCudaErrors(cudaMemset( d_predicate, numElems*sizeof(bool), 0 ));
	checkCudaErrors(cudaMemcpy ( d_position1, d_inputPos, filesize,cudaMemcpyDeviceToDevice ));
	checkCudaErrors(cudaMemcpy ( d_position2, d_inputPos, filesize, cudaMemcpyDeviceToDevice ));
	checkCudaErrors(cudaMemcpy ( d_numbers, d_inputVals, filesize,cudaMemcpyDeviceToDevice ));//Values changed after calculating most significant element.

	int blockSize =  getBlockSize(numElems);
	int gridSize = getGridSize(numElems);
	unsigned int * d_middle;
	checkCudaErrors(cudaMalloc((void**)&d_middle, sizeof(int) ));

	unsigned int maxvalue = find_max_reduce( d_numbers, numElems);//find the largest element, so the number of bits to run radix is known.
	maxvalue <<= 1; //multiply largest element by 2, so that all elements are less than the most significant bit only.
	cout<< "max value:" << maxvalue << endl;
	cout<< "num elems:" << numElems << endl;
	for ( int current_bit = 1 ; (maxvalue >> current_bit) > 0 ; current_bit++  ){
		unsigned int h_middle = 0;
		radix_predicate<<<gridSize,blockSize>>>(d_numbers, d_predicate, current_bit, numElems);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		radix_invert_predicate<<<gridSize,blockSize>>>(d_predicate, numElems);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		radix_scan<<<gridSize,blockSize>>>(d_predicate, d_position1, h_middle, d_middle, numElems  );
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpy( &h_middle, d_middle, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		radix_invert_predicate<<<gridSize,blockSize>>>(d_predicate, numElems);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		radix_scan<<<gridSize,blockSize>>>(d_predicate, d_position2, h_middle, d_middle, numElems );
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		radix_reposition<<<gridSize,blockSize>>>(d_position1, d_position2, d_predicate, d_numbers, numElems );
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	 checkCudaErrors(cudaMemcpy( d_position1, d_inputPos, filesize, cudaMemcpyDeviceToDevice  ));
}

