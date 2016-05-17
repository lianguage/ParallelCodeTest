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
#include "assert.h"

#include <cuda_runtime.h> //used for assert
#include <iostream>

#include <fstream>
#include <string>
#include <climits>


__global__ void max_reduce(   unsigned int* const d_position,
                              unsigned int* const d_result,
                              int size
                           ){
   	extern __shared__ unsigned int shared[];
   	int index = threadIdx.x + blockDim.x * blockIdx.x;
   	int tid = threadIdx.x;
   	if( index < size ){
		shared[tid] = d_position[index];
	}
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


__global__ void radix_predicate(   unsigned int * const d_input,
                                   //unsigned int* const d_position,
                                   unsigned int * const d_predicate,
                                   int current_bit,
                                   int size
                                   ){
   
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if(index >= size ){ return; }

	unsigned int x = d_input[index];
	x >>= current_bit;
	d_predicate[index] = x&1;
}


__global__ void blelloch1_reduction_scan( unsigned int * const d_elements,
							   			  int numElems
										){

   	extern __shared__ unsigned int shared[];
   	int index = threadIdx.x + blockDim.x * blockIdx.x;
   	int tid = threadIdx.x;
   	if( index < size ){
		shared[tid] = d_elements[index];
	}
   	__syncthreads();


  	for( int s = blockDim.x/2 ; s > 0 ; s >>= 1 ){
    	if( tid < s && index < size ){
			shared[tid] = max( shared[tid], shared[tid+s]);
    	}
    	__syncthreads();
   	}


   	//if( index == 0 ){

   	//}


}



__global__ void scan_inplace_threads(
									unsigned int * const d_elements,
									int numElems
									){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int bos = blockDim.x * blockIdx.x;

	for( int s = 1 ; s < blockDim.x ; s <<= 1 ){
		unsigned int val = 0;
		int spot = tid - s;
		if( spot >= 0 && index < numElems){
	 		val = d_elements[spot+bos];
		}
		__syncthreads();
		if( spot >= 0 && index < numElems){
			d_elements[index] += val;
		}
		__syncthreads();
	}
}

__global__ void scan_get_block_sum( 
								unsigned int * const d_scanned_elements,//expected to be an inclusive scan
								unsigned int * const d_blocksums,
								int numElems
							 ){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if( threadIdx.x + 1 == blockDim.x && index < numElems || index+1 == numElems ){ 
		d_blocksums[blockIdx.x] = d_scanned_elements[index];
	}
}


__global__ void scan_add_block_sum(
										unsigned int * const d_elements,
										unsigned int * const d_blocksums,
										int numElems
									 ){
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if( index < numElems && index >= blockDim.x ){
		unsigned int local_blocksum = d_blocksums[blockIdx.x - 1];
		d_elements[index] += local_blocksum;
	} 

}

__global__ void scan_polishing1( 
									unsigned int * const d_elems,
									unsigned int * const d_predicate,
									unsigned int * const d_middle,
									int numElems
								){
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	//inclusive to exclusive
	if( index < numElems ){
		unsigned int temp = 0;
		if( index != 0 ){ temp = d_elems[index -1];}
		d_elems[index] = temp;
	}

}


__global__ void scan_polishing2(
									unsigned int * const d_elems,
									unsigned int * const d_predicate,
									unsigned int * const d_middle,
									int numElems
								){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index < numElems){
		unsigned int identity = d_middle[0];
		d_elems[index] += identity;
	}

}

__global__ void scan_polishing3(
									unsigned int * const d_elems,
									unsigned int * const d_predicate,
									unsigned int * const d_middle,
									int numElems									
								){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index == numElems - 1 && index < numElems){ 
		d_middle[0] = d_elems[index];
		d_middle[0] += d_predicate[index];
	}
}


__global__ void radix_invert_predicate( unsigned int * const d_predicate,
                                        int size){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if( index >= size ){ return;}
	d_predicate[index] == 0 ? d_predicate[index] = 1 : d_predicate[index] = 0;
}


__global__ void radix_reposition( 
							 unsigned int* const d_position1,
						     unsigned int* const d_position2,
                             unsigned int* const d_predicate,
                             unsigned int* const d_values,
                             unsigned int* const d_value_buffer,
                             unsigned int* const d_inputPos,//check
                             unsigned int* const d_pos_buffer,
                             int size
                             	){

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index < size ){
		unsigned int displacement_index = 0;
		if(d_predicate[index] == 1){
			displacement_index = d_position2[index];
		}
		else{
			displacement_index = d_position1[index];
		}
		d_value_buffer[displacement_index] =   d_values[index];
		d_pos_buffer[displacement_index]   = d_inputPos[index];
	}
	// after this copy value_buffer to values(after synchronization)
}

__global__ void regular_relocate1(
									unsigned int* const d_values,
									unsigned int* const d_position,
									unsigned int* const d_value_buffer,
									int size
								){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index < size ){
		d_value_buffer[d_position[index]] = d_values[index];
	}
}

__global__ void regular_relocate2(
									unsigned int* const d_values,
									unsigned int* const d_position,
									unsigned int* const d_value_buffer,
									int size
								){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index < size ){
		d_values[index] = d_value_buffer[index];
		d_position[index] = index;
	}
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); } // copied from stack overflow, used to check gpuError codes while debugging.
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
   }
}

#define MAX_BLOCKSZ 1024

using namespace std;


//Functions to calculate gridSize and blockSize consistently.
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



//find max reduce works by reducing blocks then reducing the result of the blocks.
int find_max_reduce( unsigned int* const d_position , int numElems ){
	int blockSize = getBlockSize(numElems);
	int gridSize = getGridSize(numElems);
	int outFElems = gridSize * sizeof(unsigned int); //filesize of the output from one iteration of kernel

	unsigned int * d_result; //pointer to where result of max reduce kernel can be stored
	cudaMalloc( (void**)&d_result, outFElems ); 
	unsigned int * d_incopy; //a copy of input
	cudaMalloc( (void**)&d_incopy, sizeof(unsigned int)*numElems );
	cudaMemcpy( d_incopy, d_position, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice );

	int shareFSize = blockSize * sizeof(unsigned int);
	
	int s = numElems; 
	do{
		int current_gridSz = getGridSize(s);
		max_reduce<<< current_gridSz, blockSize, shareFSize>>>( d_incopy, d_result, s );
		cudaDeviceSynchronize();
		cudaMemcpy( d_incopy, d_result, current_gridSz*sizeof(unsigned int), cudaMemcpyDeviceToDevice);

		std::cout << "max_reduce called" << std::endl; //@test

		s /= blockSize;
	}while( s > blockSize );
	unsigned int result = 0;
	cudaMemcpy( &result, d_result , sizeof(unsigned int), cudaMemcpyDeviceToHost );
	return result;
}

#include <queue>
//Scan arbitrary is a function that utilises a series of kernel calls to run a Hillis and Steele scan
//It scans on block level with sync threads, then takes the block sums from each block and runs
//another scan on that in a loop until there number of elements remaining can fit into a single block.
void scan_arbitrary( 
						unsigned int * const d_elements,
						unsigned int * const d_predicate,
						unsigned int * const d_middle,
						int numElems ){

	cout << "====arbitrary scan====" << endl;
	int gridSize = getGridSize(numElems);
	int blockSize = getBlockSize(numElems);

	int cycles = 0;
	for( int tempGridSize = gridSize, 
			 tempNumElems = numElems 
		    ;tempNumElems > 1
		    ;tempNumElems = tempGridSize, 
		 	 tempGridSize = getGridSize(tempNumElems),
			 cycles++
	);//After the for loop cycles ends up being the correct size
	//running this loop for i < cycles will loop the corresponding number of cycles in this loop.

	vector<unsigned int*> D_scan_targets(cycles+1);
	vector<int> blocksum_numelems(cycles+1);
	scan_inplace_threads<<<gridSize,blockSize>>>( d_elements, numElems );
	cudaDeviceSynchronize();
	D_scan_targets[0] = d_elements;
	blocksum_numelems[0] = numElems; 

	for( int i = 0, 
		     tempGridSize = gridSize,
		     tempNumElems = numElems
		    ;i < cycles
			;i++
	){
		unsigned int * d_blockscan;
		gpuErrchk( cudaMalloc( (void**)&d_blockscan , gridSize*sizeof(unsigned int) ) );
		
		scan_get_block_sum<<<tempGridSize,blockSize>>>(D_scan_targets[i], d_blockscan, tempNumElems);
			cudaDeviceSynchronize();

		//resize dimensions to calculate blocksums
		tempNumElems = tempGridSize ;
		tempGridSize = getGridSize(tempNumElems) ;
		if( tempNumElems > 1 ){

			scan_inplace_threads<<<tempGridSize,blockSize>>>( d_blockscan, tempNumElems );
				cudaDeviceSynchronize();

			D_scan_targets[i+1] = d_blockscan;
			blocksum_numelems[i+1] = tempNumElems;

		}

	}

	for(int i =  cycles - 1 ; i > 0 ; i-- ){
		int tempGridSize = getGridSize( blocksum_numelems[i-1] );
		
		scan_add_block_sum<<<tempGridSize,blockSize>>>( D_scan_targets[i-1], D_scan_targets[i], blocksum_numelems[i-1] );
		cudaDeviceSynchronize();
		//Consider freeing memory here
	}


	for( int i = 1 ; i < cycles ; i++ ){
		cudaFree(D_scan_targets[i]);// free all the elements except the first since it is used again after function end.
	}

	scan_polishing1<<<gridSize,blockSize>>>( d_elements, d_predicate, d_middle, numElems);
			cudaDeviceSynchronize();
	scan_polishing2<<<gridSize,blockSize>>>( d_elements, d_predicate, d_middle, numElems);
			cudaDeviceSynchronize();
	scan_polishing3<<<gridSize,blockSize>>>( d_elements, d_predicate, d_middle, numElems);
			cudaDeviceSynchronize();
	
}


//Note the use of global memory in the reposition phases.
//I wasn't sure how else to ensure that relocations happened in sync so I ran them in separate kernels so that
// each step was guaranteed to be completed before moving onto the next ie-  block level synchronisation was needed.
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_numbers,
               unsigned int* const d_position1,
               const size_t numElems)
{

	const int filesize = numElems*sizeof(unsigned int);

	unsigned int* d_predicate;
	unsigned int* d_position2;
	unsigned int* d_numbers_buffer;
	unsigned int* d_inputPos_buffer;

	int blockSize =  getBlockSize(numElems);
	int gridSize = getGridSize(numElems);


	gpuErrchk(cudaMalloc((void**)&d_predicate, filesize ));
	gpuErrchk(cudaMalloc((void**)&d_position2, filesize ));
	gpuErrchk(cudaMalloc((void**)&d_numbers_buffer, filesize ));
	gpuErrchk(cudaMalloc((void**)&d_inputPos_buffer, filesize ));

	//move all elements such that the position vector is in sorted order.
	regular_relocate1<<<gridSize,blockSize>>>(d_inputVals, d_inputPos, d_numbers_buffer, numElems);
		cudaDeviceSynchronize();
	regular_relocate2<<<gridSize,blockSize>>>(d_inputVals, d_inputPos, d_numbers_buffer, numElems);
		cudaDeviceSynchronize();

	//set scan destination to 0;
	gpuErrchk(cudaMemset( d_predicate, 0, filesize));
	//gpuErrchk(cudaMemset( d_numbers_buffer, 0 , filesize ));
	gpuErrchk(cudaMemcpy ( d_position1, d_inputPos, filesize, cudaMemcpyDeviceToDevice ));
	gpuErrchk(cudaMemcpy ( d_position2, d_inputPos, filesize, cudaMemcpyDeviceToDevice ));
	gpuErrchk(cudaMemcpy ( d_numbers, d_inputVals, filesize, cudaMemcpyDeviceToDevice ));//Values changed after calculating most significant element.


	unsigned int * d_middle;
	gpuErrchk(cudaMalloc((void**)&d_middle, sizeof(unsigned int) ));
	unsigned int bitsig = find_max_reduce( d_numbers, numElems);
	cout<< "bitsig:" << bitsig << endl;
	if( bitsig <= UINT_MAX/2 ){ bitsig <<= 1; }
	else{ bitsig = UINT_MAX; }


	for ( int current_bit = 0 ; (bitsig >> current_bit) > 1 ; current_bit++  ){

		cout << "currentbit:" << current_bit << endl;
		gpuErrchk(cudaMemset(d_middle,0, sizeof(unsigned int)));
			cudaDeviceSynchronize();

		radix_predicate<<<gridSize,blockSize>>>(d_numbers, d_predicate, current_bit, numElems);
			cudaDeviceSynchronize();

		radix_invert_predicate<<<gridSize,blockSize>>>(d_predicate, numElems);
			cudaDeviceSynchronize();

		cudaMemcpy(d_position1, d_predicate, filesize, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();

		scan_arbitrary(d_position1 ,d_predicate, d_middle, numElems);
			cudaDeviceSynchronize();

		radix_invert_predicate<<<gridSize,blockSize>>>(d_predicate, numElems);
			cudaDeviceSynchronize();

		cudaMemcpy(d_position2, d_predicate, filesize, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();

		scan_arbitrary(d_position2 , d_predicate, d_middle, numElems);
			cudaDeviceSynchronize();

		radix_reposition<<<gridSize,blockSize>>>(d_position1, d_position2, d_predicate, d_numbers, d_numbers_buffer, d_inputPos, d_inputPos_buffer, numElems );
			cudaDeviceSynchronize();
		cudaMemcpy(d_numbers, d_numbers_buffer, filesize, cudaMemcpyDeviceToDevice );
		cudaMemcpy(d_inputPos, d_inputPos_buffer, filesize, cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
	}

	gpuErrchk( cudaMemcpy( d_inputVals, d_numbers, filesize, cudaMemcpyDeviceToDevice ));
	gpuErrchk( cudaMemcpy( d_position1, d_inputPos, filesize, cudaMemcpyDeviceToDevice ));

	cudaFree(d_predicate);
	cudaFree(d_position2);
	cudaFree(d_middle);
}


int main(int argc, char * argv[]){
	if(argc !=  2){
		cout <<   "usage: " << argv[0] << "<filename>" << endl;
	}
	else{
		string line;
		ifstream myfile(argv[1]);
		int size = 0;
		int filesize = 0;
		int lines = 0;
		unsigned int * h_numbers;
		unsigned int * h_position;
		if( myfile.is_open()){
			getline(myfile,line);
			size = atoi(line.c_str());//first line of file is assumed to show number of elements in file.
			filesize = sizeof(int)*size;
			h_numbers = (unsigned int *)malloc(filesize);
			h_position = (unsigned int *)malloc(filesize);
			int i = 0;
			while(getline(myfile, line)){
				 h_numbers[i] = atoi(line.c_str());
				 h_position[i] = i;
				 lines++;
				 i++;
			}
			myfile.close();
		}
		else {
			cout << "Sorry mate, can't load file" << endl;
			return 0;
		}
			//allocate memory for device
		unsigned int * d_numbers;
		unsigned int * d_position;
		unsigned int * d_out_numbers;
		unsigned int * d_out_position;

		gpuErrchk( cudaMalloc((void**)&d_numbers, filesize));
		cudaDeviceSynchronize();
		gpuErrchk( cudaMalloc((void**)&d_position, filesize));
		cudaDeviceSynchronize();
		gpuErrchk( cudaMalloc((void**)&d_out_numbers, filesize));
		cudaDeviceSynchronize();
		gpuErrchk( cudaMalloc((void**)&d_out_position, filesize));
		cudaDeviceSynchronize();

		//copy memory host to device
		gpuErrchk( cudaMemcpy( d_numbers, h_numbers, filesize , cudaMemcpyHostToDevice ));
		cudaDeviceSynchronize();
		gpuErrchk( cudaMemcpy( d_position, h_position, filesize, cudaMemcpyHostToDevice ));
		cudaDeviceSynchronize();

		//run sort
		your_sort(d_numbers, d_position, d_out_numbers, d_out_position, lines );
		//cudaDeviceSynchronize();

		gpuErrchk( cudaMemcpy( h_numbers, d_out_numbers ,filesize, cudaMemcpyDeviceToHost));
		gpuErrchk( cudaMemcpy (h_position, d_position, filesize, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		ofstream sortedfile("sorted");
	    if( sortedfile.is_open()){
	      for(int i = 0 ; i < lines ; i++){
	        sortedfile << std::to_string(h_numbers[i]) << "         " <<   to_string(h_position[i])   <<"\n";
	      }
	    }
		std::cout << "finished" <<std::endl;

		return 0;

	}

}
