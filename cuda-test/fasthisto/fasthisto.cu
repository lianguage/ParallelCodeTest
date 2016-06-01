
#include "assert.h"

#include <cuda_runtime.h> //used for assert
#include <iostream>


#include <fstream>
#include <string>
#include <climits>

#include <vector>
#include <deque>

#define MAX_BLOCKSZ 1024


__global__ void block_Reduce_dynamic( unsigned int * d_input, unsigned int * d_atomic_sync, int numElems ){ //Always create a copy of input and padd it with zeros
  
  int index =  threadIdx.x +  blockIdx.x * blockDim.x;

  if( index <= numElems/2 ){
    d_input[index] += d_input[index + numElems/2 + 1];
    d_input[index + numElems/2 + 1] = 0;
  }

  if(threadIdx.x == 0){
    atomicAdd(d_atomic_sync, 1);
    if( d_atomic_sync[0] == gridDim.x && gridDim.x > 1 ){
      d_atomic_sync[0] = 0;
      int newGridDim = ((numElems/2+1) + MAX_BLOCKSZ - 1)/MAX_BLOCKSZ;
      block_Reduce_dynamic<<< newGridDim, MAX_BLOCKSZ  >>> ( d_input, d_atomic_sync, numElems/2+1 );
    }
  }
}

__global__ void block_Reduce( unsigned int * d_input, int totalBlocks ){ //Always create a copy of input and padd it with zeros

  int index =  threadIdx.x +  blockIdx.x * blockDim.x;
  if( blockIdx.x < totalBlocks){
    d_input[index] += d_input[index + blockDim.x * gridDim.x ];
    //d_input[index + blockDim.x * gridDim.x] = 0;
  }
  if(blockIdx.x == 0){
    if( 2*gridDim.x < totalBlocks ){
      d_input[threadIdx.x] += d_input[(totalBlocks-1) * blockDim.x + threadIdx.x ];
    }
  }

}

__global__ void thread_reduce( unsigned int * d_input ){
    
    extern __shared__ unsigned int shared[];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    shared[tid] = d_input[index];
    __syncthreads();

    for( int s = blockDim.x/2 ; s > 0 ; s >>= 1 ){
      if( tid < s ){
        shared[tid] += shared[tid+s];
      }
      __syncthreads();
    }

    if(tid == 0){
        d_input[ 0 ] = shared[0];
    }
}


__global__ void radix_predicate( 

  const unsigned int * const d_input,
        unsigned int * const d_predicate,
                         int binNumber,
                         int numElems
)
{
  int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index >= numElems ){ return; } //@note early return is ok, because __syncthreads() was not used.

  int input = (d_input[index]==binNumber);
  d_predicate[index] = input;
}


using namespace std;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); } // copied from stack overflow, used to check gpuError codes while debugging.
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{

   if (code != cudaSuccess) 
   {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }

}


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


void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo, //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free
  //cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());

  //delete[] h_vals;
  //delete[] h_histo;
  //delete[] your_histo;
  //unsigned int * h_histo = (unsigned int*) malloc( numBins * sizeof(unsigned int));//histogram is probably better held on a latency optimised system.



  int gridSize  = getGridSize ( numElems );
  int blockSize = getBlockSize( numElems );
  int filesize  = 1 + gridSize * MAX_BLOCKSZ * sizeof(unsigned int); //+1 space is used as a single global memory location for atomic operations add +1 to get actual pointer
  vector<unsigned int> h_histo( numBins, 0 );
  int total = 0;
  unsigned int * d_predicate ;
  gpuErrchk( cudaMalloc( (void**)&d_predicate , filesize ) );

  for( int binNumber = 0; binNumber < numBins ; binNumber++ ){
    gpuErrchk( cudaMemset( d_predicate, 0, filesize));
    radix_predicate<<<gridSize,blockSize>>>( d_vals, d_predicate+1, binNumber, numElems);
    
    /*if( numElems > MAX_BLOCKSZ ){
      block_Reduce_dynamic<<<gridSize/2+1, MAX_BLOCKSZ >>>( d_predicate+1, d_predicate, numElems );
    }*/
    cudaDeviceSynchronize();
    for( int blocksRemaining = gridSize/2,
             previous        = gridSize;
         blocksRemaining > 0;
         previous        = blocksRemaining,
         blocksRemaining >>= 1
        )
    {
      block_Reduce<<<blocksRemaining,MAX_BLOCKSZ>>>( d_predicate+1, previous);
      cudaDeviceSynchronize();
    }

    //cudaDeviceSynchronize();
    //unsigned int * h_debug = (unsigned int*) malloc( filesize );
    //cudaMemcpy( h_debug, d_predicate+1, filesize, cudaMemcpyDeviceToHost);
    thread_reduce <<<1,MAX_BLOCKSZ, MAX_BLOCKSZ * sizeof(unsigned int)>>>( d_predicate+1 );
    cudaMemcpy( h_histo.data()+binNumber, d_predicate+1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cout  << "bin#" << binNumber << ": " << h_histo[binNumber] << endl;
    total += h_histo[binNumber];
    //h_histo[binNumber] = arbitrary_reduce( d_predicate, numElems);
    //cout << scanned_predicate << endl;
    //cudaDeviceSynchronize();

  }
  cudaMemcpy( d_histo, h_histo.data(), numBins * sizeof(unsigned int) , cudaMemcpyHostToDevice );
  cout << "total: " << total << endl;
  cudaFree(d_predicate);
}


//test section
int main(int argc, char * argv[]){
  if(argc != 2){
    cout << "usage: " << argv[0] << "<filename>" << endl;
  }
  else{
    string line;
    ifstream myfile(argv[1]);
    int numElems = 0;
    int filesize = 0;
    int lines = 0;
    unsigned int * h_values;
    if( myfile.is_open()){
      getline(myfile,line);
      numElems = atoi(line.c_str());//first line of file is assumed to show number of elements in file.
      filesize = sizeof(unsigned int)*numElems;
      h_values = (unsigned int *)malloc(filesize);
      int i = 0;
      while(getline(myfile, line)){
         h_values[i] = atoi(line.c_str());
         lines++;
         i++;
      }
      myfile.close();
    }
    else {
      cout << "Sorry mate, can't load file" << endl;
      return 0;
    }

    unsigned int * d_values;
    unsigned int * d_histo;
    
    const int NUMBINS = 1024; //defined in main to prevent usage outside of main.
    unsigned int binsize = NUMBINS * sizeof(unsigned int);
    unsigned int * h_histo = (unsigned int *)malloc(binsize);

    gpuErrchk( cudaMalloc((void**)&d_values, filesize ));
    gpuErrchk( cudaMalloc((void**)&d_histo, NUMBINS*sizeof(unsigned int) ));

    gpuErrchk( cudaMemcpy( d_values, h_values, filesize, cudaMemcpyHostToDevice ) );

    computeHistogram( d_values, d_histo, NUMBINS, numElems );  
    gpuErrchk( cudaMemcpy( h_histo, d_histo, binsize, cudaMemcpyDeviceToHost ));

  }
}
