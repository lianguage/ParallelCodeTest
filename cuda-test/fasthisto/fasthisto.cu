/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible. We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated. This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it! Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/
#include "assert.h"

#include <cuda_runtime.h> //used for assert
#include <iostream>

#include <fstream>
#include <string>
#include <climits>




#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); } // copied from stack overflow, used to check gpuError codes while debugging.
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}

__global__ void sum_reduce( unsigned int* const d_position,
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
      if( tid < s && index < size){
      shared[tid] =  shared[tid] + shared[tid+s];
      }
      __syncthreads();
    }

    if(tid == 0){
        d_result[ blockIdx.x ] = shared[tid];
    }

}

__global__ void radix_predicate( unsigned int * const d_input,
                                   //unsigned int* const d_position,
                                   bool * const d_predicate,
                                   int current_bit,
                                   int size
                                   ){
   
  int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index >= size ){ return; }

  unsigned int x = d_input[index];
  x >>= current_bit;
  d_predicate[index] = x&1;
}

__global__ void predicate_relocate( unsigned int * const d_values,
                                    unsigned int * const d_newlocation,
                                    bool * predicate,
                                    int totalNumElems,
                                    int newNumElems ){


}


/*
__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo, //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
}*/


#define MAX_BLOCKSZ 512

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

/*
int find_sum_reduce( unsigned int* const d_position , int numElems ){
  int blockSize = getBlockSize_SYNC(numElems);
  int gridSize = getGridSize_SYNC(numElems);

  unsigned int * d_result; //pointer to where result of max reduce kernel can be stored
  cudaMalloc( (void**)&d_result, gridSize * sizeof(unsigned int) ); 
  unsigned int * d_incopy; //a copy of input
  cudaMalloc( (void**)&d_incopy, sizeof(unsigned int)*numElems );
  cudaMemcpy( d_incopy, d_position, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice );

  int shareFSize = blockSize * sizeof(unsigned int);
  
  int s = numElems;
  do{
    int current_gridSz = getGridSize_SYNC(s);
    sum_reduce<<< current_gridSz, blockSize, shareFSize>>>( d_incopy, d_result, s );
    cudaDeviceSynchronize();
    cudaMemcpy( d_incopy, d_result, current_gridSz*sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    s /= blockSize + 1;
  }while( s > blockSize );
  unsigned int result = 0;
  cudaMemcpy( &result, d_result , sizeof(unsigned int), cudaMemcpyDeviceToHost );
  return result;
}*/


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
  unsigned int * h_histo = (unsigned int*) malloc( numBins * sizeof(unsigned int));//histogram is probably better held on a latency optimised system.


  for( int shift = 1 ; (numBins >> shift) > 0 ; shift++ ){
    
  }

}

using namespace std;

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
    unsigned int binsize = NUMBINS * sizeof(unsigned int)
    unsigned int * h_histo = (unsigned int *)cudaMalloc(binsize);




    gpuErrchk( cudaMalloc((void**)&d_values, filesize ));
    gpuErrchk( cudaMalloc((void**)&d_histo, NUMBINS*sizeof(unsigned int) ));

    gpuErrchk( cudaMemcpy( d_values, h_values, filesize, cudaMemcpyHostToDevice ) );
    computeHistogram( d_values, d_histo, NUMBINS, numElems );

    gpuErrchk( cudaMemcpy( h_histo, d_histo, binsize, cudaMemcpyDeviceToHost ));

  }

}
