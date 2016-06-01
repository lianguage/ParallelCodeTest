
#include "assert.h"

#include <cuda_runtime.h> //used for assert
#include <iostream>


#include <fstream>
#include <string>
#include <climits>

#include <vector>
#include <deque>


//backup version stored as kt_backup.cu
#define MAX_BLOCKSZ 512


__global__ void bookmark_blocksum(  

  unsigned int* d_elements,  
            int blockSumNumElems, 
            int nextmark
)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if( 2*index+1 < blockSumNumElems){
    //d_elements[blocks_mark + index] = 0;
    d_elements[2*index] +=  d_elements[nextmark + blockIdx.x];
    d_elements[2*index + 1] +=  d_elements[nextmark + blockIdx.x];
  }
  else{
    
    //@todo throw some kind of error. consider using assert or something here instead.

  }
}


__global__ void blelloch_threadsum(

  unsigned int* d_elements, 
            int numElems, 
            int nextmark){
    //@note @unexplained behavior, see blelloch-weirdbug.cu
    extern __shared__ unsigned int shared[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    if( 2*index+1 < numElems  ){
      shared[2*tid] = d_elements[2*index];
      shared[2*tid+1] = d_elements[2*index + 1 ];
    }
    __syncthreads();

    int s = 1;
    for( int d = blockDim.x ; d > 0  ; d >>= 1 ){
      __syncthreads();
      if( tid < d){ 
        unsigned int addn = shared[s*(2*tid + 1) - 1];
        unsigned int dest = shared[s*(2*tid + 2) - 1];
        shared[s*(2*tid + 2) - 1] = dest + addn;
      }
      s *= 2;
    }

    if(tid == 0  ){
      d_elements[nextmark + blockIdx.x] = shared[2*blockDim.x - 1];
      shared[2*blockDim.x - 1] = 0;
    }

    for( int d = 1 ; d < 2*blockDim.x ; d *= 2 ){
      s >>= 1;
      __syncthreads();
      if(tid < d){
        unsigned int addn = shared[s*(2*tid + 1) - 1];
        unsigned int dest = shared[s*(2*tid + 2) - 1];
        shared[s*(2*tid + 1) - 1] = dest;
        shared[s*(2*tid + 2) - 1] = dest + addn;
      }
    }
    __syncthreads();
    
    if( 2*index+1 < numElems  ){
      d_elements[2*index] = shared[2*tid];
      d_elements[2*index + 1] = shared[2*tid + 1];
    }
}

//only use this for the last block of an array.
__global__ void hillisteel_tailsum(

  unsigned int * const d_elements, 
                   int numElems, 
                   int nextmark
)
{

  int tid = threadIdx.x;
  extern __shared__ unsigned int shared[];

  if(tid < numElems){
    shared[tid] = d_elements[tid];
  }
  __syncthreads();

  for( int s = 1 ; s < blockDim.x ; s <<= 1 ){
    unsigned int val = 0;
    int spot = tid - s;
    if( spot >= 0 && tid < numElems ){
      val = shared[spot];
    }
    __syncthreads();
    if( spot >= 0 && tid < numElems ){
      shared[tid] += val;
    }
    __syncthreads();
  }
  if( tid+1 < numElems ){
    d_elements[tid+1] = shared[tid];
  }
  else if( tid == numElems-1){
    //d_elements[nextmark] = shared[tid];
    d_elements[0] = 0;
  }

}


template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n)
{
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { 
    sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    i += gridSize;
  }
  __syncthreads();
  if (blockSize >= 512){ if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256){ if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128){ if (tid < 64 ) { sdata[tid] += sdata[tid + 64 ]; } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
  }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}




__global__ void sum_blelloch_reduce( 
  
  unsigned int* const d_elements,
  unsigned int* const d_results,
          int numElems)
{
    extern __shared__ unsigned int shared[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    if( 2*index+1 < numElems  ){
      shared[2*tid] = d_elements[2*index];
      shared[2*tid+1] = d_elements[2*index + 1 ];
    }
    __syncthreads();

    int s = 1;
    for( int d = blockDim.x ; d > 0  ; d >>= 1 ){
      __syncthreads();
      if( tid < d){ 
        unsigned int addn = shared[s*(2*tid + 1) - 1];
        unsigned int dest = shared[s*(2*tid + 2) - 1];
        shared[s*(2*tid + 2) - 1] = dest + addn;
      }
      s *= 2;
    }

    if(tid == 0  ){
      d_results[blockIdx.x] = shared[2*blockDim.x - 1];
    }

}

__global__ void sum_reduce(   unsigned int* const d_position,
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
      int tmp= 0 ;
      if(tid < s && index < size){
        shared[tid] += shared[tid + s];
      } 
      __syncthreads();
    }

    if(tid == 0){
        d_result[ blockIdx.x ] = shared[tid];
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

  int input = (d_input[index]==binNumber) ;
  d_predicate[index] = input;
}



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

unsigned int find_sum_reduce( const unsigned int* const d_input , int numElems ){
  int blockSize = getBlockSize(numElems);
  int gridSize = getGridSize(numElems);
  int outFElems = gridSize * sizeof(unsigned int); //filesize of the output from one iteration of kernel

  unsigned int * d_result; //pointer to where result of max reduce kernel can be stored
  cudaMalloc( (void**)&d_result, outFElems ); 
  unsigned int * d_input_copy; //a copy of input
  cudaMalloc( (void**)&d_input_copy, sizeof(unsigned int)*numElems );
  cudaMemcpy( d_input_copy, d_input, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice );

  int shareFSize = blockSize * sizeof(unsigned int);
  
  int s = numElems; 
  do{
    int current_gridSz = getGridSize(s);
    sum_reduce<<< current_gridSz, blockSize, shareFSize>>>( d_input_copy, d_result, s );
    cudaDeviceSynchronize();
    cudaMemcpy( d_input_copy, d_result, current_gridSz*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    s /= blockSize;
  }while( s > blockSize );
  unsigned int result = 0;
  cudaMemcpy( &result, d_result , sizeof(unsigned int), cudaMemcpyDeviceToHost );
  return result;
}

unsigned int arbitrary_scan( unsigned int * d_input, int numElems)
{
    unsigned int last = 0;
    gpuErrchk( cudaMemcpy( &last, d_input + (numElems - 1) , sizeof(unsigned int), cudaMemcpyDeviceToHost ));



    //this section is used to set up starting conditions for the kernel
    //as well as information for it to continue - eg the bookmarks are used
    //by the kernel to find indexes for the blocksumming stages of the scan.
    int bookmarks_sz =  128; //@note this is probably un-necessarily large, with a block size of 512, this will only be resizes if numElems ~= 512^128, which is huge.
    vector<int> h_bookmarks(bookmarks_sz, 0); //@improve? 128 is arbitrarily selected, since the final size is unknown, and a resize every single loop is not preferable. Hence a resize every '32' is used.
    int workneeded = numElems;
    int depth = 0;
    int tmp_elems = (numElems+(MAX_BLOCKSZ*2) -1)/(MAX_BLOCKSZ*2);
    while(tmp_elems > 1 ){
      depth++;
      if( depth+1 >= bookmarks_sz-1 ){ //if too deep, increase size to allow for more bookmark entries. 
        bookmarks_sz += 128;
        h_bookmarks.resize(bookmarks_sz); 
      }
      h_bookmarks[depth] = workneeded; //@note depth 0 case was not specifically handled by loop, but is set to 0 by initialisation of the vector, and is simply skipped by loop.
      workneeded += tmp_elems;
      tmp_elems = (tmp_elems+(MAX_BLOCKSZ*2) - 1)/(MAX_BLOCKSZ*2);
    }
    h_bookmarks[depth+1] = workneeded;

    int worksize = (workneeded+1) * sizeof(unsigned int); //plus 1 since sometimes the kernel will operate on an index 1 more than max without checking.
    int filesize = numElems * sizeof(unsigned int);
    int shareSize = 2*MAX_BLOCKSZ * sizeof(unsigned int);
    
    unsigned int * d_elements;
    gpuErrchk( cudaMalloc((void**)&d_elements, worksize));
    //gpuErrchk( cudaMemset( d_elements, 0, worksize));
    gpuErrchk( cudaMemcpy( d_elements, d_input, filesize , cudaMemcpyDeviceToDevice )); //change back to filesize@test
    for(  int i = 0,
              it_numElems = numElems
          ; i <= depth ; i++ )
    {
      
      //Set up initial conditions
      int it_remSz  = it_numElems%(2*MAX_BLOCKSZ),
          it_gridSz = it_numElems/(2*MAX_BLOCKSZ),
          remMark   = h_bookmarks[i+1] - it_remSz,
          nextmark  = h_bookmarks[i+1] - h_bookmarks[i];
      //run the relevant kernel if conditions are correct
      if( it_gridSz ){ blelloch_threadsum<<<it_gridSz, MAX_BLOCKSZ, shareSize>>>( d_elements + h_bookmarks[i] , it_numElems, nextmark ); }
      if( it_remSz  ){ hillisteel_tailsum<<<1, it_remSz,shareSize>>>( d_elements + remMark, it_remSz, nextmark + it_gridSz ); }
      cudaDeviceSynchronize();

      //increment the elements in loop
      it_numElems = it_gridSz + (it_remSz >= 1) ;
      it_gridSz   = it_numElems;
    }
    
    for( int i = depth ; i > 0 ; i--){
      int it_numElems  = h_bookmarks[i] - h_bookmarks[i-1],
          it_blockSz   = getBlockSize(it_numElems),
          it_gridSz    = getGridSize (it_numElems),
          nextmark     = h_bookmarks[i] - h_bookmarks[i-1];
      bookmark_blocksum<<<it_gridSz,it_blockSz>>>(d_elements + h_bookmarks[i-1], it_numElems, nextmark);
      cudaDeviceSynchronize();
    }

    //Calculate the value that would be taken by summing all elements - ie same as last element in an inclusive scan.
    unsigned int reduced = 0;
    gpuErrchk( cudaMemcpy( &reduced, d_elements + (numElems - 1) , sizeof(unsigned int), cudaMemcpyDeviceToHost ));
    reduced += last;
    //copy results to d_input
    gpuErrchk( cudaMemcpy( d_input, d_elements, filesize, cudaMemcpyDeviceToHost)); 
    cudaFree(d_elements);
    return reduced;

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



  int filesize  = numElems * sizeof(unsigned int);
  int gridSize  = getGridSize ( numElems );
  int blockSize = getBlockSize( numElems );
  vector<unsigned int> h_histo( numBins, 0 );

  for( int binNumber = 0; binNumber < numBins ; binNumber++ ){
    unsigned int * d_predicate ;
    gpuErrchk( cudaMalloc( (void**)&d_predicate , filesize) );
    radix_predicate<<<gridSize,blockSize>>>( d_vals, d_predicate, binNumber, numElems);

    h_histo[binNumber] = arbitrary_scan( d_predicate, numElems);
    //cout << scanned_predicate << endl;
    //cudaDeviceSynchronize();
    cudaFree(d_predicate);

  }
  cudaMemcpy( d_histo, h_histo.data(), numBins * sizeof(unsigned int) , cudaMemcpyHostToDevice );

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
    unsigned int binsize = NUMBINS * sizeof(unsigned int);
    unsigned int * h_histo = (unsigned int *)malloc(binsize);

    gpuErrchk( cudaMalloc((void**)&d_values, filesize ));
    gpuErrchk( cudaMalloc((void**)&d_histo, NUMBINS*sizeof(unsigned int) ));

    gpuErrchk( cudaMemcpy( d_values, h_values, filesize, cudaMemcpyHostToDevice ) );

    computeHistogram_test( d_values, d_histo, NUMBINS, numElems );  
    gpuErrchk( cudaMemcpy( h_histo, d_histo, binsize, cudaMemcpyDeviceToHost ));



  }

}
