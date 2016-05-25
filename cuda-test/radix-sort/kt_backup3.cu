
#include "assert.h"

#include <cuda_runtime.h> //used for assert
#include <iostream>

#include <vector>

#include <fstream>
#include <string>
#include <climits>

//backup version stored as kt_backup.cu

//#define MAXSHID2 2*blockDim.x-1
#define INNERGRID gridDim.x/blockDim.x
#define MAX_BLOCKSZ 512

__device__ int d_getBlockSize(int numElems){
  if(numElems > MAX_BLOCKSZ){
    return MAX_BLOCKSZ;
  }
  else{
    return numElems;
  }
}

__device__ int d_getGridSize(int numElems){
  return (numElems + MAX_BLOCKSZ -1)/MAX_BLOCKSZ;
}

__global__ void bookmark_blocksum( 
  unsigned int* d_elements, 
           int* d_bookmarks,
            int blockSumNumElems,
            int depth)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if( depth > 0 && 2*index+1 < blockSumNumElems){
    int blocks_mark = d_bookmarks[depth-1];
    int sum_mark = d_bookmarks[depth];
    //d_elements[blocks_mark + index] = 0;
    d_elements[blocks_mark + 2*index] +=  d_elements[sum_mark + blockIdx.x];
    d_elements[blocks_mark + 2*index + 1 ] +=  d_elements[sum_mark + blockIdx.x];
  }
  else{
    //throw some kind of error. consider using assert or something here instead.
  }

}


__global__ void blelloch_threadsum(
  unsigned int* d_elements,
            int numElems,
            int nextmark)
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

    if(gridDim.x > 1){
      d_elements[nextmark + blockIdx.x] = shared[2*blockDim.x - 1];
    }
    
    if(tid == 0  ){
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
    
    d_elements[2*index] = shared[2*tid];
    d_elements[2*index + 1] = shared[2*tid + 1];
    __syncthreads();
    
}


__global__ void hillisteel_tailsum( //only use this for the last block of an array.
                  unsigned int * const d_elements,
                                   int numElems,
                                   int nextmark)
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
  d_elements[tid] = shared[tid];

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

void arbitrary_scan( unsigned int * h_elements, int numElems){
    //allocate memory for device
    unsigned int * d_elements;
    int shareSize = 2*MAX_BLOCKSZ * sizeof(unsigned int);

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
      h_bookmarks[depth] = workneeded; //note depth 0 case was not specifically handled by loop, but is set to 0 by initialisation of the vector, and is simply skipped by loop.
      workneeded += tmp_elems;
      tmp_elems = (tmp_elems+(MAX_BLOCKSZ*2) - 1)/(MAX_BLOCKSZ*2);
    }
    h_bookmarks[depth+1] = workneeded;

    int worksize = (workneeded+1) * sizeof(unsigned int); //plus 1 since sometimes the kernel will operate on an index 1 more than max without checking.
    int filesize = numElems * sizeof(unsigned int);

    gpuErrchk( cudaMalloc((void**)&d_elements, worksize));
    gpuErrchk( cudaMemset( d_elements, 0, worksize));
    gpuErrchk( cudaMemcpy( d_elements, h_elements, filesize , cudaMemcpyHostToDevice )); //change back to filesize@test
    
    
    for(  int i = 0,
              it_numElems = numElems
          ; i <= depth ; i++ )
    {
      int it_remSz  = it_numElems%(2*MAX_BLOCKSZ);
      int it_gridSz = it_numElems/(2*MAX_BLOCKSZ);
      int remMark   = h_bookmarks[i+1] - it_remSz;
      int nextmark  = h_bookmarks[i+1] - h_bookmarks[i];
      std::cout << "h_bookmarks[" << i+1 << "]" <<h_bookmarks[i+1] << std::endl;
      std::cout << "remMark:"   << remMark   << std::endl;
      std::cout << "it_remSz:"  << it_remSz  << std::endl;
      std::cout << "it_gridSz:" << it_gridSz << std::endl;
      std::cout << "shareSize:" << shareSize << std::endl;
      if( it_gridSz ){ blelloch_threadsum<<<it_gridSz, MAX_BLOCKSZ, shareSize>>>( d_elements + h_bookmarks[i] , it_numElems, nextmark ); }
      //cudaDeviceSynchronize();
      if( it_remSz  ){ hillisteel_tailsum<<<1, it_remSz,shareSize>>>( d_elements + remMark, it_remSz, nextmark ); }
      cudaDeviceSynchronize();
      
      it_numElems = it_gridSz + (it_remSz >= 1) ;
      it_gridSz   = it_numElems;
    }
    //blelloch_threadsum<<<gridSize,blockSize,shareSize>>>(d_elements, d_bookmarks, numElems, 0)  ;
    /*
    for( int i = depth ; i >= 1 ; i--){
      int it_numElems  = h_bookmarks[i] - h_bookmarks[i-1],
          it_blockSz   = getBlockSize(it_numElems),
          it_gridSz    = getGridSize (it_numElems);
      blelloch_blocksum<<<it_gridSz,it_blockSz>>>(d_elements, d_bookmarks,it_numElems, i);
      cudaDeviceSynchronize();
    }*/
    unsigned int* h_test_elements = (unsigned int*)malloc(worksize*sizeof(unsigned int));
    gpuErrchk( cudaMemcpy( h_test_elements, d_elements, worksize, cudaMemcpyDeviceToHost)); //@test - used for checking memory in cuda-gdb.
    gpuErrchk( cudaMemcpy( h_elements, d_elements, filesize, cudaMemcpyDeviceToHost)); 
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

    if( myfile.is_open()){
      getline(myfile,line);
      size = atoi(line.c_str());//first line of file is assumed to show number of elements in file.
      filesize = sizeof(int)*size;
      h_numbers = (unsigned int *)malloc(filesize);
      int i = 0;
      while(getline(myfile, line)){
         h_numbers[i] = atoi(line.c_str());

         lines++;
         i++;
      }
      myfile.close();
    }
    else {
      cout << "Sorry mate, can't load file" << endl;
      return 0;
    }

    arbitrary_scan(h_numbers, lines);

    ofstream sortedfile("sorted");
      if( sortedfile.is_open()){
        for(int i = 0 ; i < lines; i++){
          sortedfile << std::to_string(h_numbers[i]) <<"\n";
        }
      }
    std::cout << "finished" <<std::endl;

    return 0;

  }
}