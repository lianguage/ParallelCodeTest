
#include "assert.h"

#include <cuda_runtime.h> //used for assert
#include <iostream>

#include <vector>

#include <fstream>
#include <string>
#include <climits>

//backup version stored as kt_backup.cu

#define MAXSHID2 2*blockDim.x-1
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

__global__ void blelloch_blocksums( 
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


__global__ void blelloch2_reduction(
  unsigned int* d_elements,
           int* d_bookmarks,
            int numElems,
            int depth)
{
    extern __shared__ unsigned int shared[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    int bookmark = d_bookmarks[depth];
    //int nextmark = d_bookmarks[depth+1];

    if( 2*index + 1 < numElems ){
      shared[2*tid] = d_elements[2*index + bookmark];
      shared[2*tid+1] = d_elements[2*index+1 + bookmark];
    }
    __syncthreads();

    //this loop will work in a way that a binomial tree rooted at tid = MAXSHID2 will be formed
    //to root it at zero, remove the MAXSHID2 component.
    for( int s = 1; s <= blockDim.x ; s <<= 1 ){
      if( 2*s*tid + s < 2*blockDim.x ){ // in the books tid = 0 is shown as the first, however in this code it refers to the last element in the respective diagram. see: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
        unsigned int dest = shared[MAXSHID2 - 2*s*tid]; //the inclusion of blockdim and subtracting the 2*s*tid factor in the index reverses the order of if only the factor is used.
        unsigned int addn = shared[MAXSHID2 - 2*s*tid - s];
        shared[MAXSHID2 - 2*s*tid] = dest + addn;
      }
      __syncthreads();
    }
    if(gridDim.x > 1){
      int nextmark = d_bookmarks[depth+1];
      d_elements[nextmark + blockIdx.x] = shared[MAXSHID2];
    }

    if(tid == 0  ){
      shared[MAXSHID2] = 0;
    }

    for( int s = blockDim.x ; s > 0 ; s >>= 1){
      if( 2*s*tid + s < 2*blockDim.x ){
        unsigned int dest = shared[MAXSHID2 - 2*s*tid];
        unsigned int addn = shared[MAXSHID2 - 2*s*tid - s];
        shared[MAXSHID2 - 2*s*tid] = dest + addn;
        shared[MAXSHID2 - 2*s*tid - s] = dest;
      }
      __syncthreads();
    }

    if( 2*index + 1 < numElems ){
      d_elements[2*index + bookmark] = shared[2*tid];
      d_elements[2*index + 1 + bookmark] = shared[2*tid + 1];
    }
    __syncthreads();

    
    if( index == 0 && gridDim.x > 1 ){
      int innerblockdim = min(blockDim.x,gridDim.x); 
      int innergriddim = (gridDim.x + blockDim.x -1)/blockDim.x;
      blelloch2_reduction<<<innergriddim,innerblockdim>>>( d_elements, d_bookmarks , gridDim.x, depth+1 );
    }
    else if( index == 0 && gridDim.x == 1){
      int bsum_numElems = d_bookmarks[depth] - d_bookmarks[depth-1];
      int bsum_blockdim = d_getBlockSize(bsum_numElems);
      int bsum_griddim = d_getGridSize(bsum_numElems);
      blelloch_blocksums<<<bsum_griddim,bsum_blockdim>>>(d_elements, d_bookmarks, bsum_numElems, depth);
    }

    
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
    int gridSize = getGridSize(numElems/2);
    int blockSize = getBlockSize(numElems/2);
    int shareSize = 2*blockSize * sizeof(unsigned int);

    //this section is used to set up starting conditions for the kernel
    //as well as information for it to continue - eg the bookmarks are used
    //by the kernel to find indexes for the blocksumming stages of the scan.
    int bookmarks_sz =  128; //@note this is probably un-necessarily large, with a block size of 512, this will only be resizes if numElems ~= 512^128, which is huge.
    vector<int> h_bookmarks(bookmarks_sz, 0); //@improve? 128 is arbitrarily selected, since the final size is unknown, and a resize every single loop is not preferable. Hence a resize every '32' is used.
    int workneeded = numElems;
    int depth = 0;
    for( int tmp_elems = gridSize ; tmp_elems > 1 ; tmp_elems = getGridSize(tmp_elems/blockSize) ) {
      depth++;
      if( depth >= bookmarks_sz ){ //if too deep, increase size to allow for more bookmark entries. 
        bookmarks_sz += 128;
        h_bookmarks.resize(bookmarks_sz); 
      }
      h_bookmarks[depth] = workneeded; //note depth 0 case was not specifically handled by loop, but is set to 0 by initialisation of the vector, and is simply skipped by loop.
      workneeded += tmp_elems;
    }
    int* d_bookmarks;
    gpuErrchk( cudaMalloc((void**)&d_bookmarks, (depth+1)*sizeof(int) ));
    gpuErrchk( cudaMemcpy(d_bookmarks, h_bookmarks.data(), (depth+1)*sizeof(int), cudaMemcpyHostToDevice ));

    int worksize = (workneeded+1) * sizeof(unsigned int); //plus 1 since sometimes the kernel will operate on an index 1 more than max without checking.
    int filesize = numElems * sizeof(unsigned int);

    gpuErrchk( cudaMalloc((void**)&d_elements, worksize));
    cudaDeviceSynchronize();

    //copy memory host to device
    gpuErrchk( cudaMemcpy( d_elements, h_elements, filesize , cudaMemcpyHostToDevice )); //change back to filesize@test
    cudaDeviceSynchronize();

    blelloch2_reduction<<<gridSize,blockSize,shareSize>>>( d_elements, d_bookmarks, numElems, 0 );
    cudaDeviceSynchronize();

    gpuErrchk( cudaMemcpy( h_elements, d_elements, filesize, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
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