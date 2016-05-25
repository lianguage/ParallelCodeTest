
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


__global__ void blelloch_blocksums( 
  unsigned int* d_elements,
            int blocks_mark, //blocks is the larger array
            int sum_mark, //gets added to blocks
            int blockSumNumElems)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if( 2*index+1 < blockSumNumElems){
    d_elements[blocks_mark + 2*index] +=  d_elements[sum_mark + blockIdx.x];
    d_elements[blocks_mark + 2*index + 1 ] +=  d_elements[sum_mark + blockIdx.x];
  }
  else{
    //throw some kind of error. consider using assert or something here instead.
  }


}


__global__ void blelloch_threadscan(
  unsigned int* d_elements,
            int bookmark,
            int numElems
)
{
    extern __shared__ unsigned int shared[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

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

    //if gridDim is greater than one, store the block sum in the bookmarked section
    //of global memory pointed to by d_elements.
    if(gridDim.x > 1){
      d_elements[bookmark + 1 + blockIdx.x] = shared[MAXSHID2];
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


inline int getBlockSize(int numElems){
  if(numElems > MAX_BLOCKSZ){
    return MAX_BLOCKSZ;
  }
  else{
    return numElems;
  }
}

inline int getGridSize(int numElems){
  return (numElems + MAX_BLOCKSZ -1)/MAX_BLOCKSZ;
}

void arbitrary_scan( unsigned int * h_elements, int numElems){
    //allocate memory for device
    unsigned int * d_elements;
    int gridSize  = getGridSize(numElems/2);
    int blockSize = getBlockSize(numElems/2);
    int shareSize = 2*blockSize * sizeof(unsigned int);

    //calculate array size(workneeded) and number of iterations(depth)
    int workneeded = numElems;
    int depth = 0;
    int bookmarks_sz =  32; 
    vector<int> h_bookmarks(bookmarks_sz, 0); //@improve? 128 is arbitrarily selected, since the final size is unknown, and a resize every single loop is not preferable. Hence a resize every '32' is used.
    for( int tmp_elems = gridSize ; tmp_elems > 1 ; tmp_elems = getGridSize(tmp_elems/blockSize) ) {
      depth++;
      if( depth >= bookmarks_sz ){ //if too deep, increase size to allow for more bookmark entries. 
        bookmarks_sz += 128;
        h_bookmarks.resize(bookmarks_sz); 
      }
      h_bookmarks[depth] = workneeded;
      workneeded += tmp_elems;
    }
    int worksize = (workneeded+1) * sizeof(unsigned int); //plus 1 since sometimes the kernel will operate on an index 1 more than max without checking.
    int filesize = numElems * sizeof(unsigned int);

    gpuErrchk( cudaMalloc((void**)&d_elements, worksize));
    gpuErrchk( cudaMemcpy( d_elements, h_elements, filesize , cudaMemcpyHostToDevice )); //change back to filesize@test

    //blelloch2_reduction<<<gridSize,blockSize,shareSize>>>( d_elements, 0, numElems );

    for( int i = 0,
             bookmark    = 0,
             it_gridSz   = gridSize,
             it_blockSz  = blockSize,
             it_numElems = numElems
         ;i <= depth ; i++ ){
      blelloch_threadscan<<<it_gridSz,it_blockSz,shareSize>>>( d_elements, bookmark, it_numElems );
      cudaDeviceSynchronize();
      bookmark   += it_numElems;
      it_numElems = it_gridSz;
      it_gridSz   = getGridSize (it_numElems);
      it_blockSz  = getBlockSize(it_numElems);
    }

    /*
    for( int i = depth ; i >= 0 ; i-- ){
      int blocks_mark = h_bookmarks[i - 1];
      int sum_mark = h_bookmarks[i];
      int blockSumNumElems = sum_mark - blocks_mark;
      int it_blockSz = getBlockSize(blockSumNumElems);
      int it_gridSz = getGridSize(blockSumNumElems);
      blelloch_blocksums<<<it_gridSz,it_blockSz>>>( d_elements, blocks_mark, sum_mark, blockSumNumElems);
      cudaDeviceSynchronize();
    }*/

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