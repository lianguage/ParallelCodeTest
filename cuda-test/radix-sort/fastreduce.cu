#include "assert.h"

#include <cuda_runtime.h> //used for assert
#include <iostream>

#include <fstream>
#include <string>
#include <climits>


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
      d_elements[nextmark + blockIdx.x] = shared[2*blockDim.x - 1];
      shared[2*blockDim.x - 1] = 0;
    }

}


using namespace std;








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
