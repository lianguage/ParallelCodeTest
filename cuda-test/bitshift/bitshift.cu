#include <iostream>
#include <string>
#include <fstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void radix_predicate(   unsigned int* const d_input,
                                   //unsigned int* const d_position,
                                   bool* const d_predicate,
                                   int bitsig,
                                   int size
                                   ){
   
   int index = threadIdx.x + blockDim.x*blockIdx.x;
   if(index > size ){ return; }

   //int position = d_position[index];
   //predicate: (i & 1) == 0;
   unsigned int x = d_input[index];
   x >>= bitsig;
   d_predicate[index] = x&1;
}

__global__ void radix_scan1( bool* const d_predicate,
                            unsigned int* const d_xscan,
                            unsigned int identity,
                            unsigned int * d_middle,
                            int size ){ //identity variable is used to set the first term. This is used so that the indexes for 
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index<size) {d_xscan[index] = d_predicate[index]; }
	else{return;}

	for( int s = 1 ; s < size ; s <<= 1 ){
		unsigned int val = 0;
		int spot = index - s;
		if( spot >= 0){
		 val = d_xscan[spot];
		}
		__syncthreads();
		if( spot >= 0){
		 d_xscan[index] += val;
		}
		__syncthreads();
	}
	if( index == 0 ){ d_xscan[index] = 0 ;}
	else{
		unsigned int temp = d_xscan[index-1];
		d_xscan[index] = temp;
	}
	__syncthreads();
	d_xscan[index] += identity;
	if( index == size - 1){ *d_middle = d_xscan[index] + d_predicate[size-1]; }

}

__global__ void radix_invert_predicate( bool* const d_predicate,
                                        int size){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if( index > size ){ return;}
	d_predicate[index] = !d_predicate[index];
}


__global__ void radix_split( unsigned int* const d_position1,
						     unsigned int* const d_position2,
                             bool* const d_predicate,
                             unsigned int* const d_values,
                             int size){

	//todo
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index > size ){ return; }
	unsigned int value= 0;
	if(d_predicate[index]){
		value = d_values[d_position2[index]];
	}
	else{
		value = d_values[d_position1[index]];
	}
	d_values[index] = value;

}



#define MAX_BLOCKSZ 1024

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

using namespace std;

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
			//cout << "moo:" << size << endl;
			while(getline(myfile, line)){
				 //cout << line << endl;
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
		unsigned int * d_numbers;
		unsigned int * d_middle;
		unsigned int h_middle = 0;
		unsigned int * d_position1;
		unsigned int * d_position2;
		unsigned int * h_position1 = (unsigned int*) malloc( sizeof(unsigned int)*lines);
		unsigned int * h_position2 = (unsigned int*) malloc( sizeof(unsigned int)*lines);
		bool * d_bools;
		bool * h_bools = (bool*)malloc(sizeof(bool)*lines);

		gpuErrchk(cudaMalloc((void**)&d_numbers, filesize ));
		gpuErrchk(cudaMalloc((void**)&d_middle, sizeof(unsigned int) ));
		gpuErrchk(cudaMalloc((void**)&d_position1, filesize ));
		gpuErrchk(cudaMalloc((void**)&d_position2, filesize ));
		gpuErrchk(cudaMalloc( (void**) & d_bools, lines*sizeof(bool)));

		gpuErrchk(cudaMemcpy( d_numbers, h_numbers , filesize , cudaMemcpyHostToDevice ));
		gpuErrchk(cudaMemcpy( d_position1, h_position , filesize , cudaMemcpyHostToDevice ));
		gpuErrchk(cudaMemcpy( d_position2, h_position , filesize , cudaMemcpyHostToDevice ));
		gpuErrchk(cudaMemset( d_bools, lines, false ));

		int blockSize = getBlockSize(lines);
		int gridSize = getGridSize(lines);
		radix_predicate<<<gridSize,blockSize>>> (d_numbers,d_bools, 1,lines);
		radix_scan1<<<gridSize,blockSize>>> (d_bools, d_position1, h_middle, d_middle , lines);
		radix_invert_predicate<<<gridSize,blockSize>>> (d_bools, lines);
		gpuErrchk(cudaMemcpy( &h_middle, d_middle, sizeof(unsigned int) , cudaMemcpyDeviceToHost ));
		radix_scan1<<<gridSize,blockSize>>> (d_bools, d_position2, h_middle, d_middle , lines);
		radix_split<<<gridSize,blockSize>>> (d_position1, d_position2, d_bools, d_numbers, lines);



		gpuErrchk( cudaMemcpy( h_numbers, d_numbers, filesize , cudaMemcpyDeviceToHost));
		gpuErrchk( cudaMemcpy( h_bools, d_bools, lines, cudaMemcpyDeviceToHost));
		gpuErrchk( cudaMemcpy( h_position1, d_position1, filesize, cudaMemcpyDeviceToHost));
		gpuErrchk( cudaMemcpy( h_position2, d_position2, filesize, cudaMemcpyDeviceToHost));
		ofstream sortedfile("sorted");
		if( sortedfile.is_open()){
			sortedfile << " bools : {";
			for(int i = 0 ; i < lines ; i++){
				sortedfile << std::to_string(h_bools[i]) << ", ";
				//std::cout << "adsf" <<std::endl; //@test
			}
			sortedfile << "} \n";
			sortedfile << " numbers : {";
			for(int i = 0 ; i < lines ; i++){
				sortedfile << std::to_string(h_numbers[i]) << ", ";
				//std::cout << "adsf" <<std::endl; //@test
			}
			sortedfile << "} \n";
			sortedfile << " position1 : {";
			for(int i = 0 ; i < lines ; i++){
				sortedfile << std::to_string(h_position1[i]) << ", ";
				//std::cout << "adsf" <<std::endl; //@test
			}
			sortedfile << "} \n";
			sortedfile << " position 2 : {";
			for(int i = 0 ; i < lines ; i++){
				sortedfile << std::to_string(h_position2[i]) << ", ";
				//std::cout << "adsf" <<std::endl; //@test
			}
			sortedfile << "} \n";
			
				
		}
		std::cout << gridSize << endl;
		std::cout << "moo" <<std::endl;

	}


}
