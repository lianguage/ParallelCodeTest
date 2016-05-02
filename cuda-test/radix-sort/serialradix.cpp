#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <iostream>
#include <climits>
#include <fstream>
#include <string>

//A simple un-optimized reference radix sort calculation
//Only deals with power-of-2 radices


void your_sort(unsigned int* inputVals,
               unsigned int* inputPos,
               unsigned int* outputVals,
               unsigned int* outputPos,
               const size_t numElems)
{
  const int numBits = 1;
  const int numBins = 1 << numBits;

  unsigned int *binHistogram = new unsigned int[numBins];
  unsigned int *binScan      = new unsigned int[numBins];

  unsigned int *vals_src = inputVals;
  unsigned int *pos_src  = inputPos;

  unsigned int *vals_dst = outputVals;
  unsigned int *pos_dst  = outputPos;

  //a simple radix sort - only guaranteed to work for numBits that are multiples of 2
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;

    memset(binHistogram, 0, sizeof(unsigned int) * numBins); //zero out the bins
    memset(binScan, 0, sizeof(unsigned int) * numBins); //zero out the bins

    //perform histogram of data & mask into bins
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      binHistogram[bin]++;
    }

    //perform exclusive prefix sum (scan) on binHistogram to get starting
    //location for each bin
    for (unsigned int j = 1; j < numBins; ++j) {
      binScan[j] = binScan[j - 1] + binHistogram[j - 1];
    }

    //Gather everything into the correct location
    //need to move vals and positions
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      vals_dst[binScan[bin]] = vals_src[j];
      pos_dst[binScan[bin]]  = pos_src[j];
      binScan[bin]++;
    }

    //swap the buffers (pointers only)
    std::swap(vals_dst, vals_src);
    std::swap(pos_dst, pos_src);
  }

  //we did an even number of iterations, need to copy from input buffer into output
  std::copy(inputVals, inputVals + numElems, outputVals);
  std::copy(inputPos, inputPos + numElems, outputPos);

  delete[] binHistogram;
  delete[] binScan;
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
    unsigned int * h_out_numbers  = (unsigned int *)malloc(filesize);
    unsigned int * h_out_position = (unsigned int *)malloc(filesize);

    //run sort
    your_sort(h_numbers, h_position, h_out_numbers, h_out_position, lines );
    //cudaDeviceSynchronize();


    ofstream sortedfile("sorted");
    if( sortedfile.is_open()){
      for(int i = 0 ; i < lines ; i++){
        sortedfile << std::to_string(h_numbers[i]) << "         " <<  to_string(h_out_numbers[i]) << "         " <<   to_string(h_position[i])  << "       "  << to_string(h_out_position[i]) <<"\n";
      }
    }
    std::cout << "finished" <<std::endl;

    return 0;

  }

}