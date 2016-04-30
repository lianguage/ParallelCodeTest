/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include "stdio.h"

//1: find Max kernel
__global__ void reduction_max(const float* const d_in,
                     float* d_out,
                     int imgSize)
{ 
    //extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.y;
    int tid = threadIdx.x;
    if( myId > imgSize){return;}

    //sdata[myId] = d_in[myId];
    __syncthreads();
    /*
    for( int s = blockDim.x/2 ; s > 0 ; s >>= 1){
      if(tid < s){
        sdata[myId] = max(sdata[myId], sdata[myId+s]);
      }
      __syncthreads();
    }
    */
    //if(tid == 0){
    //  d_out[blockIdx.x] = sdata[0];
    //}

    //*d_out = 1.0;
}

//2 : find Min kernel
__global__ void reduction_min(const float* const d_in,
                     float* d_out,
                     int imgSize)
{ 
    //extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.y;
    int tid = threadIdx.x;
    if( myId > imgSize){return;}

    //sdata[myId] = d_in[myId];
    __syncthreads();
    /*
    for( int s = blockDim.x/2 ; s > 0 ; s >>= 1){
      if(tid < s){
        sdata[myId] = min(sdata[myId], sdata[myId+s]);
      }
      __syncthreads();
    }*/
    //if(tid == 0){
    //  d_out[blockIdx.x] = sdata[0];
    //}
    //*d_out = 1.0;

}

//3 : simple initial histogram kernel
__global__ void histogram( float min, float max, int numBins,const float * const d_in, int * d_histogram ){

    int myId = threadIdx.x + blockDim.x * blockIdx.x; 
    int binId =  (d_in[myId] - min) / (max-min) * numBins;
    atomicAdd( &(d_histogram[binId]), 1 );

}


//4: simple initial cdf kernel
//NOTE: d_histogram needs to be copied to d_cdf BEFORE running kernel, otherwise pointless
__global__ void exclusive_scan_cdf( int * d_histogram, unsigned int* const d_cdf , int numBins ){
    int myId = threadIdx.x + blockDim.x * blockIdx.x; //assuming flat
    d_cdf[myId] = d_histogram[myId]; //copy all values of d_histogram to d_cdf
    __syncthreads();
    for(int i = 0 ; ; i++){
      int neighbour = pow((double)2, (double)i );
      if( neighbour >= numBins ){break;}
      if( (myId - neighbour) >= 0){
         d_cdf[myId] += d_cdf[myId-neighbour];
      }
      __syncthreads();
    }
    //__syncthreads();

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)*/
    const dim3 blockSize(32,32,1);
    const dim3 gridSize((numCols+32-1)/32,(numRows+32-1)/32,1);
    const int flatBlockSize = 1024;
    const int flatGridSize = (numCols*numRows+1024-1)/1024;
    const int imgSize = numCols*numRows;
    
    
    //1: Find max
    float *d_lumi_max;
    float *h_lumi_max = (float*)malloc( sizeof(float));
    cudaMalloc(&d_lumi_max, sizeof(float)*imgSize);

    //reduction_max<<<flatGridSize,flatBlockSize, sizeof(float)*flatBlockSize*flatGridSize>>>( d_logLuminance, d_lumi_max, imgSize );
    reduction_max<<<flatGridSize,flatBlockSize>>>( d_logLuminance, d_lumi_max, imgSize );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy( h_lumi_max, d_lumi_max, sizeof(float), cudaMemcpyDeviceToHost ));
    max_logLum = *h_lumi_max;

    //2: Find minimum
    float *d_lumi_min;
    float *h_lumi_min = (float*)malloc( sizeof(float));
    cudaMalloc(&d_lumi_min, sizeof(float)*imgSize);

    //reduction_min<<<flatGridSize,flatBlockSize, sizeof(float)*flatBlockSize*flatGridSize>>>( d_logLuminance, d_lumi_min, imgSize );
    reduction_min<<<flatGridSize,flatBlockSize>>>( d_logLuminance, d_lumi_min, imgSize );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy( h_lumi_min, d_lumi_min, sizeof(float), cudaMemcpyDeviceToHost ));
    min_logLum = *h_lumi_min;
    
    max_logLum = 9999;
    min_logLum = 30;

    //3:  Generate Histogram
    int * d_histogram;
    //int * h_histogram = (int*)malloc( sizeof(int) * numBins);
    //cudaMalloc(&d_histogram, sizeof(int)*numBins );
    cudaMemset(d_histogram, 0, sizeof(int)*numBins );
    histogram<<< flatGridSize,flatBlockSize >>>( max_logLum , min_logLum , numBins, d_logLuminance, d_histogram );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    //checkCudaErrors(cudaMemcpy( h_histogram, d_histogram, sizeof(int)*numBins, cudaMemcpyDeviceToHost ));
/*
    //4:  Exclusive scan CDF
    int cdfBlockFlat = 128;
    int cdfGridFlat = (numBins+128-1)/128;
    exclusive_scan_cdf<<< cdfGridFlat, cdfBlockFlat >>> ( d_histogram, d_cdf , numBins);
    //checkCudaErrors(cudaMemcpy( h_histogram, d_histogram, sizeof(int)*numBins, cudaMemcpyDeviceToHost ));
*/
}