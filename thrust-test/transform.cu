#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

int main(void)
{
  // allocate three device_vectors with 10 elements
  thrust::device_vector<int> X(10);
  thrust::device_vector<int> Y(10);
  thrust::device_vector<int> Z(10);

  // initialize X to 0,1,2,3, ....
  std::cout << "initialize X to 0,1,2,3, ...." << std::endl;
  thrust::sequence(X.begin(), X.end());
  thrust::copy(X.begin(), X.end(), std::ostream_iterator<int>(std::cout, "\n"));


  // compute Y = -X
  std::cout << "compute Y = -X:" << std::endl;
  thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());
  thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));
  
  // fill Z with twos
  std::cout << "fill Z with twos" << std::endl;
  thrust::fill(Z.begin(), Z.end(), 2);
  thrust::copy(Z.begin(), Z.end(), std::ostream_iterator<int>(std::cout, "\n"));

  // compute Y = X mod 2
  std::cout << "compute Y = X mod 2" << std::endl;
  thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());
  thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));

  // replace all the ones in Y with tens
  std::cout << "replace all the ones in Y with tens" << std::endl;
  thrust::replace(Y.begin(), Y.end(), 1, 10);
  thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));

  return 0;    
}