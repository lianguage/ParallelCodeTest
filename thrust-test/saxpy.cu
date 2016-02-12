#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

struct saxpy_functor
{
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  { 
    return a * x + y;
  }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  thrust::device_vector<float> temp(X.size());

  // temp <- A
  thrust::fill(temp.begin(), temp.end(), A);

  // temp <- A * X
  thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

  // Y <- A * X + Y
  thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

#define BIG 262144

int main(){


  saxpy_functor moo(1234.51234);
  std::cout << moo( 4345.434, 345325.34 ) << std::endl;
  thrust::device_vector<float> x(BIG);
  thrust::device_vector<float> y(BIG);
  
}