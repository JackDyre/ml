#include <iostream>
int main(void) {
  cudaDeviceProp prop;
  auto err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    return 1;
  }
  std::cout << "sm_" << prop.major << prop.minor;
}
