#include "device_ptr.h"
#include <cassert>

DevicePtr &DevicePtr::operator=(DevicePtr &&other) noexcept {
  Ptr::operator=(std::move(other));
  return *this;
}

DevicePtr::~DevicePtr() { _free(const_cast<float *>(as_inner())); }

float *DevicePtr::_alloc(std::size_t size) {
  float *ptr;
  auto err = cudaMalloc(&ptr, size * sizeof(float));
  assert(err == cudaSuccess);
  return ptr;
}

void DevicePtr::_free(float *ptr) {
  if (ptr != nullptr) {
    auto err = cudaFree(const_cast<float *>(ptr));
    assert(err == cudaSuccess);
  }
}
