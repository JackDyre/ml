#include "device_ptr.h"
#include <cassert>

DevicePtr::DevicePtr() : Ptr() {}

DevicePtr::DevicePtr(float *ptr) : Ptr(ptr) {}

DevicePtr::DevicePtr(DevicePtr &&other) noexcept : Ptr(std::move(other)) {}

DevicePtr &DevicePtr::operator=(DevicePtr &&other) noexcept {
  Ptr::operator=(std::move(other));
  return *this;
}

DevicePtr::~DevicePtr() = default;

void DevicePtr::alloc(std::size_t size) {
  float *ptr;
  auto err = cudaMalloc(&ptr, size * sizeof(float));
  assert(err == cudaSuccess);
  *this = DevicePtr(ptr);
}

void DevicePtr::free() {
  if (!is_null()) {
    auto err = cudaFree(const_cast<float *>(as_inner()));
    assert(err = cudaSuccess);
  }
}
