#include "host_ptr.h"
#include <cassert>

HostPtr::HostPtr() : Ptr() {}

HostPtr::HostPtr(float *ptr) : Ptr(ptr) {}

HostPtr::HostPtr(HostPtr &&other) noexcept : Ptr(std::move(other)) {}

HostPtr &HostPtr::operator=(HostPtr &&other) noexcept {
  Ptr::operator=(std::move(other));
  return *this;
}

HostPtr::~HostPtr() { _free(const_cast<float *>(as_inner())); }

float *HostPtr::_alloc(std::size_t size) {
  float *ptr = static_cast<float *>(std::malloc(size * sizeof(float)));
  assert(ptr != nullptr);
  return ptr;
}

void HostPtr::_free(float *ptr) {
  if (!is_null()) {
    std::free(const_cast<float *>(ptr));
  }
}
