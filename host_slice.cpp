#include "host_slice.h"

// Constructors
HostSlice::HostSlice(std::size_t size) : Slice(size, false), host_ptr() {}

HostSlice::HostSlice(HostPtr host_ptr, std::size_t size)
    : Slice(size, true), host_ptr(std::move(host_ptr)) {}

// Move constructor
HostSlice::HostSlice(HostSlice &&other) noexcept
    : Slice(std::move(other)), host_ptr(std::move(other.host_ptr)) {}

// Move assignment operator
HostSlice &HostSlice::operator=(HostSlice &&other) noexcept {
  if (this != &other) {
    Slice::operator=(std::move(other));
    host_ptr = std::move(other.host_ptr);
  }
  return *this;
}

// Virtual method implementations
const float *HostSlice::as_raw_inner() { return host_ptr.as_inner(); }

const float *HostSlice::as_valid_inner() {
  if (is_allocated()) {
    host_ptr.alloc_mut_unchecked(get_size());
  }
  return host_ptr.as_inner();
}
