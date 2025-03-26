#include "host_slice.h"


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
  if (!is_allocated()) {
    host_ptr.alloc_mut_unchecked(count());
    allocated = true;
  }
  return host_ptr.as_inner();
}
