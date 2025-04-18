#include "device_slice.h"

// Move assignment operator
DeviceSlice &DeviceSlice::operator=(DeviceSlice &&other) noexcept {
  if (this != &other) {
    Slice::operator=(std::move(other));
    device_ptr = std::move(other.device_ptr);
  }
  return *this;
}

// Virtual method implementations
const float *DeviceSlice::as_raw_inner() { return device_ptr.as_inner(); }

const float *DeviceSlice::as_valid_inner() {
  if (!is_allocated()) {
    device_ptr.alloc_mut_unchecked(count());
    allocated = true;
  }
  return device_ptr.as_inner();
}
