#include "device_slice.h"

// Constructors
DeviceSlice::DeviceSlice(std::size_t size)
    : Slice(size, false), device_ptr() {}

DeviceSlice::DeviceSlice(DevicePtr device_ptr, std::size_t size)
    : Slice(size, true), device_ptr(std::move(device_ptr)) {}

// Move constructor
DeviceSlice::DeviceSlice(DeviceSlice &&other) noexcept
    : Slice(std::move(other)), device_ptr(std::move(other.device_ptr)) {}

// Move assignment operator
DeviceSlice &DeviceSlice::operator=(DeviceSlice &&other) noexcept {
  if (this != &other) {
    Slice::operator=(std::move(other));
    device_ptr = std::move(other.device_ptr);
  }
  return *this;
}

// Virtual method implementations
const float *DeviceSlice::as_raw_inner() {
  return device_ptr.as_inner();
}

const float *DeviceSlice::as_valid_inner() {
  if (is_allocated()) {
    device_ptr.alloc_mut_unchecked(size());
  }
  return device_ptr.as_inner();
}