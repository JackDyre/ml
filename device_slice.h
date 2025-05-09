#ifndef DEVICE_SLICE_H
#define DEVICE_SLICE_H

#include "device_ptr.h"
#include "slice.h"

class DeviceSlice : public Slice {
private:
  DevicePtr device_ptr;

public:
  // Constructors
  DeviceSlice(std::size_t count) : Slice(count, false), device_ptr() {}
  DeviceSlice(DevicePtr device_ptr, std::size_t count)
      : Slice(count, true), device_ptr(std::move(device_ptr)) {}

  // Move constructor
  DeviceSlice(DeviceSlice &&other) noexcept
      : Slice(std::move(other)), device_ptr(std::move(other.device_ptr)) {}
  // Move assignment operator
  DeviceSlice &operator=(DeviceSlice &&other) noexcept;

  // Virtual method implementations
  const float *as_raw_inner() override;
  const float *as_valid_inner() override;
};

#endif // DEVICE_SLICE_H