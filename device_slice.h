#ifndef DEVICE_SLICE_H
#define DEVICE_SLICE_H

#include "device_ptr.h"
#include "slice.h"

class DeviceSlice : public Slice {
private:
  DevicePtr device_ptr;

public:
  // Constructors
  DeviceSlice(std::size_t count);
  DeviceSlice(DevicePtr device_ptr, std::size_t count);

  // Move constructor
  DeviceSlice(DeviceSlice &&other) noexcept;
  // Move assignment operator
  DeviceSlice &operator=(DeviceSlice &&other) noexcept;

  // Virtual method implementations
  const float *as_raw_inner() override;
  const float *as_valid_inner() override;
};

#endif // DEVICE_SLICE_H