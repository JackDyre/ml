#ifndef DUAL_SLICE_H
#define DUAL_SLICE_H

#include "device_slice.h"
#include "host_slice.h"
#include "types.h"
#include <cstddef>
#include <memory>

class DualSlice {
private:
  std::shared_ptr<HostSlice> host_slice;
  std::shared_ptr<DeviceSlice> device_slice;

  DeviceOpt state;

public:
  // Regular constructors
  DualSlice(std::size_t count);
  DualSlice(HostSlice host_slice);
  DualSlice(DeviceSlice device_slice);
  DualSlice(HostSlice host_slice, DeviceSlice device_slice, DeviceOpt state);

  // Rule of 5
  // Destructor
  ~DualSlice() = default;

  // Copy constructor
  DualSlice(const DualSlice &other) = default;
  // Copy assignment
  DualSlice &operator=(const DualSlice &other) = default;

  // Move constructor
  DualSlice(DualSlice &&other) noexcept = default;
  // Move assignment
  DualSlice &operator=(DualSlice &&other) noexcept = default;

  void ensure_on_host();
  void ensure_on_device();
  void ensure_on(DeviceStrict device);

  const float *get_host_raw_inner();
  const float *get_device_raw_inner();
  const float *get_raw_inner(DeviceStrict device);
  const float *get_host_valid_inner();
  const float *get_device_valid_inner();
  const float *get_valid_inner(DeviceStrict device);
};

#endif // !DUAL_SLICE_H
