#ifndef DUAL_SLICE_H
#define DUAL_SLICE_H

#include "device_slice.h"
#include "host_slice.h"
#include "types.h"
#include <cstddef>

class DualSlice {
private:
  HostSlice host_slice;
  DeviceSlice device_slice;

  DeviceOpt state;

public:
  DualSlice(std::size_t size);
  DualSlice(HostSlice host_slice);
  DualSlice(DeviceSlice device_slice);
  DualSlice(HostSlice host_slice, DeviceSlice device_slice, DeviceOpt state);

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
