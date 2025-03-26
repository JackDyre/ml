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
  DualSlice(std::size_t count)
      : host_slice(std::make_shared<HostSlice>(count)),
        device_slice(std::make_shared<DeviceSlice>(count)), state(_OPT_NONE) {}
  DualSlice(HostSlice host_slice)
      : host_slice(std::make_shared<HostSlice>(std::move(host_slice))),
        device_slice(std::make_shared<DeviceSlice>(host_slice.count())),
        state(_OPT_HOST) {}
  DualSlice(DeviceSlice device_slice)
      : host_slice(std::make_shared<HostSlice>(device_slice.count())),
        device_slice(std::make_shared<DeviceSlice>(std::move(device_slice))),
        state(_OPT_DEVICE) {}
  DualSlice(HostSlice host_slice, DeviceSlice device_slice, DeviceOpt state)
      : host_slice(std::make_shared<HostSlice>(std::move(host_slice))),
        device_slice(std::make_shared<DeviceSlice>(std::move(device_slice))),
        state(state) {}

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
