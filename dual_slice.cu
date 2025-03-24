#include "dual_slice.h"
#include "util.h"

DualSlice::DualSlice(std::size_t count)
    : host_slice(std::make_shared<HostSlice>(count)),
      device_slice(std::make_shared<DeviceSlice>(count)), state(_OPT_NONE) {}

DualSlice::DualSlice(HostSlice host_slice)
    : host_slice(std::make_shared<HostSlice>(std::move(host_slice))),
      device_slice(std::make_shared<DeviceSlice>(host_slice.count())),
      state(_OPT_HOST) {}

DualSlice::DualSlice(DeviceSlice device_slice)
    : host_slice(std::make_shared<HostSlice>(device_slice.count())),
      device_slice(std::make_shared<DeviceSlice>(std::move(device_slice))),
      state(_OPT_DEVICE) {}

DualSlice::DualSlice(HostSlice host_slice, DeviceSlice device_slice,
                     DeviceOpt state)
    : host_slice(std::make_shared<HostSlice>(std::move(host_slice))),
      device_slice(std::make_shared<DeviceSlice>(std::move(device_slice))),
      state(state) {}

void DualSlice::ensure_on_host() {
  switch (state) {
  case _OPT_HOST:
    break;
  case _OPT_DEVICE: {
    auto err =
        cudaMemcpy((void *)host_slice->as_valid_inner(),
                   (const void *)device_slice->as_valid_inner(),
                   host_slice->count() * sizeof(float), cudaMemcpyDeviceToHost);
    panic_on_cuda_error(err);
    state = _OPT_HOST;
    break;
  }
  case _OPT_NONE:
    host_slice->as_valid_inner();
    state = _OPT_HOST;
    break;
  }
}

void DualSlice::ensure_on_device() {
  switch (state) {
  case _OPT_DEVICE:
    break;
  case _OPT_HOST: {
    auto err =
        cudaMemcpy((void *)device_slice->as_valid_inner(),
                   (const void *)host_slice->as_valid_inner(),
                   host_slice->count() * sizeof(float), cudaMemcpyHostToDevice);
    panic_on_cuda_error(err);
    state = _OPT_DEVICE;
    break;
  }
  case _OPT_NONE:
    device_slice->as_valid_inner();
    state = _OPT_DEVICE;
    break;
  }
}

void DualSlice::ensure_on(DeviceStrict device) {
  switch (device) {
  case _STRICT_HOST:
    ensure_on_host();
    break;
  case _STRICT_DEVICE:
    ensure_on_device();
    break;
  }
}

const float *DualSlice::get_host_raw_inner() {
  return host_slice->as_raw_inner();
}

const float *DualSlice::get_device_raw_inner() {
  return device_slice->as_raw_inner();
}

const float *DualSlice::get_raw_inner(DeviceStrict device) {
  switch (device) {
  case _STRICT_HOST:
    return get_host_raw_inner();
  case _STRICT_DEVICE:
    return get_device_raw_inner();
  }
  return nullptr;
}

const float *DualSlice::get_host_valid_inner() {
  ensure_on_host();
  return host_slice->as_raw_inner();
}

const float *DualSlice::get_device_valid_inner() {
  ensure_on_device();
  return device_slice->as_raw_inner();
}

const float *DualSlice::get_valid_inner(DeviceStrict device) {
  switch (device) {
  case _STRICT_HOST:
    return get_host_valid_inner();
  case _STRICT_DEVICE:
    return get_device_valid_inner();
  }
  return nullptr;
}
