#include "dual_slice.h"

DualSlice::DualSlice(std::size_t size)
    : host_slice(HostSlice(size)), device_slice(DeviceSlice(size)),
      state(_OPT_NONE) {}

DualSlice::DualSlice(HostSlice host_slice)
    : host_slice(std::move(host_slice)),
      device_slice(DeviceSlice(host_slice.size())), state(_OPT_HOST) {}

DualSlice::DualSlice(DeviceSlice device_slice)
    : host_slice(HostSlice(device_slice.size())),
      device_slice(std::move(device_slice)), state(_OPT_DEVICE) {}

DualSlice::DualSlice(HostSlice host_slice, DeviceSlice device_slice,
                     DeviceOpt state)
    : host_slice(std::move(host_slice)), device_slice(std::move(device_slice)),
      state(state) {}

void DualSlice::ensure_on_host() {
  if (state == _OPT_HOST) {
    return;
  }

  if (device_slice.is_allocated()) {
    cudaMemcpy((void *)host_slice.as_valid_inner(),
               (const void *)device_slice.as_valid_inner(),
               host_slice.size() * sizeof(float), cudaMemcpyDeviceToHost);
    state = _OPT_HOST;
  } else {
    host_slice.as_valid_inner();
    return;
  }
}

void DualSlice::ensure_on_device() {
  if (state == _OPT_DEVICE) {
    return;
  }

  if (host_slice.is_allocated()) {
    cudaMemcpy((void *)device_slice.as_valid_inner(),
               (const void *)host_slice.as_valid_inner(),
               host_slice.size() * sizeof(float), cudaMemcpyHostToDevice);
    state = _OPT_DEVICE;
  } else {
    device_slice.as_valid_inner();
    return;
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
  return host_slice.as_raw_inner();
}

const float *DualSlice::get_device_raw_inner() {
  return device_slice.as_raw_inner();
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
  return host_slice.as_raw_inner();
}

const float *DualSlice::get_device_valid_inner() {
  ensure_on_device();
  return device_slice.as_raw_inner();
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
