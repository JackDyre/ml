#ifndef DEVICE_PTR_H
#define DEVICE_PTR_H

#include "ptr.h"
#include <cstddef>

class DevicePtr : public Ptr {
public:
  DevicePtr() : Ptr() {}
  explicit DevicePtr(float *ptr) : Ptr(ptr) {}
  DevicePtr(DevicePtr &&other) noexcept : Ptr(std::move(other)) {}
  DevicePtr &operator=(DevicePtr &&other) noexcept;
  ~DevicePtr() override;

public:
  float *_alloc(std::size_t size) override;
  void _free(float *ptr) override;
};

#endif // DEVICE_PTR_H
