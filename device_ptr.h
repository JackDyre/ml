#ifndef DEVICE_PTR_H
#define DEVICE_PTR_H

#include "ptr.h"
#include <cstddef>

class DevicePtr : public Ptr {
public:
  DevicePtr();
  explicit DevicePtr(float *ptr);
  DevicePtr(DevicePtr &&other) noexcept;
  DevicePtr &operator=(DevicePtr &&other) noexcept;
  ~DevicePtr() override;

protected:
  void alloc(std::size_t size) override;
  void free() override;
};

#endif // DEVICE_PTR_H
