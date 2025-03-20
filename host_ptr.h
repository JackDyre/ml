#ifndef HOST_PTR_H
#define HOST_PTR_H

#include "ptr.h"
#include <cstddef>
#include <cstdlib>

class HostPtr : public Ptr {
public:
  HostPtr();
  explicit HostPtr(float *ptr);
  HostPtr(HostPtr &&other) noexcept;
  HostPtr &operator=(HostPtr &&other) noexcept;
  ~HostPtr() override;

public:
  float *_alloc(std::size_t size) override;
  void _free(float *ptr) override;
};

#endif // HOST_PTR_H
