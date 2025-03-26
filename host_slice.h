#ifndef HOST_SLICE_H
#define HOST_SLICE_H

#include "host_ptr.h"
#include "slice.h"

class HostSlice : public Slice {
private:
  HostPtr host_ptr;

public:
  // Constructors
  HostSlice(std::size_t count) : Slice(count, false), host_ptr() {}
  HostSlice(HostPtr host_ptr, std::size_t count)
      : Slice(count, true), host_ptr(std::move(host_ptr)) {}

  // Move constructor
  HostSlice(HostSlice &&other) noexcept
      : Slice(std::move(other)), host_ptr(std::move(other.host_ptr)) {}
  // Move assignment operator
  HostSlice &operator=(HostSlice &&other) noexcept;

  // Virtual method implementations
  const float *as_raw_inner() override;
  const float *as_valid_inner() override;
};

#endif // HOST_SLICE_H
