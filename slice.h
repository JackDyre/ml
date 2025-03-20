#ifndef SLICE_H
#define SLICE_H

#include <cstddef>

class Slice {
protected:
  std::size_t _size;
  bool allocated = false;

  explicit Slice(std::size_t size, bool allocated);

public:
  // Destructor
  virtual ~Slice() = default;

  // Copy constructor
  Slice(const Slice &other) = delete;
  // Copy assignment operator
  Slice &operator=(const Slice &other) = delete;

  // Move constructor
  Slice(Slice &&other) noexcept;
  // Move assignment operator
  Slice &operator=(Slice &&other) noexcept;

  virtual const float *as_raw_inner() = 0;
  virtual const float *as_valid_inner() = 0;
  bool is_allocated();
  std::size_t size();
};

#endif // !SLICE_H
