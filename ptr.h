#ifndef LAZY_PTR_H
#define LAZY_PTR_H

#include <cstddef>
#include <utility>

class Ptr {
private:
  float *_inner;

public:
  Ptr() : _inner(nullptr) {}
  explicit Ptr(float *ptr) : _inner(ptr) {}

  const float *as_inner();
  bool is_null();
  void alloc_mut_unchecked(std::size_t size);

  virtual ~Ptr() = default;

  // Delete copy operations
  Ptr(const Ptr &) = delete;
  Ptr &operator=(const Ptr &) = delete;

  // Move constructor
  Ptr(Ptr &&other) noexcept;

  // Move assignment operator
  Ptr &operator=(Ptr &&other) noexcept;

public:
  // Immutably allocate a ptr the provided size.
  virtual float *_alloc(std::size_t size) = 0;
  // Immutably free the provided ptr.
  virtual void _free(float *ptr) = 0;
};

#endif // LAZY_PTR_H
