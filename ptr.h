#ifndef LAZY_PTR_H
#define LAZY_PTR_H

#include <cstddef>
#include <utility>

class Ptr {
private:
  float *_inner;

public:
  Ptr();
  explicit Ptr(float *ptr);
  const float *as_inner();
  bool is_null();
  virtual ~Ptr();

  // Delete copy operations
  Ptr(const Ptr &) = delete;
  Ptr &operator=(const Ptr &) = delete;

  // Move constructor
  Ptr(Ptr &&other) noexcept;

  // Move assignment operator
  Ptr &operator=(Ptr &&other) noexcept;

protected:
  virtual void alloc(std::size_t size);
  virtual void free();
};

#endif // LAZY_PTR_H
