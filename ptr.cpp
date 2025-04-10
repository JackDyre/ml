#include "ptr.h"
#include <cstddef>

const float *Ptr::as_inner() { return _inner; }

bool Ptr::is_null() { return _inner == nullptr; }

void Ptr::alloc_mut_unchecked(std::size_t size) { _inner = _alloc(size); }

Ptr::Ptr(Ptr &&other) noexcept : _inner(other._inner) {
  other._inner = nullptr;
}

Ptr &Ptr::operator=(Ptr &&other) noexcept {
  if (this->_inner != other._inner) {
    this->_inner = other._inner;
    other._inner = nullptr;
  }
  return *this;
}
