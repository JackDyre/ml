#include "ptr.h"
#include <cstddef>

Ptr::Ptr() : _inner(nullptr) {}

Ptr::Ptr(float *ptr) : _inner(ptr) {}

const float *Ptr::as_inner() { return _inner; }

bool Ptr::is_null() { return _inner == nullptr; }

Ptr::~Ptr() { _free(_inner); }

Ptr::Ptr(Ptr &&other) noexcept : _inner(other._inner) {
  other._inner = nullptr;
}

Ptr &Ptr::operator=(Ptr &&other) noexcept {
  if (this->_inner != other._inner) {
    this->_free(this->_inner);
    this->_inner = other._inner;
    other._inner = nullptr;
  }
  return *this;
}

Ptr::Ptr(std::size_t size) : Ptr(Ptr::_alloc(size)) {}
