#include "ptr.h"

Ptr::Ptr() : _inner(nullptr) {}

Ptr::Ptr(float *ptr) : _inner(ptr) {}

const float *Ptr::as_inner() { return _inner; }

bool Ptr::is_null() { return _inner == nullptr; }

Ptr::~Ptr() { free(); }

Ptr::Ptr(Ptr &&other) noexcept : _inner(other._inner) {
  other._inner = nullptr;
}

Ptr &Ptr::operator=(Ptr &&other) noexcept {
  if (this != &other) {
    free();
    _inner = other._inner;
    other._inner = nullptr;
  }
  return *this;
}
