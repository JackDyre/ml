#include "slice.h"

Slice::Slice(std::size_t size, bool allocated)
    : _size(size), allocated(allocated) {}

// Move constructor
Slice::Slice(Slice &&other) noexcept
    : _size(other._size), allocated(other.allocated) {
  other._size = 0;
  other.allocated = false;
}

// Move assignment operator
Slice &Slice::operator=(Slice &&other) noexcept {
  if (this != &other) {
    _size = other._size;
    allocated = other.allocated;

    other._size = 0;
    other.allocated = false;
  }
  return *this;
}

bool Slice::is_allocated() { return allocated; }

std::size_t Slice::size() { return _size; }
