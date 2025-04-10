#include "slice.h"

// Move constructor
Slice::Slice(Slice &&other) noexcept
    : _count(other._count), allocated(other.allocated) {
  other._count = 0;
  other.allocated = false;
}

// Move assignment operator
Slice &Slice::operator=(Slice &&other) noexcept {
  if (this != &other) {
    _count = other._count;
    allocated = other.allocated;

    other._count = 0;
    other.allocated = false;
  }
  return *this;
}

bool Slice::is_allocated() { return allocated; }

std::size_t Slice::count() { return _count; }
