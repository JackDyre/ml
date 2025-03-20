#include "slice.h"

Slice::Slice(std::size_t size, bool allocated)
    : size(size), allocated(allocated) {}

// Move constructor
Slice::Slice(Slice &&other) noexcept
    : size(other.size), allocated(other.allocated) {
  other.size = 0;
  other.allocated = false;
}

// Move assignment operator
Slice &Slice::operator=(Slice &&other) noexcept {
  if (this != &other) {
    size = other.size;
    allocated = other.allocated;

    other.size = 0;
    other.allocated = false;
  }
  return *this;
}

bool Slice::is_allocated() { return allocated; }

std::size_t Slice::get_size() { return size; }
