#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include <cstddef>
typedef struct MatrixFill {
  float *d_ptr;

  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  float val;
} MatrixFill;

#endif // !KERNEL_TYPES_H
