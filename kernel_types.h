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

typedef struct MatrixRand {
  float *d_ptr;

  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  int seed;

  float low;
  float high;
} MatrixRand;

#endif // !KERNEL_TYPES_H
