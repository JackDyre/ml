#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include <cstddef>
typedef struct MatrixFill {
  float *ptr;

  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  float val;
} MatrixFill;

typedef struct MatrixRand {
  float *ptr;

  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  int seed;

  float low;
  float high;
} MatrixRand;

typedef struct MatrixAdd {
  float *dst_ptr;
  float *other_ptr;

  std::size_t rows;
  std::size_t cols;

  std::size_t dst_stride;
  std::size_t other_stride;
} MatrixAdd;

typedef struct MatrixMul {
  float *dst_ptr;
  float *l_ptr;
  float *r_ptr;

  std::size_t dst_rows;
  std::size_t dst_cols;
  std::size_t inner_dim;

  std::size_t dst_stride;
  std::size_t l_stride;
  std::size_t r_stride;
} MatrixMul;

#endif // !KERNEL_TYPES_H
