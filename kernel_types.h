#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include "types.h"
#include <cstddef>

typedef struct MatrixFill {
  float *ptr;
  Shape shape;
  IndexSpec idx_spec;
  float val;
} MatrixFill;

typedef struct MatrixRand {
  float *ptr;
  Shape shape;
  IndexSpec idx_spec;
  int seed;
  float low;
  float high;
} MatrixRand;

typedef struct MatrixAdd {
  float *dst_ptr;
  float *l_ptr;
  float *r_ptr;
  Shape shape;
  IndexSpec dst_idx_spec;
  IndexSpec l_idx_spec;
  IndexSpec r_idx_spec;
} MatrixAdd;

typedef struct MatrixMul {
  float *dst_ptr;
  float *l_ptr;
  float *r_ptr;
  Shape shape;
  std::size_t inner_dim;
  IndexSpec dst_idx_spec;
  IndexSpec l_idx_spec;
  IndexSpec r_idx_spec;
} MatrixMul;

typedef struct MatrixRelu {
  float *src_ptr;
  float *dst_ptr;
  Shape shape;
  IndexSpec src_idx_spec;
  IndexSpec dst_idx_spec;
} MatrixRelu;

typedef struct MatrixSE {
  float *dst_ptr;
  float *a_ptr;
  float *b_ptr;
  Shape shape;
  IndexSpec dst_idx_spec;
  IndexSpec a_idx_spec;
  IndexSpec b_idx_spec;
} MatrixSE;

typedef struct MatrixReluDeriv {
  float *src_ptr;
  float *dst_ptr;
  Shape shape;
  IndexSpec src_idx_spec;
  IndexSpec dst_idx_spec;
} MatrixReluDeriv;

#endif // !KERNEL_TYPES_H
