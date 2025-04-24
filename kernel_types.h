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

typedef struct MatrixElemWiseMul {
  float *dst_ptr;
  float *l_ptr;
  float *r_ptr;
  Shape shape;
  IndexSpec dst_idx_spec;
  IndexSpec l_idx_spec;
  IndexSpec r_idx_spec;
} MatrixElemWiseMul;

typedef struct MatrixActGrad {
  float *dst_ptr;
  float *next_grad_b_ptr;
  float *next_w_ptr;
  Shape shape;
  size_t k_max;
  IndexSpec dst_idx_spec;
  IndexSpec next_grad_b_idx_spec;
  IndexSpec next_w_idx_spec;
} MatrixActGrad;

typedef struct MatrixSEDeriv {
  float *dst_ptr;
  float *a_ptr;
  float *b_ptr;
  Shape shape;
  IndexSpec dst_idx_spec;
  IndexSpec a_idx_spec;
  IndexSpec b_idx_spec;
} MatrixSEDeriv;

typedef struct MatrixWeightGrad {
  float *dst_ptr;
  float *b_grad_ptr;
  float *a_prev_ptr;
  Shape shape;
  IndexSpec dst_idx_spec;
  IndexSpec b_grad_idx_spec;
  IndexSpec a_prev_idx_spec;
} MatrixWeightGrad;

#endif // !KERNEL_TYPES_H
