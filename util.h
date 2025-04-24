#ifndef UTIL_H
#define UTIL_H

#include <cassert>

#define mat_idx(r, c, s) (r) * (s) + (c)
#define mat_idx_spec(r, c, s) (r) * (s.stride) + (c) + (s.ptr_offset)

#define panic_on_cuda_error(err) assert(err == cudaSuccess)

// Original ReLU implementations
#define relu(x) (x) <= 0 ? 0 : (x)
#define relu_deriv(x) (x) <= 0 ? 0 : 1

// Sigmoid implementations with relu names for compatibility
// #define relu(x) 1.0f / (1.0f + expf(-(x)))
// #define relu_deriv(x) relu(x) * (1.0f - relu(x))

#endif // !UTIL_H
