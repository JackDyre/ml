#ifndef UTIL_H
#define UTIL_H

#include <cassert>

#define mat_idx(r, c, s) (r) * (s) + (c)
#define mat_idx_spec(r, c, s) (r) * (s.stride) + (c) + (s.ptr_offset)

#define panic_on_cuda_error(err) assert(err == cudaSuccess)

#define relu(x) (x) <= 0 ? 0 : (x)
#define relu_deriv(x) (x) <= 0 ? 0 : 1

#endif // !UTIL_H
