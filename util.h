#ifndef UTIL_H
#define UTIL_H

#include <cassert>

#define mat_idx(r, c, s) (r) * (s) + (c)

#define panic_on_cuda_error(err) assert(err == cudaSuccess)

#define relu(x) (x) < 0 ? 0 : (x)

#endif // !UTIL_H
