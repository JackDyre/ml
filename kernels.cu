#ifndef KERNELS_CU
#define KERNELS_CU

#include <cassert>
#include <curand_kernel.h>

#define ptr_idx(stride, row, col) ((row) * stride + (col))

template <typename T>
__global__ void matrix_fill_kernel(T *dst, T val, size_t rows, size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    dst[ptr_idx(cols, row, col)] = val;
  }
}

template <typename T>
__global__ void matrix_fill_rand_kernel(T *dst, size_t rows, size_t cols, T low,
                                        T high, unsigned seed) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    curandState state;
    curand_init(seed, ptr_idx(cols, row, col), 0, &state);
    dst[ptr_idx(cols, row, col)] = low + (high - low) * curand_uniform(&state);
  }
}

template <typename T>
__global__ void matrix_add_kernel(T *dst, T *other, size_t rows, size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    dst[ptr_idx(cols, row, col)] += other[ptr_idx(cols, row, col)];
  }
}

template <typename T>
__global__ void matrix_mul_kernel(T *dst, T *a, T *b, size_t rows, size_t cols,
                                  size_t inner_size) {

  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    T sum = 0;
    for (size_t i = 0; i < inner_size; i++) {
      sum += a[ptr_idx(inner_size, row, i)] * b[ptr_idx(cols, i, col)];
    }
    dst[ptr_idx(cols, row, col)] = sum;
  }
}

#endif // KERNELS_CU
