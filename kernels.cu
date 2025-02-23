#include <cassert>
#include <curand_kernel.h>

#ifndef KERNELS_CU
#define KERNELS_CU

#define ptr_idx(stride, row, col) ((row) * stride + (col))

const dim3 THREADS_PER_BLOCK = dim3(16, 16, 1);
dim3 blocks_per_grid(dim3 inp_dim) {
  return dim3((inp_dim.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x,
              (inp_dim.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y,
              (inp_dim.z + THREADS_PER_BLOCK.z - 1) / THREADS_PER_BLOCK.z);
}

template <typename T>
__global__ void _matrix_fill_kernel(T *dst, T val, size_t rows, size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    dst[ptr_idx(cols, row, col)] = val;
  }
}

template <typename T>
void launch_matrix_fill_kernel(T *dst, T val, size_t rows, size_t cols) {
  _matrix_fill_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                        THREADS_PER_BLOCK>>>(dst, val, rows, cols);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

template <typename T>
__global__ void _matrix_fill_rand_kernel(T *dst, size_t rows, size_t cols,
                                         T low, T high, unsigned seed) {
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
void launch_matrix_fill_rand_kernel(T *dst, size_t rows, size_t cols, T low,
                                    T high, unsigned seed) {
  _matrix_fill_rand_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                             THREADS_PER_BLOCK>>>(dst, rows, cols, low, high,
                                                  seed);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

template <typename T>
__global__ void _matrix_add_kernel(T *dst, T *other, size_t rows, size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    dst[ptr_idx(cols, row, col)] += other[ptr_idx(cols, row, col)];
  }
}

template <typename T>
void launch_matrix_add_kernel(T *dst, T *other, size_t rows, size_t cols) {
  _matrix_add_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                       THREADS_PER_BLOCK>>>(dst, other, rows, cols);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

template <typename T>
__global__ void _matrix_mul_kernel(T *dst, T *a, T *b, size_t rows, size_t cols,
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

template <typename T>
void launch_matrix_mul_kernel(T *dst, T *a, T *b, size_t rows, size_t cols,
                              size_t inner_size) {
  _matrix_mul_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                       THREADS_PER_BLOCK>>>(dst, a, b, rows, cols, inner_size);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

#endif // KERNELS_CU
