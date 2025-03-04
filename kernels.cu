#include "kernels.h"
#include <cassert>
#include <curand_kernel.h>
#include <cub/device/device_reduce.cuh>

#define ptr_idx(stride, row, col) ((row) * stride + (col))

const dim3 THREADS_PER_BLOCK = dim3(16, 16, 1);
dim3 blocks_per_grid(dim3 inp_dim) {
  return dim3((inp_dim.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x,
              (inp_dim.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y,
              (inp_dim.z + THREADS_PER_BLOCK.z - 1) / THREADS_PER_BLOCK.z);
}

__global__ void _matrix_fill_kernel(float *dst, float val, size_t rows,
                                    size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    dst[ptr_idx(cols, row, col)] = val;
  }
}

void launch_matrix_fill_kernel(float *dst, float val, size_t rows,
                               size_t cols) {
  _matrix_fill_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                        THREADS_PER_BLOCK>>>(dst, val, rows, cols);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

__global__ void _matrix_fill_rand_kernel(float *dst, size_t rows, size_t cols,
                                         float low, float high, unsigned seed) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    curandState state;
    curand_init(seed, ptr_idx(cols, row, col), 0, &state);
    dst[ptr_idx(cols, row, col)] = low + (high - low) * curand_uniform(&state);
  }
}

void launch_matrix_fill_rand_kernel(float *dst, size_t rows, size_t cols,
                                    float low, float high, unsigned seed) {
  _matrix_fill_rand_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                             THREADS_PER_BLOCK>>>(dst, rows, cols, low, high,
                                                  seed);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

__global__ void _matrix_add_kernel(float *dst, float *other, size_t rows,
                                   size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    dst[ptr_idx(cols, row, col)] += other[ptr_idx(cols, row, col)];
  }
}

void launch_matrix_add_kernel(float *dst, float *other, size_t rows,
                              size_t cols) {
  _matrix_add_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                       THREADS_PER_BLOCK>>>(dst, other, rows, cols);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

__global__ void _matrix_mul_kernel(float *dst, float *a, float *b, size_t rows,
                                   size_t cols, size_t inner_size) {

  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    float sum = 0;
    for (size_t i = 0; i < inner_size; i++) {
      sum += a[ptr_idx(inner_size, row, i)] * b[ptr_idx(cols, i, col)];
    }
    dst[ptr_idx(cols, row, col)] = sum;
  }
}

void launch_matrix_mul_kernel(float *dst, float *a, float *b, size_t rows,
                              size_t cols, size_t inner_size) {
  _matrix_mul_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                       THREADS_PER_BLOCK>>>(dst, a, b, rows, cols, inner_size);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

__global__ void _matrix_mse_kernel(float *dst, float *output, float *target,
                                   size_t rows, size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    size_t idx = ptr_idx(cols, row, col);
    float diff = target[idx] - output[idx];
    dst[idx] = diff * diff;
  }
}

void launch_matrix_mse_kernel(float *dst, float *output, float *target,
                              size_t rows, size_t cols) {
  _matrix_mse_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                       THREADS_PER_BLOCK>>>(dst, output, target, rows, cols);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

__global__ void _matrix_relu_kernel(float *dst, size_t rows, size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    size_t idx = ptr_idx(cols, row, col);
    dst[idx] = relu(dst[idx]);
  }
}

void launch_matrix_relu_kernel(float *dst, size_t rows, size_t cols) {
  _matrix_relu_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                        THREADS_PER_BLOCK>>>(dst, rows, cols);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

__global__ void _matrix_gradient_step_kernel(float *param, float *grad,
                                             float lr, size_t rows,
                                             size_t cols) {
  assert(threadIdx.z == 0 && blockIdx.z == 0);

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    size_t idx = ptr_idx(cols, row, col);
    param[idx] -= lr * grad[idx];
  }
}

void launch_matrix_gradient_step_kernel(float *param, float *grad, float lr,
                                        size_t rows, size_t cols) {
  _matrix_gradient_step_kernel<<<blocks_per_grid(dim3(rows, cols, 1)),
                                 THREADS_PER_BLOCK>>>(param, grad, lr, rows,
                                                      cols);
  assert(cudaSuccess == cudaDeviceSynchronize());
}

float calculate_dev_mse_cost(float *output_ptr, float *target_ptr, size_t num_elems) {
  float *dst;
  cudaMalloc(&dst, num_elems * sizeof(float));

  launch_matrix_mse_kernel(dst, output_ptr, target_ptr, num_elems, 1);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  float *d_out;
  cudaMalloc(&d_out, sizeof(float));

  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dst, d_out, num_elems);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dst, d_out, num_elems);

  float cost;
  cudaMemcpy(&cost, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dst);
  cudaFree(d_temp_storage);
  cudaFree(d_out);

  return cost / num_elems;
}
