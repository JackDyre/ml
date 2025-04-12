#include "kernel_types.h"
#include "kernels.h"
#include "util.h"
#include <cassert>
#include <cstddef>
#include <curand_kernel.h>

#define thread_block_idx(dim) blockDim.dim *blockIdx.dim + threadIdx.dim

const auto THREADS_PER_BLOCK = dim3(16, 16, 16);

#define block_shape_dim(launch_shape, dim)                                     \
  (launch_shape.dim + THREADS_PER_BLOCK.dim - 1) / THREADS_PER_BLOCK.dim

#define kernel_config(x, y, z) THREADS_PER_BLOCK, blocks_per_grid(dim3(x, y, z))

dim3 blocks_per_grid(dim3 launch_shape) {
  return dim3(block_shape_dim(launch_shape, x),
              block_shape_dim(launch_shape, y),
              block_shape_dim(launch_shape, z));
}

__global__ void kernel_matrix_fill(MatrixFill args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.shape.rows || c >= args.shape.cols || z != 0) {
    return;
  }

  args.ptr[mat_idx_spec(r, c, args.idx_spec)] = args.val;
}

void device_matrix_fill(MatrixFill args) {
  auto l_rows = args.shape.rows;
  auto l_cols = args.shape.cols;
  kernel_matrix_fill<<<kernel_config(l_rows, l_cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_rand(MatrixRand args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.shape.rows || c >= args.shape.cols || z != 0) {
    return;
  }

  auto idx = mat_idx_spec(r, c, args.idx_spec);

  curandState state;
  curand_init(args.seed, idx, 0, &state);

  args.ptr[idx] = args.low + (args.high - args.low) * curand_uniform(&state);
}

void device_matrix_rand(MatrixRand args) {
  auto l_rows = args.shape.rows;
  auto l_cols = args.shape.cols;
  kernel_matrix_rand<<<kernel_config(l_rows, l_cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_add(MatrixAdd args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.shape.rows || c >= args.shape.cols || z != 0) {
    return;
  }

  auto l_val = args.l_ptr[mat_idx_spec(r, c, args.l_idx_spec)];
  auto r_val = args.r_ptr[mat_idx_spec(r, c, args.r_idx_spec)];

  args.dst_ptr[mat_idx_spec(r, c, args.dst_idx_spec)] = l_val + r_val;
}

void device_matrix_add(MatrixAdd args) {
  auto l_rows = args.shape.rows;
  auto l_cols = args.shape.cols;
  kernel_matrix_add<<<kernel_config(l_rows, l_cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_mul(MatrixMul args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.shape.rows || c >= args.shape.cols || z != 0) {
    return;
  }

  float val = 0.0f;
  for (std::size_t i = 0; i < args.inner_dim; i++) {
    float left_val = args.l_ptr[mat_idx_spec(r, i, args.l_idx_spec)];
    float right_val = args.r_ptr[mat_idx_spec(i, c, args.r_idx_spec)];
    val += left_val * right_val;
  }
  args.dst_ptr[mat_idx_spec(r, c, args.dst_idx_spec)] = val;
}

void device_matrix_mul(MatrixMul args) {
  auto l_rows = args.shape.rows;
  auto l_cols = args.shape.cols;
  kernel_matrix_mul<<<kernel_config(l_rows, l_cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_relu(MatrixRelu args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.shape.rows || c >= args.shape.cols || z != 0) {
    return;
  }

  args.dst_ptr[mat_idx_spec(r, c, args.dst_idx_spec)] =
      relu(args.src_ptr[mat_idx_spec(r, c, args.src_idx_spec)]);
}

void device_matrix_relu(MatrixRelu args) {
  auto l_rows = args.shape.rows;
  auto l_cols = args.shape.cols;
  kernel_matrix_relu<<<kernel_config(l_rows, l_cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_se(MatrixSE args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.shape.rows || c >= args.shape.cols || z != 0) {
    return;
  }

  auto diff = args.a_ptr[mat_idx_spec(r, c, args.a_idx_spec)] -
              args.b_ptr[mat_idx_spec(r, c, args.b_idx_spec)];

  args.dst_ptr[mat_idx_spec(r, c, args.dst_idx_spec)] = diff * diff;
}

void device_matrix_se(MatrixSE args) {
  auto l_rows = args.shape.rows;
  auto l_cols = args.shape.cols;
  kernel_matrix_se<<<kernel_config(l_rows, l_cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_relu_deriv(MatrixReluDeriv args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.shape.rows || c >= args.shape.cols || z != 0) {
    return;
  }

  args.dst_ptr[mat_idx_spec(r, c, args.dst_idx_spec)] =
      relu_deriv(args.src_ptr[mat_idx_spec(r, c, args.src_idx_spec)]);
}

void device_matrix_relu_deriv(MatrixReluDeriv args) {
  auto l_rows = args.shape.rows;
  auto l_cols = args.shape.cols;
  kernel_matrix_relu_deriv<<<kernel_config(l_rows, l_cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}
