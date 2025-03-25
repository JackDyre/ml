#include "kernels.h"
#include "util.h"
#include <cassert>
#include <cstddef>
#include <curand_kernel.h>

#define thread_block_idx(dim) blockDim.dim *blockIdx.dim + threadIdx.dim
#define panic_on_cuda_error(err) assert(err == cudaSuccess)

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

  if (r >= args.rows || c >= args.cols || z != 0) {
    return;
  }

  args.ptr[mat_idx(r, c, args.stride)] = args.val;
}

void device_matrix_fill(MatrixFill args) {
  kernel_matrix_fill<<<kernel_config(args.rows, args.cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_rand(MatrixRand args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.rows || c >= args.cols || z != 0) {
    return;
  }

  curandState state;
  curand_init(args.seed, mat_idx(r, c, args.stride), 0, &state);

  args.ptr[mat_idx(r, c, args.stride)] =
      args.low + (args.high - args.low) * curand_uniform(&state);
}

void device_matrix_rand(MatrixRand args) {
  kernel_matrix_rand<<<kernel_config(args.rows, args.cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_add(MatrixAdd args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.rows || c >= args.cols || z != 0) {
    return;
  }

  auto other_val = args.other_ptr[mat_idx(r, c, args.other_stride)];

  args.dst_ptr[mat_idx(r, c, args.dst_stride)] += other_val;
}

void device_matrix_add(MatrixAdd args) {
  kernel_matrix_add<<<kernel_config(args.rows, args.cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}

__global__ void kernel_matrix_mul(MatrixMul args) {
  std::size_t r = thread_block_idx(x);
  std::size_t c = thread_block_idx(y);
  std::size_t z = thread_block_idx(z);

  if (r >= args.dst_rows || c >= args.dst_cols || z != 0) {
    return;
  }

  float val = 0.0f;
  for (std::size_t i = 0; i < args.inner_dim; i++) {
    float left_val = args.l_ptr[mat_idx(r, i, args.l_stride)];
    float right_val = args.r_ptr[mat_idx(i, c, args.r_stride)];
    val += left_val * right_val;
  }
  args.dst_ptr[mat_idx(r, c, args.dst_stride)] = val;
}

void device_matrix_mul(MatrixMul args) {
  kernel_matrix_mul<<<kernel_config(args.dst_rows, args.dst_cols, 1)>>>(args);
  auto err = cudaDeviceSynchronize();
  panic_on_cuda_error(err);
}
