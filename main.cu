#include <cassert>
#include <cstdio>
#include <ctime>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>
#include <ostream>

#define DATA float

#define ptr_idx(shape, row, col) ((row) * (shape).cols + (col))

#define DIM2_KERNEL_ASSERT assert(threadIdx.z == 0 && blockIdx.z == 0);
#define DIM2_KERNEL_ROW_COL                                                    \
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;                          \
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

const size_t THREADS_PER_BLOCK_DIM = 16;

typedef enum DataLocation {
  DEVICE,
  HOST,
} DataLocation;

typedef struct Shape {
  size_t rows;
  size_t cols;
} shape;

__global__ void matrix_fill_kernel(DATA *dst, DATA val, Shape shape) {
  DIM2_KERNEL_ASSERT
  DIM2_KERNEL_ROW_COL
  if (row < shape.rows && col < shape.cols) {
    dst[ptr_idx(shape, row, col)] = val;
  }
}

__global__ void matrix_add_from_kernel(DATA *dst, DATA *other, Shape shape) {
  DIM2_KERNEL_ASSERT
  DIM2_KERNEL_ROW_COL
  if (row < shape.rows && col < shape.cols) {
    dst[ptr_idx(shape, row, col)] += other[ptr_idx(shape, row, col)];
  }
}

__global__ void matrix_mul_from_kernel(DATA *dst, DATA *a, DATA *b,
                                       Shape dst_shape, Shape a_shape,
                                       Shape b_shape) {
  DIM2_KERNEL_ASSERT
  DIM2_KERNEL_ROW_COL
  if (row < dst_shape.rows && col < dst_shape.cols) {
    DATA sum = 0;
    for (size_t i = 0; i < a_shape.cols; i++) {
      sum += a[ptr_idx(a_shape, row, i)] * b[ptr_idx(b_shape, i, col)];
    }
    dst[ptr_idx(dst_shape, row, col)] = sum;
  }
}

__global__ void matrix_fill_rand_kernel(DATA *dst, Shape shape, DATA low,
                                        DATA high, unsigned seed) {
  DIM2_KERNEL_ASSERT
  DIM2_KERNEL_ROW_COL
  if (row < shape.rows && col < shape.cols) {
    curandState state;
    curand_init(seed, ptr_idx(shape, row, col), 0, &state);

    dst[ptr_idx(shape, row, col)] = low + (high - low) * curand_uniform(&state);
  }
}

class Matrix {
private:
  Shape shape;
  size_t size;

  DATA *h_elems;
  DATA *d_elems = NULL;

  DataLocation data_location = HOST;

  bool device_is_initialized = false;
  bool host_is_on_stack = false;

  void initialize_device() {
    assert(cudaSuccess == cudaMalloc(&d_elems, size));
    device_is_initialized = true;
  }

public:
  Matrix(size_t rows, size_t cols)
      : shape(Shape{.rows = rows, .cols = cols}),
        size(rows * cols * sizeof(DATA)) {
    h_elems = (DATA *)std::malloc(size);
    assert(h_elems != NULL);
  }

  Matrix(DATA *h_ptr, size_t rows, size_t cols, bool ptr_is_on_stack)
      : shape(Shape{.rows = rows, .cols = cols}),
        size(rows * cols * sizeof(DATA)), h_elems(h_ptr),
        host_is_on_stack(ptr_is_on_stack) {
    assert(h_elems != NULL);
  }

  ~Matrix() {
    if (device_is_initialized) {
      cudaFree(d_elems);
    }
    if (!host_is_on_stack) {
      std::free(h_elems);
    }
  }

  bool is_on_host() { return data_location == HOST; }
  bool is_on_device() { return data_location == DEVICE; }

  void to_host() {
    if (is_on_device()) {
      assert(cudaSuccess ==
             cudaMemcpy(h_elems, d_elems, size, cudaMemcpyDeviceToHost));
      data_location = HOST;
      std::cout << "Matrix moved to Host" << std::endl;
    }
  }

  void to_device() {
    if (!device_is_initialized) {
      initialize_device();
    }
    if (is_on_host()) {
      assert(cudaSuccess ==
             cudaMemcpy(d_elems, h_elems, size, cudaMemcpyHostToDevice));
      data_location = DEVICE;
      std::cout << "Matrix moved to Device" << std::endl;
    }
  }

  DATA host_get_unchecked(size_t row, size_t col) {
    assert(row < shape.rows);
    assert(col < shape.cols);
    assert(data_location == HOST);
    return h_elems[ptr_idx(shape, row, col)];
  }

  void host_set_unchecked(size_t row, size_t col, DATA val) {
    assert(row < shape.rows);
    assert(col < shape.cols);
    assert(data_location == HOST);
    h_elems[ptr_idx(shape, row, col)] = val;
  }

  DATA host_get(size_t row, size_t col) {
    to_host();
    assert(row < shape.rows);
    assert(col < shape.cols);
    return host_get_unchecked(row, col);
  }

  void host_set(size_t row, size_t col, DATA val) {
    to_host();
    assert(row < shape.rows);
    assert(col < shape.cols);
    host_set_unchecked(row, col, val);
  }

  void host_print() {
    to_host();
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "[" << std::endl;
    for (size_t r = 0; r < shape.rows; r++) {
      std::cout << "    ";
      for (size_t c = 0; c < shape.cols; c++) {
        std::cout << std::setw(10) << host_get_unchecked(r, c) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
  }

  Shape get_shape() { return shape; }

  void host_fill(DATA val) {
    to_host();
    for (size_t r = 0; r < shape.rows; r++) {
      for (size_t c = 0; c < shape.cols; c++) {
        host_set_unchecked(r, c, val);
      }
    }
  }

  void device_fill(DATA val) {
    to_device();

    dim3 threads_per_block(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM, 1);
    dim3 blocks_per_grid = dim2_blocks_per_grid(shape.rows, shape.cols);
    matrix_fill_kernel<<<blocks_per_grid, threads_per_block>>>(d_elems, val,
                                                               shape);
    assert(cudaSuccess == cudaDeviceSynchronize());
  }

  void host_add_from(Matrix &other) {
    assert(shape.rows == other.shape.rows);
    assert(shape.cols == other.shape.cols);

    to_host();
    other.to_host();

    for (size_t r = 0; r < shape.rows; r++) {
      for (size_t c = 0; c < shape.cols; c++) {
        DATA self_elem = host_get_unchecked(r, c);
        DATA other_elem = other.host_get_unchecked(r, c);
        host_set_unchecked(r, c, self_elem + other_elem);
      }
    }
  }

  void device_add_from(Matrix &other) {
    assert(shape.rows == other.shape.rows);
    assert(shape.cols == other.shape.cols);

    to_device();
    other.to_device();

    dim3 threads_per_block(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM, 1);
    dim3 blocks_per_grid = dim2_blocks_per_grid(shape.rows, shape.cols);
    matrix_add_from_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_elems, other.d_elems, shape);
    assert(cudaSuccess == cudaDeviceSynchronize());
  }

  static inline dim3 dim2_blocks_per_grid(size_t x, size_t y) {
    return dim3((x + THREADS_PER_BLOCK_DIM - 1) / THREADS_PER_BLOCK_DIM,
                (y + THREADS_PER_BLOCK_DIM - 1) / THREADS_PER_BLOCK_DIM, 1);
  }

  void host_mul_from(Matrix &a, Matrix &b) {
    assert(shape.rows == a.shape.rows);
    assert(shape.cols == b.shape.cols);
    assert(a.shape.cols == b.shape.rows);

    to_host();
    a.to_host();
    b.to_host();

    for (size_t r = 0; r < shape.rows; r++) {
      for (size_t c = 0; c < shape.cols; c++) {
        DATA sum = 0;
        for (size_t i = 0; i < a.shape.cols; i++) {
          sum += a.host_get_unchecked(r, i) * b.host_get_unchecked(i, c);
        }
        host_set_unchecked(r, c, sum);
      }
    }
  }

  void device_mul_from(Matrix &a, Matrix &b) {
    assert(shape.rows == a.shape.rows);
    assert(shape.cols == b.shape.cols);
    assert(a.shape.cols == b.shape.rows);

    to_device();
    a.to_device();
    b.to_device();

    dim3 threads_per_block(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM, 1);
    dim3 blocks_per_grid = dim2_blocks_per_grid(shape.rows, shape.cols);
    matrix_mul_from_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_elems, a.d_elems, b.d_elems, shape, a.shape, b.shape);
    assert(cudaSuccess == cudaDeviceSynchronize());
  }

  void host_fill_rand(DATA low, DATA high) {
    to_host();
    for (size_t r = 0; r < shape.rows; r++) {
      for (size_t c = 0; c < shape.cols; c++) {
        DATA val = low + (high - low) * (DATA)rand() / (DATA)RAND_MAX;
        host_set_unchecked(r, c, val);
      }
    }
  }

  void device_fill_rand(DATA low, DATA high) {
    to_device();

    dim3 threads_per_block(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM, 1);
    dim3 blocks_per_grid = dim2_blocks_per_grid(shape.rows, shape.cols);
    matrix_fill_rand_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_elems, shape, low, high, rand());
    assert(cudaSuccess == cudaDeviceSynchronize());
  }
};

int main(void) {
  srand(time(0));

  Matrix mat1(10, 10000);
  Matrix mat2(10000, 3);

  mat1.device_fill_rand(-1, 1);
  mat2.device_fill_rand(-1, 1);

  // std::cout << "Mat 1:" << std::endl;
  // mat1.host_print();
  // std::cout << "Mat 2:" << std::endl;
  // mat2.host_print();

  Matrix dst(10, 3);

  dst.device_mul_from(mat1, mat2);

  std::cout << "Mat Dst:" << std::endl;
  dst.host_print();

  return 0;
}
