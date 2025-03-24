#include "matrix.h"
#include "kernel_types.h"
#include "kernels.h"
#include "util.h"
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>

Matrix::Matrix(Shape shape)
    : slice(DualSlice(shape.rows * shape.cols)), shape(shape),
      stride(shape.cols) {}

std::size_t Matrix::elem_count() { return row_count() * col_count(); }

std::size_t Matrix::row_count() { return shape.rows; }

std::size_t Matrix::col_count() { return shape.cols; }

std::size_t Matrix::get_stride() { return stride; }

void Matrix::print_h() {
  const float *ptr = slice.get_host_valid_inner();

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "[" << std::endl;
  for (size_t r = 0; r < row_count(); r++) {
    std::cout << "\t";
    for (size_t c = 0; c < col_count(); c++) {
      std::cout << std::setw(10) << ptr[mat_idx(r, c, stride)] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
}

void Matrix::fill_d(float val) {
  float *ptr = (float *)slice.get_device_valid_inner();

  auto launch_args = MatrixFill{.d_ptr = ptr,
                                .rows = shape.rows,
                                .cols = shape.cols,
                                .stride = stride,
                                .val = val};

  device_matrix_fill(launch_args);
}

void Matrix::rand_d(float low, float high) {
  float *ptr = (float *)slice.get_device_valid_inner();

  auto launch_args = MatrixRand{.d_ptr = ptr,
                                .rows = shape.rows,
                                .cols = shape.cols,
                                .stride = stride,
                                .seed = rand(),
                                .low = low,
                                .high = high};

  device_matrix_rand(launch_args);
}
