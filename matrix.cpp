#include "matrix.h"
#include "kernel_types.h"
#include "kernels.h"
#include "types.h"
#include "util.h"
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>

std::size_t Matrix::elem_count() { return row_count() * col_count(); }

std::size_t Matrix::row_count() { return shape.rows; }

std::size_t Matrix::col_count() { return shape.cols; }

void Matrix::print_h() {
  const float *ptr = slice.get_host_valid_inner();

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "[" << std::endl;
  for (size_t r = 0; r < row_count(); r++) {
    std::cout << "\t";
    for (size_t c = 0; c < col_count(); c++) {
      auto elem = ptr[mat_idx_spec(r, c, idx_spec)];
      std::cout << std::setw(10) << elem << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
}

void Matrix::fill_d(float val) {
  float *ptr = (float *)slice.get_device_valid_inner();

  auto launch_args = MatrixFill{
      .ptr = ptr,
      .shape = shape,
      .idx_spec = idx_spec,
      .val = val,
  };

  device_matrix_fill(launch_args);
}

void Matrix::rand_d(float low, float high) {
  float *ptr = (float *)slice.get_device_valid_inner();

  auto launch_args = MatrixRand{
      .ptr = ptr,
      .shape = shape,
      .idx_spec = idx_spec,
      .seed = rand(),
      .low = low,
      .high = high,
  };

  device_matrix_rand(launch_args);
}

void Matrix::add_d(Matrix &l, Matrix &r) {
  assert(shape.rows == l.shape.rows);
  assert(shape.cols == l.shape.cols);
  assert(r.shape.rows == l.shape.rows);
  assert(r.shape.cols == l.shape.cols);

  float *dst_ptr = (float *)slice.get_device_valid_inner();
  auto dst_idx_spec = idx_spec;

  float *l_ptr = (float *)l.slice.get_device_valid_inner();
  auto l_idx_spec = l.idx_spec;

  float *r_ptr = (float *)r.slice.get_device_valid_inner();
  auto r_idx_spec = r.idx_spec;

  auto launch_args = MatrixAdd{
      .dst_ptr = dst_ptr,
      .l_ptr = l_ptr,
      .r_ptr = r_ptr,
      .shape = shape,
      .dst_idx_spec = dst_idx_spec,
      .l_idx_spec = l_idx_spec,
      .r_idx_spec = r_idx_spec,
  };

  device_matrix_add(launch_args);
}

void Matrix::mul_d(Matrix &l, Matrix &r) {
  assert(shape.rows == l.shape.rows);
  assert(shape.cols == r.shape.cols);
  assert(l.shape.cols == r.shape.rows);

  float *dst_ptr = (float *)slice.get_device_valid_inner();
  auto dst_idx_spec = idx_spec;

  float *l_ptr = (float *)l.slice.get_device_valid_inner();
  auto l_idx_spec = l.idx_spec;

  float *r_ptr = (float *)r.slice.get_device_valid_inner();
  auto r_idx_spec = r.idx_spec;

  auto launch_args = MatrixMul{
      .dst_ptr = dst_ptr,
      .l_ptr = l_ptr,
      .r_ptr = r_ptr,
      .shape = shape,
      .inner_dim = l.shape.cols,
      .dst_idx_spec = dst_idx_spec,
      .l_idx_spec = l_idx_spec,
      .r_idx_spec = r_idx_spec,
  };

  device_matrix_mul(launch_args);
}

void Matrix::relu_d(Matrix &src) {
  assert(shape.rows == src.shape.rows);
  assert(shape.cols == src.shape.cols);

  float *src_ptr = (float *)src.slice.get_device_valid_inner();
  auto src_idx_spec = src.idx_spec;

  float *dst_ptr = (float *)slice.get_device_valid_inner();
  auto dst_idx_spec = idx_spec;

  auto launch_args = MatrixRelu{
      .src_ptr = src_ptr,
      .dst_ptr = dst_ptr,
      .shape = shape,
      .src_idx_spec = src_idx_spec,
      .dst_idx_spec = dst_idx_spec,
  };

  device_matrix_relu(launch_args);
}

void Matrix::se_d(Matrix &a, Matrix &b) {
  assert(a.shape.rows == b.shape.rows);
  assert(a.shape.cols == b.shape.cols);
  assert(shape.rows == a.shape.rows);
  assert(shape.cols == a.shape.cols);

  float *dst_ptr = (float *)slice.get_device_valid_inner();
  auto dst_idx_spec = idx_spec;

  float *a_ptr = (float *)a.slice.get_device_valid_inner();
  auto a_idx_spec = a.idx_spec;

  float *b_ptr = (float *)b.slice.get_device_valid_inner();
  auto b_idx_spec = b.idx_spec;

  auto launch_args = MatrixSE{
      .dst_ptr = dst_ptr,
      .a_ptr = a_ptr,
      .b_ptr = b_ptr,
      .shape = shape,
      .dst_idx_spec = dst_idx_spec,
      .a_idx_spec = a_idx_spec,
      .b_idx_spec = b_idx_spec,
  };

  device_matrix_se(launch_args);
}

void Matrix::relu_deriv_d(Matrix &src) {
  float *src_ptr = (float *)src.slice.get_device_valid_inner();
  auto src_idx_spec = src.idx_spec;

  float *dst_ptr = (float *)slice.get_device_valid_inner();
  auto dst_idx_spec = idx_spec;

  auto launch_args = MatrixReluDeriv{
      .src_ptr = src_ptr,
      .dst_ptr = dst_ptr,
      .shape = shape,
      .src_idx_spec = src_idx_spec,
      .dst_idx_spec = dst_idx_spec,
  };

  device_matrix_relu_deriv(launch_args);
}

