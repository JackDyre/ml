#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "kernels.h"
#include "lazy_alloc.h"
#include "types.h"

#include <cassert>
#include <iomanip>
#include <iostream>

#define ptr_idx(stride, row, col) ((row) * stride + (col))

class Matrix {
private:
  Shape shape;
  LazyDeviceAllocator allocator;

public:
  Matrix(Shape shape);
  Matrix(Shape shape, LazyDeviceAllocator allocator);

  static Matrix new_owned_host(float *host_ptr, Shape shape);
  static Matrix new_borrowed_host(float *host_ptr, Shape shape);
  static Matrix new_owned_dev(float *dev_ptr, Shape shape);
  static Matrix new_borrowed_dev(float *dev_ptr, Shape shape);

  void host_print();
  void host_fill(float val);
  void dev_fill(float val);
  void host_fill_rand(float low, float high);
  void dev_fill_rand(float low, float high);
  void host_add(Matrix &other);
  void dev_add(Matrix &other);
  void host_mul(Matrix &a, Matrix &b);
  void dev_mul(Matrix &a, Matrix &b);

  void set_borrowed_host_ptr_unchecked(float *host_ptr);
  void set_owned_host_ptr_unchecked(float *host_ptr);
  void set_borrowed_dev_ptr_unchecked(float *dev_ptr);
  void set_owned_dev_ptr_unchecked(float *dev_ptr);

  float *get_host_ptr();
  float *get_dev_ptr();
  float *get_host_ptr_unchecked();
  float *get_dev_ptr_unchecked();

  Device current_device();
  Shape get_shape();
  size_t elem_count();

  void to_host();
  void to_dev();
  size_t alloc_size();
};

#endif // MATRIX_HPP
