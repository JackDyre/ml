#include "lazy_alloc.cu"
#include "types.h"

#ifndef MATRIX_CU
#define MATRIX_CU

#define ptr_idx(stride, row, col) ((row) * stride + (col))

class Matrix {
private:
  LazyDeviceAllocator allocator;
  Shape shape;

public:
  Matrix(Shape shape)
      : shape(shape), allocator(LazyDeviceAllocator::new_no_alloc(
                          shape.rows * shape.cols * sizeof(float))) {}

  Matrix(Shape shape, LazyDeviceAllocator allocator)
      : shape(shape), allocator(allocator) {}

  static Matrix new_owned_host(float *host_ptr, Shape shape) {
    return Matrix(shape, LazyDeviceAllocator::new_owned_host(
                             static_cast<void *>(host_ptr),
                             shape.rows * shape.cols * sizeof(float)));
  }

  static Matrix new_borrowed_host(float *host_ptr, Shape shape) {
    return Matrix(shape, LazyDeviceAllocator::new_borrowed_host(
                             static_cast<void *>(host_ptr),
                             shape.rows * shape.cols * sizeof(float)));
  }

  static Matrix new_owned_dev(float *dev_ptr, Shape shape) {
    return Matrix(shape, LazyDeviceAllocator::new_owned_dev(
                             static_cast<void *>(dev_ptr),
                             shape.rows * shape.cols * sizeof(float)));
  }

  static Matrix new_borrowed_dev(float *dev_ptr, Shape shape) {
    return Matrix(shape, LazyDeviceAllocator::new_borrowed_dev(
                             static_cast<void *>(dev_ptr),
                             shape.rows * shape.cols * sizeof(float)));
  }

  void host_fill(float val) {
    float *ptr = (float *)allocator.get_host_ptr();

    for (size_t r = 0; r < shape.rows; r++) {
      for (size_t c = 0; c < shape.cols; c++) {
        ptr[ptr_idx(shape.rows, r, c)] = val;
      }
    }
  }

  void dev_fill(float val) {
    float *ptr = (float *)allocator.get_dev_ptr();
    // TODO
  }
};

#endif // MATRIX_CU
