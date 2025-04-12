#ifndef MATRIX_H
#define MATRIX_H

#include "dual_slice.h"
#include "types.h"
#include <cstddef>

class Matrix {
private:
  DualSlice slice;
  Shape shape;
  IndexSpec idx_spec = IndexSpec{.stride = shape.rows, .ptr_offset = 0};

  Matrix(DualSlice slice, Shape shape, IndexSpec idx_spec)
      : slice(slice), shape(shape), idx_spec(idx_spec) {}

public:
  Matrix(Shape shape)
      : slice(DualSlice(shape.rows * shape.cols)), shape(shape) {}

  Matrix(std::size_t rows, std::size_t cols)
      : Matrix(Shape{.rows = rows, .cols = cols}) {}

  // Destructor
  ~Matrix() = default;

  // Copy constructor
  Matrix(const Matrix &other) = default;
  // Copy assignment
  Matrix &operator=(const Matrix &other) = default;

  // Move constructor
  Matrix(Matrix &&other) noexcept = default;
  // Move assignment
  Matrix &operator=(Matrix &&other) noexcept = default;

  std::size_t elem_count();
  std::size_t row_count();
  std::size_t col_count();

  void print_h();
  void fill_d(float val);
  void rand_d(float low, float high);
  void add_d(Matrix &l, Matrix &r);
  void mul_d(Matrix &l, Matrix &r);
  void relu_d(Matrix &src);
  void se_d(Matrix &a, Matrix &b);
  void relu_deriv_d(Matrix &src);

};

#endif // !MATRIX_H
