#ifndef MATRIX_H
#define MATRIX_H

#include "dual_slice.h"
#include "types.h"
#include <cstddef>

class Matrix {
private:
  DualSlice slice;
  Shape shape;
  std::size_t stride;

public:
  Matrix(Shape shape);

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
  std::size_t get_stride();

  void print_h();
  void fill_d(float val);
  void rand_d(float low, float high);
  void add_d(Matrix &other);
  void mul_d(Matrix &l, Matrix &r);
};

#endif // !MATRIX_H
