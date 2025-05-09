#include "old_matrix.h"

Matrix::Matrix(Shape shape)
    : shape(shape), allocator(LazyDeviceAllocator::new_no_alloc(
                        shape.rows * shape.cols * sizeof(float))) {}

Matrix::Matrix(Shape shape, LazyDeviceAllocator allocator)
    : shape(shape), allocator(allocator) {}

Matrix Matrix::new_owned_host(float *host_ptr, Shape shape) {
  return Matrix(shape, LazyDeviceAllocator::new_owned_host(
                           static_cast<void *>(host_ptr),
                           shape.rows * shape.cols * sizeof(float)));
}

Matrix Matrix::new_borrowed_host(float *host_ptr, Shape shape) {
  return Matrix(shape, LazyDeviceAllocator::new_borrowed_host(
                           static_cast<void *>(host_ptr),
                           shape.rows * shape.cols * sizeof(float)));
}

Matrix Matrix::new_owned_dev(float *dev_ptr, Shape shape) {
  return Matrix(shape, LazyDeviceAllocator::new_owned_dev(
                           static_cast<void *>(dev_ptr),
                           shape.rows * shape.cols * sizeof(float)));
}

Matrix Matrix::new_borrowed_dev(float *dev_ptr, Shape shape) {
  return Matrix(shape, LazyDeviceAllocator::new_borrowed_dev(
                           static_cast<void *>(dev_ptr),
                           shape.rows * shape.cols * sizeof(float)));
}

void Matrix::host_print() {
  float *ptr = (float *)allocator.get_host_ptr();

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "[" << std::endl;
  for (size_t r = 0; r < shape.rows; r++) {
    std::cout << "    ";
    for (size_t c = 0; c < shape.cols; c++) {
      std::cout << std::setw(10) << ptr[ptr_idx(shape.cols, r, c)] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
}

void Matrix::host_fill(float val) {
  float *ptr = (float *)allocator.get_host_ptr();

  for (size_t r = 0; r < shape.rows; r++) {
    for (size_t c = 0; c < shape.cols; c++) {
      ptr[ptr_idx(shape.cols, r, c)] = val;
    }
  }
}

void Matrix::dev_fill(float val) {
  float *ptr = (float *)allocator.get_dev_ptr();
  launch_matrix_fill_kernel(ptr, val, shape.rows, shape.cols);
}

void Matrix::host_fill_rand(float low, float high) {
  float *ptr = (float *)allocator.get_host_ptr();

  for (size_t r = 0; r < shape.rows; r++) {
    for (size_t c = 0; c < shape.cols; c++) {
      float val = low + (high - low) * (float)rand() / (float)RAND_MAX;
      ptr[ptr_idx(shape.rows, r, c)] = val;
    }
  }
}

void Matrix::dev_fill_rand(float low, float high) {
  float *ptr = (float *)allocator.get_dev_ptr();

  launch_matrix_fill_rand_kernel(ptr, shape.rows, shape.cols, low, high,
                                 rand());
}

void Matrix::host_add(Matrix &other) {
  assert(shape.rows == other.shape.rows);
  assert(shape.cols == other.shape.cols);

  float *self_ptr = (float *)allocator.get_host_ptr();
  float *other_ptr = (float *)other.allocator.get_host_ptr();

  for (size_t r = 0; r < shape.rows; r++) {
    for (size_t c = 0; c < shape.cols; c++) {
      size_t idx = ptr_idx(shape.cols, r, c);
      self_ptr[idx] += other_ptr[idx];
    }
  }
}

void Matrix::dev_add(Matrix &other) {
  float *self_ptr = (float *)allocator.get_dev_ptr();
  float *other_ptr = (float *)other.allocator.get_dev_ptr();

  launch_matrix_add_kernel(self_ptr, other_ptr, shape.rows, shape.cols);
}

void Matrix::host_mul(Matrix &a, Matrix &b) {
  assert(shape.rows == a.shape.rows);
  assert(shape.cols == b.shape.cols);
  assert(a.shape.cols == b.shape.rows);

  float *self_ptr = (float *)allocator.get_host_ptr();
  float *a_ptr = (float *)a.allocator.get_host_ptr();
  float *b_ptr = (float *)b.allocator.get_host_ptr();

  for (size_t r = 0; r < shape.rows; r++) {
    for (size_t c = 0; c < shape.cols; c++) {
      float sum = 0;
      for (size_t i = 0; i < a.shape.cols; i++) {
        float a_val = a_ptr[ptr_idx(a.shape.cols, r, i)];
        float b_val = b_ptr[ptr_idx(b.shape.cols, i, c)];
        sum += a_val * b_val;
      }
      self_ptr[ptr_idx(shape.cols, r, c)] = sum;
    }
  }
}

void Matrix::dev_mul(Matrix &a, Matrix &b) {
  assert(shape.rows == a.shape.rows);
  assert(shape.cols == b.shape.cols);
  assert(a.shape.cols == b.shape.rows);

  float *self_ptr = (float *)allocator.get_dev_ptr();
  float *a_ptr = (float *)a.allocator.get_dev_ptr();
  float *b_ptr = (float *)b.allocator.get_dev_ptr();

  launch_matrix_mul_kernel(self_ptr, a_ptr, b_ptr, shape.rows, shape.cols,
                           a.shape.cols);
}

void Matrix::set_borrowed_host_ptr_unchecked(float *host_ptr) {
  allocator.free_host();
  allocator.set_host_ptr_unchecked(host_ptr);
  allocator.set_host_state_unchecked(BORROWED_VALID);
  allocator.set_dev_invalid_unchecked();
}

void Matrix::set_owned_host_ptr_unchecked(float *host_ptr) {
  allocator.free_host();
  allocator.set_host_ptr_unchecked(host_ptr);
  allocator.set_host_state_unchecked(OWNED_VALID);
  allocator.set_dev_invalid_unchecked();
}

void Matrix::set_borrowed_dev_ptr_unchecked(float *dev_ptr) {
  allocator.free_dev();
  allocator.set_dev_ptr_unchecked(dev_ptr);
  allocator.set_dev_state_unchecked(BORROWED_VALID);
  allocator.set_host_invalid_unchecked();
}

void Matrix::set_owned_dev_ptr_unchecked(float *dev_ptr) {
  allocator.free_dev();
  allocator.set_dev_ptr_unchecked(dev_ptr);
  allocator.set_dev_state_unchecked(OWNED_VALID);
  allocator.set_host_invalid_unchecked();
}

float *Matrix::get_host_ptr() { return (float *)allocator.get_host_ptr(); }

float *Matrix::get_dev_ptr() { return (float *)allocator.get_dev_ptr(); }

float *Matrix::get_host_ptr_unchecked() {
  return (float *)allocator.get_host_ptr_unchecked();
}
float *Matrix::get_dev_ptr_unchecked() {
  return (float *)allocator.get_dev_ptr_unchecked();
}

Device Matrix::current_device() {
  if (allocator.dev_is_valid()) {
    return Device::DEVICE;
  } else if (allocator.host_is_valid()) {
    return Device::HOST;
  } else {
    return Device::NONE;
  }
}

Shape Matrix::get_shape() { return shape; }

size_t Matrix::elem_count() { return shape.rows * shape.cols; }

void Matrix::to_host() { allocator.ensure_on_host(); }
void Matrix::to_dev() { allocator.ensure_on_dev(); }
size_t Matrix::alloc_size() { return allocator.get_alloc_size(); }
