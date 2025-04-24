#ifndef NN_H
#define NN_H

#include "layer.h"
#include "matrix.h"
#include <cstddef>
#include <memory>
#include <vector>

class NN {
private:
  Matrix input;

  std::shared_ptr<Layer> first;
  std::shared_ptr<Layer> last;

  NN(std::shared_ptr<Layer> first, std::shared_ptr<Layer> last, Matrix input)
      : input(input), first(first), last(last) {};

public:
  // Destructor
  ~NN() = default;
  // Copy constructor
  NN(const NN &other) = default;
  // Copy assignment
  NN &operator=(const NN &other) = default;
  // Move constructor
  NN(NN &&other) noexcept = default;
  // Move assignment
  NN &operator=(NN &&other) noexcept = default;

  static NN from_sizes_vec(std::vector<size_t> shapes);
  static NN from_sizes_slice(size_t *shapes, size_t count);

  void forward();
  void back_prop(NN &grad_nn, Matrix &target);
  void set_input(Matrix &new_input);
  const Matrix get_output();
  void random(float low, float high);
  void print();
};

#endif // !NN_H
