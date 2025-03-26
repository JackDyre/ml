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
      : first(first), last(last), input(input) {};

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

  static NN from_sizes(std::vector<size_t> shapes);

  void forward();
  void set_input(Matrix new_input);
  const Matrix get_output();
};

#endif // !NN_H
