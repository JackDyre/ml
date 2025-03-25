#ifndef NN_H
#define NN_H

#include "layer.h"
#include "matrix.h"
#include <cstddef>
#include <vector>

class NN {
private:
  Matrix input;
  std::vector<Layer> layers;

public:
  NN(std::vector<Layer> layers, Matrix input);
  NN(std::vector<Layer> layers, size_t input_size);

  static NN from_sizes(std::vector<size_t> shapes);

  void forward();
  void set_input(Matrix new_input);
  const Matrix get_output();
};

#endif // !NN_H
