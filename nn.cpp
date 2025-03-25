#include "nn.h"
#include <vector>

NN::NN(std::vector<Layer> layers, Matrix input) : input(input), layers(layers) {
  for (size_t i = 0; i < layers.size(); i++) {
    if (i > 0) {
      layers[i].set_prev(&layers[i - 1]);
    }
    if (i < layers.size() - 1) {
      layers[i].set_next(&layers[i + 1]);
    }
  }
}

NN::NN(std::vector<Layer> layers, size_t input_size)
    : NN(layers, Matrix(input_size, 1)) {}

NN NN::from_sizes(std::vector<size_t> shapes) {
  auto input_size = shapes[0];

  std::vector<Layer> layers;

  return NN(layers, input_size);
}

void NN::forward() { layers[0].forward_chain(input); }
void NN::set_input(Matrix new_input) { input = std::move(new_input); }
const Matrix NN::get_output() {
  return layers[layers.size()].get_activations();
}
