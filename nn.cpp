#include "nn.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

NN NN::from_sizes(std::vector<size_t> shapes) {
  assert(shapes.size() >= 2);
  auto input_size = shapes[0];

  std::vector<std::shared_ptr<Layer>> layers;
  layers.reserve(shapes.size() - 1);

  for (size_t i = 0; i < shapes.size() - 1; i++) {
    layers.push_back(std::make_shared<Layer>(Layer(shapes[i + 1], shapes[i])));
  }

  for (size_t i = 0; i < layers.size(); i++) {
    if (i > 0) {
      layers[i]->set_prev(layers[i - 1]);
    }
    if (i < layers.size() - 1) {
      layers[i]->set_next(layers[i + 1]);
    }
  }

  auto first = layers[0];
  auto last = layers[layers.size() - 1];

  return NN(first, last, Matrix(input_size, 1));
}

void NN::forward() { first->forward_chain(input); }

void NN::set_input(Matrix new_input) { input = std::move(new_input); }

const Matrix NN::get_output() { return last->get_activations(); }
