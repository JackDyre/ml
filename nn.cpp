#include "nn.h"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

NN NN::from_sizes_vec(std::vector<size_t> shapes) {
  return NN::from_sizes_slice(shapes.data(), shapes.size());
}

NN NN::from_sizes_slice(size_t *shapes, size_t count) {
  assert(count >= 2);
  auto input_size = shapes[0];

  std::vector<std::shared_ptr<Layer>> layers;
  layers.reserve(count - 1);

  for (size_t i = 0; i < count - 1; i++) {
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

void NN::back_prop(NN &grad_nn, Matrix &target) {
  last->backprop_chain_from_target(*grad_nn.last, target, input);
}

void NN::set_input(Matrix &new_input) { input = std::move(new_input); }

const Matrix NN::get_output() { return last->get_activations(); }

void NN::random(float low, float high) {
  input.rand_d(low, high);
  first->randomize_chain(low, high);
}

void NN::print() {
  std::cout << "Input" << std::endl;
  input.print_h();

  first->print_chain(0);
}
