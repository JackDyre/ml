#include "layer.h"

Layer::Layer(std::size_t size, std::size_t prev_size)
    : Layer(Matrix(size, prev_size), Matrix(size, 1), Matrix(size, 1),
            Matrix(size, 1)) {}

Layer::Layer(Matrix weights, Matrix biases, Matrix preactivatons,
             Matrix activations)
    : weights(weights), biases(biases), preactivatons(preactivatons),
      activations(activations) {}

void Layer::set_prev(Layer *p) { prev = p; }
void Layer::set_next(Layer *n) { next = n; }

const Matrix Layer::get_activations() { return activations; }

void Layer::forward(Matrix &prev_acts) {
  preactivatons.mul_d(weights, prev_acts);
  preactivatons.add_d(biases);
  activations.relu_d(preactivatons);
}

void Layer::forward_chain(Matrix &prev_acts) {
  forward(prev_acts);

  if (next.has_value()) {
    next.value()->forward_chain(activations);
  }
}
