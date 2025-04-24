#include "layer.h"
#include <cassert>
#include <memory>

void Layer::set_prev(std::shared_ptr<Layer> p) { prev = p; }
void Layer::set_next(std::shared_ptr<Layer> n) { next = n; }

const Matrix Layer::get_activations() { return activations; }

void Layer::forward(Matrix &prev_acts) {
  preactivatons.mul_d(weights, prev_acts);
  preactivatons.add_d(preactivatons, biases);
  activations.relu_d(preactivatons);
}

void Layer::forward_chain(Matrix &prev_acts) {
  forward(prev_acts);

  if (next.has_value()) {
    next.value()->forward_chain(activations);
  }
}

void Layer::backprop(Layer &grad, Matrix &input) {
  auto prev_activations = prev.has_value() ? prev.value()->activations : input;

  auto is_next_layer = next.has_value() && grad.next.has_value();
  if (is_next_layer) {
    grad.activations.act_grad_d(grad.next.value()->biases,
                                next.value()->weights);
  }

  // Bias gradient:
  // dC/dB_i[r, 0] = dC/dA_i[r, 0] * f'(Z_i[r, 0])
  grad.biases.relu_deriv_d(this->preactivatons);
  grad.biases.elem_wise_mul_d(grad.biases, grad.activations);

  // Weight gradient:
  // dC/dW_i[r, c] = dC/dA_i[r, 0] * f'Z_i[r, 0]) * A_{i-1}[c, 0]
  grad.weights.weight_grad_d(grad.biases, prev_activations);
}

void Layer::backprop_chain(Layer &grad, Matrix &input) {
  backprop(grad, input);

  if (prev.has_value() && grad.prev.has_value()) {
    prev.value()->backprop_chain(*grad.prev.value(), input);
  }
}

void Layer::backprop_chain_from_target(Layer &grad, Matrix &target,
                                       Matrix &input) {
  assert(activations.row_count() == grad.activations.row_count());
  assert(activations.col_count() == grad.activations.col_count());
  assert(target.row_count() == activations.row_count());
  assert(target.col_count() == activations.col_count());

  grad.activations.se_deriv_d(activations, target);

  backprop_chain(grad, input);
}

void Layer::randomize(float low, float high) {
  activations.rand_d(low, high);
  weights.rand_d(low, high);
  biases.rand_d(low, high);
}

void Layer::randomize_chain(float low, float high) {
  randomize(low, high);

  if (next.has_value()) {
    next.value()->randomize_chain(low, high);
  }
}

void Layer::print(size_t layer_idx) {
  std::cout << "Layer " << layer_idx << ":" << std::endl;
  std::cout << "Weights" << std::endl;
  weights.print_h();
  std::cout << "Biases" << std::endl;
  biases.print_h();
  std::cout << "Activations" << std::endl;
  activations.print_h();
  std::cout << std::endl;
}

void Layer::print_chain(size_t layer_idx) {
  print(layer_idx);

  if (next.has_value()) {
    next.value()->print_chain(layer_idx + 1);
  }
}
