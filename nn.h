#ifndef NN_H
#define NN_H

#include <csignal>
#include <vector>

#include "matrix.h"
#include "old_kernels.h"

using std::vector;

class NN {
private:
  Matrix input;

  vector<Matrix> activations;
  vector<Matrix> weights;
  vector<Matrix> biases;

public:
  NN(size_t *layer_sizes, size_t layer_count);
  void host_fill_rand(float low, float high);
  void dev_fill_rand(float low, float high);
  void set_input_unchecked(Matrix &input);
  void dev_forward();
  void host_forward();
  Matrix *get_output();
  void print_weights();
  void print_activations();
  void print_biases();
  float dev_mse_cost(Matrix &target_output);
  float host_mse_cost(Matrix &target_output);
  void grad(NN &grad_nn, Matrix &target_output);
  void nn_step(NN &grad_nn, float lr);
};

#endif // NN_H