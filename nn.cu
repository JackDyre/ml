#include <csignal>
#include <vector>

#include <cub/device/device_reduce.cuh>

#include "matrix.cu"

#ifndef NN_CU
#define NN_CU

using std::vector;

class NN {
private:
  Matrix input;

  vector<Matrix> activations;
  vector<Matrix> weights;
  vector<Matrix> biases;

public:
  NN(size_t *layer_sizes, size_t layer_count)
      : input(Matrix(Shape{.rows = layer_sizes[0], .cols = 1})) {
    // Must have at least input and output;
    assert(layer_count >= 2);

    for (size_t i = 1; i < layer_count; i++) {
      activations.push_back(Matrix(Shape{.rows = layer_sizes[i], .cols = 1}));
      biases.push_back(Matrix(Shape{.rows = layer_sizes[i], .cols = 1}));
      weights.push_back(
          Matrix(Shape{.rows = layer_sizes[i], .cols = layer_sizes[i - 1]}));
    }
  }

  void host_fill_rand(float low, float high) {
    for (auto mat : weights) {
      mat.host_fill_rand(low, high);
    }

    for (auto mat : biases) {
      mat.host_fill_rand(low, high);
    }
  }

  void dev_fill_rand(float low, float high) {
    for (size_t i = 0; i < weights.size(); i++) {
      weights[i].dev_fill_rand(low, high);
    }

    for (size_t i = 0; i < biases.size(); i++) {
      biases[i].dev_fill_rand(low, high);
    }
  }

  void set_input_unchecked(Matrix &input) {
    Device device = input.current_device();
    assert(device != NONE);

    if (device == HOST) {
      float *host_ptr = input.get_host_ptr_unchecked();
      this->input.set_borrowed_host_ptr_unchecked(host_ptr);
    } else if (device == DEVICE) {
      float *dev_ptr = input.get_dev_ptr_unchecked();
      this->input.set_borrowed_dev_ptr_unchecked(dev_ptr);
    }
  }

  void dev_forward() {
    for (size_t i = 0; i < activations.size(); i++) {
      if (i == 0) {
        activations[i].dev_mul(weights[i], input);
      } else {
        activations[i].dev_mul(weights[i], activations[i - 1]);
      }
      activations[i].dev_add(biases[i]);
      launch_matrix_relu_kernel(activations[i].get_dev_ptr_unchecked(),
                                activations[i].get_shape().rows,
                                activations[i].get_shape().cols);
    }
  }

  Matrix *get_output() { return &activations[activations.size() - 1]; }

  void print_weights() {
    for (size_t i = 0; i < weights.size(); i++) {
      weights[i].host_print();
    }
  }

  void print_activations() {
    input.host_print();

    for (size_t i = 0; i < activations.size(); i++) {
      activations[i].host_print();
    }
  }

  void print_biases() {
    for (size_t i = 0; i < biases.size(); i++) {
      biases[i].host_print();
    }
  }

  float mse_cost(Matrix &target_output) {
    // FIXME: Allocation in fn
    // FIXME: handle cuda errors

    float *dst;
    cudaMalloc(&dst, target_output.alloc_size());

    auto output = get_output();
    output->to_dev();
    float *outp_ptr = output->get_dev_ptr_unchecked();

    target_output.to_dev();
    float *target_ptr = target_output.get_dev_ptr_unchecked();

    launch_matrix_mse_kernel(dst, outp_ptr, target_ptr,
                             output->get_shape().rows,
                             output->get_shape().cols);

    size_t num_elems = output->elem_count();

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    float *d_out;
    cudaMalloc(&d_out, sizeof(float));

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dst, d_out,
                           num_elems);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dst, d_out,
                           num_elems);

    float cost;

    cudaMemcpy(&cost, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    return cost / num_elems;
  }

  void grad(NN grad_nn, Matrix &target_output) {
    // UNIMPLEMENTED
  }
};

#endif // NN_CU
