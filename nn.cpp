#include "nn.h"

NN::NN(size_t *layer_sizes, size_t layer_count)
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

void NN::host_fill_rand(float low, float high) {
    for (auto mat : weights) {
      mat.host_fill_rand(low, high);
    }

    for (auto mat : biases) {
      mat.host_fill_rand(low, high);
    }
}

void NN::dev_fill_rand(float low, float high) {
    for (size_t i = 0; i < weights.size(); i++) {
      weights[i].dev_fill_rand(low, high);
    }

    for (size_t i = 0; i < biases.size(); i++) {
      biases[i].dev_fill_rand(low, high);
    }
}

void NN::set_input_unchecked(Matrix &input) {
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

void NN::dev_forward() {
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

void NN::host_forward() {
    for (size_t i = 0; i < activations.size(); i++) {
      if (i == 0) {
        activations[i].host_mul(weights[i], input);
      } else {
        activations[i].host_mul(weights[i], activations[i - 1]);
      }
      activations[i].host_add(biases[i]);

      float *ptr = activations[i].get_host_ptr_unchecked();
      Shape shape = activations[i].get_shape();
      for (size_t r = 0; r < shape.rows; r++) {
        for (size_t c = 0; c < shape.cols; c++) {
          size_t idx = ptr_idx(shape.cols, r, c);
          ptr[idx] = relu(ptr[idx]);
        }
      }
    }
}

Matrix* NN::get_output() { 
    return &activations[activations.size() - 1]; 
}

void NN::print_weights() {
    for (size_t i = 0; i < weights.size(); i++) {
      weights[i].host_print();
    }
}

void NN::print_activations() {
    input.host_print();

    for (size_t i = 0; i < activations.size(); i++) {
      activations[i].host_print();
    }
}

void NN::print_biases() {
    for (size_t i = 0; i < biases.size(); i++) {
      biases[i].host_print();
    }
}

float NN::dev_mse_cost(Matrix &target_output) {
    auto output = get_output();
    float *outp_ptr = output->get_dev_ptr();
    float *target_ptr = target_output.get_dev_ptr();
    
    size_t num_elems = output->elem_count();
    return calculate_dev_mse_cost(outp_ptr, target_ptr, num_elems);
}

float NN::host_mse_cost(Matrix &target_output) {
    auto output = get_output();
    output->to_host();
    float *output_ptr = output->get_host_ptr_unchecked();

    target_output.to_host();
    float *target_ptr = target_output.get_host_ptr_unchecked();

    Shape shape = output->get_shape();
    size_t num_elems = output->elem_count();
    float sum_squared_diff = 0.0f;

    for (size_t r = 0; r < shape.rows; r++) {
      for (size_t c = 0; c < shape.cols; c++) {
        size_t idx = ptr_idx(shape.cols, r, c);
        float diff = target_ptr[idx] - output_ptr[idx];
        sum_squared_diff += diff * diff;
      }
    }

    return sum_squared_diff / num_elems;
}

void NN::grad(NN &grad_nn, Matrix &target_output) {
    // Initialize gradient at 0
    grad_nn.input.dev_fill(0);
    for (size_t l = 0; l < activations.size(); l++) {
      grad_nn.weights[l].dev_fill(0);
      grad_nn.biases[l].dev_fill(0);
      grad_nn.activations[l].dev_fill(0);
    }

    // Perform backpropagation
    for (long int l = static_cast<long int>(activations.size()) - 1; l >= 0;
         l--) {
      float *target_ptr = target_output.get_host_ptr();

      float *g_a = grad_nn.activations[l].get_host_ptr();
      float *g_b = grad_nn.biases[l].get_host_ptr();
      float *g_w = grad_nn.weights[l].get_host_ptr();

      float *g_a_next = (static_cast<size_t>(l) < activations.size() - 1)
                            ? grad_nn.activations[l + 1].get_host_ptr()
                            : NULL;

      float *a_prev =
          (l == 0) ? input.get_host_ptr() : activations[l - 1].get_host_ptr();

      float *a = activations[l].get_host_ptr();
      float *b = biases[l].get_host_ptr();
      float *w = weights[l].get_host_ptr();

      float *w_next = (static_cast<size_t>(l) < activations.size() - 1)
                          ? weights[l + 1].get_host_ptr()
                          : NULL;
      float *b_next = (static_cast<size_t>(l) < activations.size() - 1)
                          ? biases[l + 1].get_host_ptr()
                          : NULL;

      // return;

      for (size_t r = 0; r < activations[l].get_shape().rows; r++) {

        // Gradient of layer activations
        if (static_cast<size_t>(l) == activations.size() - 1) {

          // We are computing the output layer
          size_t idx = ptr_idx(1, r, 0);
          g_a[idx] = 2 * (a[idx] - target_ptr[idx]);
        } else {
          // We are computing non-output activations
          float val = 0;
          for (size_t k = 0; k < activations[l + 1].get_shape().rows; k++) {
            float z_next = b_next[ptr_idx(1, k, 0)];
            for (size_t i = 0; i < activations[l].get_shape().rows; i++) {
              z_next += w_next[ptr_idx(weights[l + 1].get_shape().cols, k, i)] *
                        a[ptr_idx(1, i, 0)];
            }
            val += g_a_next[ptr_idx(1, k, 0)] * d_relu(z_next) *
                   w_next[ptr_idx(weights[l + 1].get_shape().cols, k, r)];
          }
          g_a[ptr_idx(1, r, 0)] = val;
        }

        // Gradient of layer weights
        float z = b[ptr_idx(1, r, 0)];
        size_t prev_size =
            (l == 0 ? input.get_shape() : activations[l - 1].get_shape()).rows;
        for (size_t i = 0; i < prev_size; i++) {
          z += w[ptr_idx(weights[l].get_shape().cols, r, i)] *
               a_prev[ptr_idx(1, i, 0)];
        }

        g_b[ptr_idx(1, r, 0)] = g_a[ptr_idx(1, r, 0)] * d_relu(z);

        size_t w_cols = weights[l].get_shape().cols;
        for (size_t c = 0; c < w_cols; c++) {
          g_w[ptr_idx(w_cols, r, c)] =
              g_a[ptr_idx(1, r, 0)] * d_relu(z) * a_prev[ptr_idx(1, c, 0)];
        }
      }
    }
}

void NN::nn_step(NN &grad_nn, float lr) {
    // Update weights and biases based on gradients
    for (size_t l = 0; l < weights.size(); l++) {
      // Get device pointers (these calls ensure data is on device)
      float *g_w = grad_nn.weights[l].get_dev_ptr();
      float *g_b = grad_nn.biases[l].get_dev_ptr();
      float *w = weights[l].get_dev_ptr();
      float *b = biases[l].get_dev_ptr();

      // Update weights using gradient descent
      Shape w_shape = weights[l].get_shape();
      Shape b_shape = biases[l].get_shape();

      launch_matrix_gradient_step_kernel(w, g_w, lr, w_shape.rows,
                                         w_shape.cols);
      launch_matrix_gradient_step_kernel(b, g_b, lr, b_shape.rows,
                                         b_shape.cols);
    }
}
