#include "nn.cu"

int main(void) {
  // srand(time(0));

  const int input_dim = 2;
  const int output_dim = 1;

  size_t layer_sizes[] = {input_dim, 2, output_dim};
  NN nn(layer_sizes, sizeof(layer_sizes) / sizeof(size_t));
  NN grad_nn(layer_sizes, sizeof(layer_sizes) / sizeof(size_t));

  nn.dev_fill_rand(-1, 1);

  Matrix input(Shape{.rows = input_dim, .cols = 1});
  input.dev_fill_rand(0, 1);
  nn.set_input_unchecked(input);

  nn.dev_forward();

  Matrix target_output(Shape{.rows = output_dim, .cols = 1});
  target_output.dev_fill_rand(0, 1);

  float cost = nn.dev_mse_cost(target_output);
  std::cout << "Cost before training:\n" << cost << std::endl;

  nn.print_weights();

  int n = 100;
  for (int i = 0; i < n; i++) {
    nn.set_input_unchecked(input);
    nn.dev_forward();
    nn.grad(grad_nn, target_output);
    nn.nn_step(grad_nn, .01);
  }

  nn.set_input_unchecked(input);
  nn.dev_forward();
  float new_cost = nn.dev_mse_cost(target_output);
  std::cout << "Cost after training:\n" << new_cost << std::endl;

  nn.print_weights();

  return 0;
}
