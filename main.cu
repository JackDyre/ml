#include "nn.cu"

int main(void) {
  srand(time(0));

  size_t layer_sizes[] = {8, 60, 15, 20};
  NN nn(layer_sizes, sizeof(layer_sizes) / sizeof(size_t));

  nn.dev_fill_rand(-1, 1);

  Matrix input(Shape{.rows = 8, .cols = 1});
  input.dev_fill_rand(-1, 1);
  nn.set_input_unchecked(input);

  nn.dev_forward();
  nn.get_output()->host_print();

  Matrix target_output(Shape{.rows = 20, .cols = 1});
  target_output.dev_fill_rand(-1, 1);
  target_output.host_print();

  float cost = nn.mse_cost(target_output);

  std::cout << cost << std::endl;

  return 0;
}
