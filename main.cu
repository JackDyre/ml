#include "nn.cu"

int main(void) {
  srand(time(0));

  size_t layer_sizes[] = {2, 3, 4};
  NN nn(layer_sizes, sizeof(layer_sizes) / sizeof(size_t));

  nn.dev_fill_rand(-1, 1);

  Matrix input(Shape{.rows = 2, .cols = 1});
  input.dev_fill_rand(-1, 1);
  assert(input.current_device() == DEVICE);
  nn.set_input_unchecked(input);

  nn.dev_forward();
  nn.get_output()->host_print();

  Matrix target_output(Shape{.rows = 4, .cols = 1});
  target_output.dev_fill_rand(-1, 1);
  target_output.host_print();

  float cost = nn.mse_cost(target_output);

  std::cout << cost << std::endl;

  return 0;
}
