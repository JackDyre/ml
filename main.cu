#include "nn.cu"

int main(void) {
  srand(time(0));

  const int input_dim = 2;
  const int output_dim = 1;

  // const int epoch_count = 10000;

  const float LEARNING_RATE = .01;

  size_t layer_sizes[] = {input_dim, 2, output_dim};
  NN nn(layer_sizes, sizeof(layer_sizes) / sizeof(size_t));
  NN grad_nn(layer_sizes, sizeof(layer_sizes) / sizeof(size_t));

  nn.dev_fill_rand(-1, 1);

  float inp1[2] = {0, 0};
  float out1[1] = {0};

  float inp2[2] = {1, 0};
  float out2[1] = {1};

  float inp3[2] = {0, 1};
  float out3[1] = {1};

  float inp4[2] = {1, 1};
  float out4[1] = {1};

  Shape inp_shape = Shape{.rows = 2, .cols = 1};
  Shape out_shape = Shape{.rows = 1, .cols = 1};

  Matrix inps[] = {Matrix::new_borrowed_host(inp1, inp_shape),
                   Matrix::new_borrowed_host(inp2, inp_shape),
                   Matrix::new_borrowed_host(inp3, inp_shape),
                   Matrix::new_borrowed_host(inp4, inp_shape)};
  Matrix outs[] = {Matrix::new_borrowed_host(out1, out_shape),
                   Matrix::new_borrowed_host(out2, out_shape),
                   Matrix::new_borrowed_host(out3, out_shape),
                   Matrix::new_borrowed_host(out4, out_shape)};

  float cost_before = 0;
  for (int i = 0; i < 4; i++) {
    nn.set_input_unchecked(inps[i]);
    nn.dev_forward();
    cost_before += nn.dev_mse_cost(outs[i]);
  }
  std::cout << "Cost before: " << cost_before << std::endl;

  while (true) {
    // for (int i = 0; i < epoch_count; i++) {
    for (int k = 0; k < 4; k++) {
      nn.set_input_unchecked(inps[k]);
      nn.dev_forward();
      nn.grad(grad_nn, outs[k]);
      nn.nn_step(grad_nn, LEARNING_RATE);
    }

    float cost = 0;
    for (int i = 0; i < 4; i++) {
      nn.set_input_unchecked(inps[i]);
      nn.dev_forward();
      cost += nn.dev_mse_cost(outs[i]);
    }
    std::cout << "Cost: " << cost << std::endl;
  }

  float cost_after = 0;
  for (int i = 0; i < 4; i++) {
    nn.set_input_unchecked(inps[i]);
    nn.dev_forward();
    cost_after += nn.dev_mse_cost(outs[i]);
  }
  std::cout << "Cost before: " << cost_after << std::endl;

  return 0;
}
